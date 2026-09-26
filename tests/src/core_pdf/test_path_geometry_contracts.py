import numpy as np
import pytest

from core_pdf.impl import render_paths as paths
from core_pdf.impl.capture_records import CapturedPath, CapturedSubpath
from core_pdf_cythonized import signed_area_coverage


@pytest.mark.parametrize(
    ("pattern", "phase", "expected"),
    [
        ([2, 1], 0, [(0, 2), (3, 5), (6, 8), (9, 10)]),
        ([2, 1], 1, [(0, 1), (2, 4), (5, 7), (8, 10)]),
        ([2, 1], -1, [(1, 3), (4, 6), (7, 9)]),
        ([2], 0, [(0, 2), (4, 6), (8, 10)]),
        ([0, 2], 0, [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_dashes_follow_length_and_phase_on_each_axis(pattern, phase, expected, axis):
    def point(value):
        return (value, 0) if axis == 0 else (0, value)

    path = CapturedSubpath([point(0), point(10)])
    result = paths.dash_subpath(path, (pattern, phase))
    assert [(piece.points[0][axis], piece.points[-1][axis]) for piece in result] == expected
    assert all(not piece.closed for piece in result)
    assert path.points == [point(0), point(10)]


@pytest.mark.parametrize("pattern", [[], [0], [0, 0], [-1, 0]])
def test_empty_or_zero_dash_pattern_preserves_original_subpath(pattern):
    path = CapturedSubpath([(0, 0), (10, 0)])
    assert paths.dash_subpath(path, (pattern, 5))[0] is path


@pytest.mark.parametrize("points", [[], [(0, 0)], [(0, 0), (0, 0)]])
def test_degenerate_dashed_subpaths_have_no_stroked_segments(points):
    assert paths.dash_subpath(CapturedSubpath(points), ([2, 1], 0)) == []


@pytest.mark.parametrize("closed_duplicate", [False, True])
def test_solid_dash_covering_closed_path_retains_closure(closed_duplicate):
    points = [(0, 0), (4, 0), (4, 4), (0, 4)]
    path = CapturedSubpath(points + ([points[0]] if closed_duplicate else []), closed=True)
    (piece,) = paths.dash_subpath(path, ([20, 1], 0))
    assert piece.closed
    assert piece.points == points + [points[0]]


@pytest.mark.parametrize("phase", [0, 1, 2, 10])
def test_collinear_vertices_do_not_restart_dash_phase(phase):
    simple = CapturedSubpath([(0, 0), (10, 0)])
    divided = CapturedSubpath([(0, 0), (1, 0), (1, 0), (4, 0), (7, 0), (10, 0)])

    def intervals(path):
        return [(p.points[0], p.points[-1]) for p in paths.dash_subpath(path, ([2, 1], phase))]

    assert intervals(simple) == intervals(divided)


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 3, 3),
        (0.25, 0.5, 2.5, 2.75),
        (-1, -1, 2.5, 2.5),
        (-3, 0, -1, 3),
        (4, 0, 5, 3),
        (0, -3, 3, -1),
        (0, 4, 3, 5),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_rectangle_coverage_equals_pixel_intersection_area(box, reverse):
    x0, y0, x1, y1 = box
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if reverse:
        points.reverse()
    edges = np.array([(*a, *b) for a, b in zip(points, points[1:] + points[:1])])
    actual = signed_area_coverage(edges, 3, 3)
    expected = [
        [
            max(0, min(x + 1, x1) - max(x, x0)) * max(0, min(y + 1, y1) - max(y, y0))
            for x in range(3)
        ]
        for y in range(3)
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(("width", "height"), [(0, 3), (3, 0), (-1, 3), (3, -1), (3, 3)])
def test_empty_edge_coverage_respects_target_shape(width, height):
    actual = signed_area_coverage(np.empty((0, 4)), width, height)
    assert actual.shape == (max(height, 0), max(width, 0))
    assert not actual.any()


def test_closed_dash_crossing_seam_remains_one_continuous_piece():
    path = CapturedSubpath([(0, 0), (4, 0), (4, 4), (0, 4)], closed=True)
    pieces = paths.dash_subpath(path, ([6, 2], 2))
    assert [piece.points for piece in pieces] == [
        [(0, 2), (0, 0), (4, 0)],
        [(4, 2), (4, 4), (0, 4)],
    ]


@pytest.mark.parametrize("radius", [0, 1, 4])
def test_circle_path_preserves_center_and_radius(radius):
    path = paths.circle_path(3, 5, radius)
    (subpath,) = path.subpaths
    assert subpath.closed
    assert len(subpath.points) == 32
    for x, y in subpath.points:
        assert (x - 3) ** 2 + (y - 5) ** 2 == pytest.approx(radius**2)


@pytest.mark.parametrize("opacity", [0, 128, 255])
def test_a_butt_capped_line_keeps_its_shape_whatever_its_opacity(opacity):
    pixels = bytearray(8 * 8 * 4)
    target = np.frombuffer(pixels, dtype=np.uint8).reshape(8, 8, 4)
    shape = np.zeros((8, 8), dtype=np.uint8)
    alpha = paths.rasterize_unclipped_line_normal(
        target,
        0,
        8,
        1,
        2,
        4,
        6,
        4,
        2,
        (200, 100, 50, opacity),
        (0, 0, 8, 8),
        return_source_alpha=True,
        source_shape=shape,
    )
    if opacity == 0:
        assert alpha is None
    else:
        assert alpha is not None
        assert alpha[3, 3] == opacity
    assert shape[3, 3] == 255
    np.testing.assert_array_equal(shape, shape[::-1])
    np.testing.assert_array_equal(shape, shape[:, ::-1])
    if alpha is not None:
        np.testing.assert_array_equal(target[:, :, 3], alpha)
    if opacity == 0:
        assert not target.any()
    else:
        np.testing.assert_array_equal(target[3, 3], [200, 100, 50, opacity])
    assert not shape[:, :2].any()


@pytest.mark.parametrize("box", [(0, 0, 0, 8), (0, 0, 8, 0)])
def test_line_rasterization_with_empty_pixel_box_is_a_noop(box):
    target = np.zeros((8, 8, 4), dtype=np.uint8)
    assert (
        paths.rasterize_unclipped_line_normal(
            target, 0, 8, 1, 2, 4, 6, 4, 2, (200, 100, 50, 255), box
        )
        is None
    )
    assert not target.any()


def test_a_deferred_outline_builds_the_same_subpaths_as_an_eager_one():
    xs = np.array([0.0, 4.0, 4.0, 0.0, 1.0, 3.0, 2.0])
    ys = np.array([0.0, 0.0, 4.0, 4.0, 1.0, 1.0, 3.0])
    spans = [(0, 4, True), (4, 7, True)]
    deferred = CapturedPath.deferred_outline(xs, ys, spans)
    eager = CapturedPath(
        [
            CapturedSubpath(list(zip(xs[start:end].tolist(), ys[start:end].tolist())), closed=True)
            for start, end, _closes in spans
        ]
    )
    assert [subpath.points for subpath in deferred.subpaths] == [
        subpath.points for subpath in eager.subpaths
    ]
    assert deferred.bbox() == eager.bbox()
    assert deferred.fill_edges() == eager.fill_edges()


def test_a_deferred_outline_is_an_ordinary_captured_path():
    # The renderer tests for this type by identity rather than with isinstance,
    # so a deferred outline must not be a distinct class.
    path = CapturedPath.deferred_outline(np.array([0.0, 1.0]), np.array([0.0, 1.0]), [(0, 2, True)])
    assert type(path) is CapturedPath


def test_axis_aligned_rect_answers_a_deferred_outline_without_building_points():
    # Two subpaths cannot be a rectangle, and neither can one of five points.
    xs = np.array([0.0, 4.0, 4.0, 0.0, 1.0, 3.0, 2.0])
    ys = np.array([0.0, 0.0, 4.0, 4.0, 1.0, 1.0, 3.0])
    two = CapturedPath.deferred_outline(xs, ys, [(0, 4, True), (4, 7, True)])
    assert two.axis_aligned_rect() is None
    assert two._deferred is not None

    five = CapturedPath.deferred_outline(xs, ys, [(0, 5, True)])
    assert five.axis_aligned_rect() is None
    assert five._deferred is not None


def test_a_deferred_outline_that_really_is_a_rectangle_is_still_recognized():
    xs = np.array([1.0, 5.0, 5.0, 1.0])
    ys = np.array([2.0, 2.0, 7.0, 7.0])
    path = CapturedPath.deferred_outline(xs, ys, [(0, 4, True)])
    assert path.axis_aligned_rect() == (1.0, 2.0, 5.0, 7.0)


def test_a_four_point_deferred_outline_that_is_not_a_rectangle_is_rejected():
    xs = np.array([0.0, 4.0, 5.0, 1.0])
    ys = np.array([0.0, 0.0, 4.0, 4.0])
    path = CapturedPath.deferred_outline(xs, ys, [(0, 4, True)])
    assert path.axis_aligned_rect() is None


def test_reading_subpaths_is_what_clears_the_deferred_state():
    # Reading is the only supported way to fill a deferred path: the class
    # deliberately has no __setattr__ or property, so a direct assignment would
    # leave the spans in place for axis_aligned_rect to keep answering from.
    path = CapturedPath.deferred_outline(np.array([0.0, 1.0]), np.array([0.0, 1.0]), [(0, 2, True)])
    assert path._deferred is not None
    built = path.subpaths
    assert path._deferred is None
    assert path.subpaths is built

    path.subpaths = [CapturedSubpath([(9.0, 9.0), (8.0, 8.0)], closed=False)]
    assert path.subpaths[0].points == [(9.0, 9.0), (8.0, 8.0)]


def test_an_unknown_attribute_still_raises():
    path = CapturedPath.deferred_outline(np.array([0.0, 1.0]), np.array([0.0, 1.0]), [(0, 2, True)])
    with pytest.raises(AttributeError):
        getattr(path, "no_such_attribute")
    assert not hasattr(path, "no_such_attribute")
