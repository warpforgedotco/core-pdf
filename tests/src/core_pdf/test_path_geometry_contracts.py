import numpy as np
import pytest

from core_pdf.impl._impl.capture.records import CapturedSubpath
from core_pdf.impl._impl.render import paths


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
    result = paths.internal_dash_subpath(path, (pattern, phase))
    assert [(piece.points[0][axis], piece.points[-1][axis]) for piece in result] == expected
    assert all(not piece.closed for piece in result)
    assert path.points == [point(0), point(10)]


@pytest.mark.parametrize("pattern", [[], [0], [0, 0], [-1, 0]])
def test_empty_or_zero_dash_pattern_preserves_original_subpath(pattern):
    path = CapturedSubpath([(0, 0), (10, 0)])
    assert paths.internal_dash_subpath(path, (pattern, 5))[0] is path


@pytest.mark.parametrize("points", [[], [(0, 0)], [(0, 0), (0, 0)]])
def test_degenerate_dashed_subpaths_have_no_stroked_segments(points):
    assert paths.internal_dash_subpath(CapturedSubpath(points), ([2, 1], 0)) == []


@pytest.mark.parametrize("closed_duplicate", [False, True])
def test_solid_dash_covering_closed_path_retains_closure(closed_duplicate):
    points = [(0, 0), (4, 0), (4, 4), (0, 4)]
    path = CapturedSubpath(points + ([points[0]] if closed_duplicate else []), closed=True)
    (piece,) = paths.internal_dash_subpath(path, ([20, 1], 0))
    assert piece.closed
    assert piece.points == points + [points[0]]


@pytest.mark.parametrize("phase", [0, 1, 2, 10])
def test_collinear_vertices_do_not_restart_dash_phase(phase):
    simple = CapturedSubpath([(0, 0), (10, 0)])
    divided = CapturedSubpath([(0, 0), (1, 0), (1, 0), (4, 0), (7, 0), (10, 0)])

    def intervals(path):
        return [
            (p.points[0], p.points[-1]) for p in paths.internal_dash_subpath(path, ([2, 1], phase))
        ]

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
    actual = paths.internal_signed_area_coverage(edges, 3, 3)
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
    actual = paths.internal_signed_area_coverage(np.empty((0, 4)), width, height)
    assert actual.shape == (max(height, 0), max(width, 0))
    assert not actual.any()


def test_closed_dash_crossing_seam_remains_one_continuous_piece():
    path = CapturedSubpath([(0, 0), (4, 0), (4, 4), (0, 4)], closed=True)
    pieces = paths.internal_dash_subpath(path, ([6, 2], 2))
    assert [piece.points for piece in pieces] == [
        [(0, 2), (0, 0), (4, 0)],
        [(4, 2), (4, 4), (0, 4)],
    ]


@pytest.mark.parametrize("radius", [0, 1, 4])
def test_circle_path_preserves_center_and_radius(radius):
    path = paths.internal_circle_path(3, 5, radius)
    (subpath,) = path.subpaths
    assert subpath.closed
    assert len(subpath.points) == 32
    for x, y in subpath.points:
        assert (x - 3) ** 2 + (y - 5) ** 2 == pytest.approx(radius**2)


@pytest.mark.parametrize("cap", [0, 1, 2])
@pytest.mark.parametrize("opacity", [0, 128, 255])
@pytest.mark.parametrize("supplied_views", [False, True])
def test_line_caps_preserve_shape_independently_of_opacity(cap, opacity, supplied_views):
    pixels = bytearray(8 * 8 * 4)
    target = np.frombuffer(pixels, dtype=np.uint8).reshape(8, 8, 4)
    shape = np.zeros((8, 8), dtype=np.uint8)
    alpha = paths.rasterize_unclipped_line_normal(
        pixels,
        8,
        0,
        8,
        1,
        2,
        4,
        6,
        4,
        2,
        (200, 100, 50, opacity),
        cap,
        (0, 0, 8, 8),
        target_pixels=target if supplied_views else None,
        x_coords=np.arange(8, dtype=np.float64) if supplied_views else None,
        y_coords=np.arange(8, dtype=np.float64) if supplied_views else None,
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
    if cap == 0:
        assert not shape[:, :2].any()
    elif cap == 1:
        assert 0 < shape[3, 1] < 255
    else:
        assert shape[3, 1] == 255


@pytest.mark.parametrize("box", [(0, 0, 0, 8), (0, 0, 8, 0)])
def test_line_rasterization_with_empty_pixel_box_is_a_noop(box):
    pixels = bytearray(8 * 8 * 4)
    assert (
        paths.rasterize_unclipped_line_normal(
            pixels, 8, 0, 8, 1, 2, 4, 6, 4, 2, (200, 100, 50, 255), 0, box
        )
        is None
    )
    assert not any(pixels)


@pytest.mark.parametrize("endpoints", [((2, 4), (6, 4)), ((4, 2), (4, 6)), ((2, 2), (6, 6))])
@pytest.mark.parametrize("reverse", [False, True])
def test_square_cap_coverage_matches_independent_rectangle_sampling(
    endpoints: tuple[tuple[float, float], tuple[float, float]], reverse: bool
) -> None:
    start, end = endpoints[::-1] if reverse else endpoints
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    polygon = [
        (start[0] - ux - uy, start[1] - uy + ux),
        (end[0] + ux - uy, end[1] + uy + ux),
        (end[0] + ux + uy, end[1] + uy - ux),
        (start[0] - ux + uy, start[1] - uy - ux),
    ]

    def inside(x, y):
        signs = [
            (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        ]
        return all(v >= 0 for v in signs) or all(v <= 0 for v in signs)

    expected = np.array(
        [
            [
                round(
                    255
                    * sum(
                        inside(x + sx, 8 - y - sy)
                        for sx in (0.125, 0.375, 0.625, 0.875)
                        for sy in (0.125, 0.375, 0.625, 0.875)
                    )
                    / 16
                )
                for x in range(8)
            ]
            for y in range(8)
        ],
        dtype=np.uint8,
    )
    actual = paths.rasterize_unclipped_line_normal(
        bytearray(8 * 8 * 4),
        8,
        0,
        8,
        1,
        *start,
        *end,
        2,
        (200, 100, 50, 255),
        2,
        (0, 0, 8, 8),
        return_source_alpha=True,
    )
    np.testing.assert_array_equal(actual, expected)
