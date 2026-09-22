import numpy as np
import pytest

from core_pdf.impl.fonts import font_program_truetype as tt


def internal_casteljau(points, t):
    values = list(points)
    while len(values) > 1:
        values = [
            tuple((1 - t) * a + t * b for a, b in zip(left, right, strict=True))
            for left, right in zip(values, values[1:], strict=False)
        ]
    return values[0]


@pytest.mark.parametrize("segments", [1, 2, 6, 8])
@pytest.mark.parametrize("offset", [0, -10, 0.25])
@pytest.mark.parametrize("cubic", [False, True])
def test_curve_flattening_matches_de_casteljau_and_ends_at_final_point(segments, offset, cubic):
    points = [(offset, 0), (offset + 2, 5), (offset + 7, -3)]
    if cubic:
        points.append((offset + 10, 1))
        actual = tt.internal_flatten_cubic(
            points[0], points[1], points[2], points[3], segments=segments
        )
    else:
        actual = tt.internal_flatten_quadratic(points[0], points[1], points[2], segments=segments)
    expected = [internal_casteljau(points, index / segments) for index in range(1, segments + 1)]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert actual[-1] == points[-1]
    assert len(actual) == segments


@pytest.mark.parametrize("end_operator", ["closePath", "endPath", None])
def test_recorded_contours_close_once_and_ignore_unstarted_commands(end_operator):
    recording = [
        ("lineTo", ((99, 99),)),
        ("qCurveTo", ((1, 1), (2, 2))),
        ("curveTo", ((1, 1), (2, 2), (3, 3))),
        ("moveTo", ((0, 0),)),
        ("lineTo", ((3, 0),)),
        ("lineTo", ((1, 3),)),
    ]
    if end_operator is not None:
        recording.extend([(end_operator, ()), (end_operator, ())])
    assert tt.internal_recording_to_contours(recording) == [[(0, 0), (3, 0), (1, 3), (0, 0)]]


def test_new_move_closes_previous_contour_and_discards_single_point_contours():
    recording = [
        ("moveTo", ((99, 99),)),
        ("moveTo", ((0, 0),)),
        ("lineTo", ((1, 1),)),
        ("moveTo", ((3, 3),)),
        ("lineTo", ((4, 4),)),
        ("lineTo", ((3, 3),)),
    ]
    assert tt.internal_recording_to_contours(recording) == [
        [(0, 0), (1, 1), (0, 0)],
        [(3, 3), (4, 4), (3, 3)],
    ]


@pytest.mark.parametrize("implicit_close", [False, True])
def test_quadratic_control_chain_uses_implied_midpoints(implicit_close):
    endpoint = None if implicit_close else (6, 0)
    recording = [("moveTo", ((0, 0),)), ("qCurveTo", ((2, 4), (4, 4), endpoint)), ("closePath", ())]
    (actual,) = tt.internal_recording_to_contours(recording)
    end = (0, 0) if implicit_close else (6, 0)
    expected = (
        [(0, 0)]
        + [internal_casteljau([(0, 0), (2, 4), (3, 4)], index / 6) for index in range(1, 7)]
        + [internal_casteljau([(3, 4), (4, 4), end], index / 6) for index in range(1, 7)]
    )
    if expected[-1] != expected[0]:
        expected.append(expected[0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("operands", [(), ((1, 1),), ((1, 1), (2, 2))])
def test_incomplete_cubic_operands_preserve_current_point(operands):
    contour = [(0.0, 0.0)]
    assert tt.internal_append_cubic(contour, (0, 0), operands) == (0, 0)
    assert contour == [(0, 0)]


@pytest.mark.parametrize(
    ("operands", "start", "expected"),
    [
        ((), (0, 0), (1, 1)),
        ((None,), None, (1, 1)),
        (((2, 2),), (0, 0), (2, 2)),
        ((None,), (0, 0), (0, 0)),
    ],
)
def test_quadratic_empty_and_straight_line_controls(operands, start, expected):
    contour = [(1.0, 1.0)]
    assert tt.internal_append_quadratic(contour, (1, 1), start, operands) == expected
    assert contour[-1] == expected


def test_cubic_chain_starts_each_segment_at_previous_endpoint():
    recording = [
        ("moveTo", ((0, 0),)),
        ("curveTo", ((1, 3), (2, 3), (3, 0), (4, -3), (5, -3), (6, 0))),
    ]
    (actual,) = tt.internal_recording_to_contours(recording)
    assert actual[8] == (3, 0)
    assert actual[16] == (6, 0)
    assert actual[-1] == (0, 0)
    assert len(actual) == 18
