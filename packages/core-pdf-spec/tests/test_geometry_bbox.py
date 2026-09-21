import math

import pytest

from core_pdf_spec.s_08_graphics.geometry import transform_bbox

AXIS_ALIGNED = (0.75, 0.0, 0.0, -1.25, 100.0, 200.0)
ROTATED = (0.0, 1.0, -1.0, 0.0, 50.0, 60.0)


def general_formula(
    bbox: tuple[float, float, float, float], matrix: tuple[float, ...]
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    xs = (x0 * a + y0 * c + e, x1 * a + y0 * c + e, x0 * a + y1 * c + e, x1 * a + y1 * c + e)
    ys = (x0 * b + y0 * d + f, x1 * b + y0 * d + f, x0 * b + y1 * d + f, x1 * b + y1 * d + f)
    return (min(xs), min(ys), max(xs), max(ys))


@pytest.mark.parametrize("matrix", [AXIS_ALIGNED, ROTATED, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)])
@pytest.mark.parametrize(
    "bbox",
    [
        (1.5, 2.25, 9.75, 12.5),
        (-4.0, -8.0, -1.0, -2.0),
        (0.0, 0.0, 0.0, 0.0),
        (3.0, 4.0, 3.0, 4.0),
        (9.0, 9.0, 1.0, 1.0),
    ],
)
def test_bbox_transform_matches_the_general_corner_formula(matrix, bbox) -> None:
    assert transform_bbox(bbox, matrix) == general_formula(bbox, matrix)


def test_rotation_is_not_taken_through_the_axis_aligned_path() -> None:
    bbox = (0.0, 0.0, 2.0, 4.0)
    assert transform_bbox(bbox, ROTATED) == (46.0, 60.0, 50.0, 62.0)


def test_axis_aligned_extents_carry_the_sign_of_a_negative_zero_translation() -> None:
    bbox = (0.0, 0.0, 0.0, 1.0)
    matrix = (1.0, 0.0, 0.0, -1.0, 0.0, -0.0)
    result = transform_bbox(bbox, matrix)
    assert result == (0.0, -1.0, 0.0, 0.0)
    assert math.copysign(1.0, result[3]) < 0.0
    assert math.copysign(1.0, general_formula(bbox, matrix)[3]) > 0.0


def test_axis_aligned_path_keeps_an_infinite_extent_off_the_other_axis() -> None:
    bbox = (0.0, math.inf, 0.0, 0.0)
    matrix = (0.75, 0.0, 0.0, 0.75, 100.0, 200.0)
    result = transform_bbox(bbox, matrix)
    assert result[0] == 100.0
    assert result[2] == 100.0
    assert result[3] == math.inf
    assert math.isnan(general_formula(bbox, matrix)[0])


def test_non_axis_aligned_matrices_keep_the_general_formula_for_infinities() -> None:
    bbox = (0.0, math.inf, 0.0, 0.0)
    assert math.isnan(transform_bbox(bbox, ROTATED)[1])
