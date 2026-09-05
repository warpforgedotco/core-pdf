from __future__ import annotations

from core_pdf.impl._impl.model.geometry import (
    overlap_ratio_min,
    overlap_ratio_min_exact,
)


def test_overlap_ratio_min_uses_smaller_box_area() -> None:
    assert overlap_ratio_min((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 5.0, 10.0)) == 1.0


def test_overlap_ratio_min_exact_does_not_floor_sub_unit_area() -> None:
    small = (0.0, 0.0, 0.5, 0.5)
    assert overlap_ratio_min(small, small) == 0.25
    assert overlap_ratio_min_exact(small, small) == 1.0
    assert overlap_ratio_min_exact(small, (1.0, 1.0, 2.0, 2.0)) == 0.0
