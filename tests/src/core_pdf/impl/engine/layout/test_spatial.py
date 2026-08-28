from __future__ import annotations

from core_pdf.impl.capture_model.geometry import overlap_ratio_min, overlap_ratio_min_exact
from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)


def test_spatial_index_returns_intersections_in_insertion_order() -> None:
    index = SpatialIndex(
        (
            ("left", (0.0, 0.0, 10.0, 10.0)),
            ("right", (20.0, 0.0, 30.0, 10.0)),
            ("middle", (8.0, 0.0, 22.0, 10.0)),
        )
    )

    assert index.intersecting((5.0, 0.0, 25.0, 10.0)) == ("left", "right", "middle")


def test_spatial_index_uses_strict_positive_area_intersections() -> None:
    index = SpatialIndex(((0, (0.0, 0.0, 10.0, 10.0)), (1, (10.0, 0.0, 20.0, 10.0))))

    assert index.intersecting((10.0, 0.0, 12.0, 10.0)) == (1,)
    assert bbox_intersection_area((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)) == 0.0


def test_spatial_index_handles_large_overflow_boxes_without_duplicates() -> None:
    index = SpatialIndex(
        (
            ("page", (0.0, 0.0, 100.0, 100.0)),
            ("small", (40.0, 40.0, 45.0, 45.0)),
        ),
        target_cell_count=256,
        max_cells_per_item=4,
    )

    assert index.intersecting((42.0, 42.0, 43.0, 43.0)) == ("page", "small")


def test_spatial_index_vectorizes_large_candidate_sets() -> None:
    index = SpatialIndex.from_boxes(
        ((float(i), 0.0, float(i + 2), 2.0) for i in range(64)),
        target_cell_count=1,
    )

    assert index.intersecting((10.0, 0.0, 12.0, 2.0)) == (9, 10, 11)


def test_spatial_index_ignores_empty_and_invalid_boxes() -> None:
    index = SpatialIndex(
        (
            ("empty", (0.0, 0.0, 0.0, 10.0)),
            ("bad", (0.0, 0.0, float("nan"), 10.0)),
            ("valid", (1.0, 1.0, 2.0, 2.0)),
        )
    )

    assert len(index) == 1
    assert index.intersecting((0.0, 0.0, 3.0, 3.0)) == ("valid",)


def test_overlap_ratio_min_uses_smaller_box_area() -> None:
    assert overlap_ratio_min((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 5.0, 10.0)) == 1.0


def test_overlap_ratio_min_exact_does_not_floor_sub_unit_area() -> None:
    small = (0.0, 0.0, 0.5, 0.5)
    assert overlap_ratio_min(small, small) == 0.25
    assert overlap_ratio_min_exact(small, small) == 1.0
    assert overlap_ratio_min_exact(small, (1.0, 1.0, 2.0, 2.0)) == 0.0
