# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any, ClassVar, Self

import numpy

from core_pdf.impl.records import Record
from core_pdf.impl.runtime.array_views import finite_median

frozen_setattr = object.__setattr__


class LayoutRegion(Record):
    __slots__ = ("indexes", "x_start_order", "y_start_order", "y_center_order")

    indexes: numpy.ndarray
    x_start_order: numpy.ndarray
    y_start_order: numpy.ndarray
    y_center_order: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = (
        "indexes",
        "x_start_order",
        "y_start_order",
        "y_center_order",
    )
    __match_args__ = ("indexes", "x_start_order", "y_start_order", "y_center_order")

    def __init__(
        self,
        indexes: numpy.ndarray,
        x_start_order: numpy.ndarray,
        y_start_order: numpy.ndarray,
        y_center_order: numpy.ndarray,
    ) -> None:
        frozen_setattr(self, "indexes", indexes)
        frozen_setattr(self, "x_start_order", x_start_order)
        frozen_setattr(self, "y_start_order", y_start_order)
        frozen_setattr(self, "y_center_order", y_center_order)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"indexes={self.indexes!r}, "
            f"x_start_order={self.x_start_order!r}, "
            f"y_start_order={self.y_start_order!r}, "
            f"y_center_order={self.y_center_order!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.indexes == other.indexes
            and self.x_start_order == other.x_start_order
            and self.y_start_order == other.y_start_order
            and self.y_center_order == other.y_center_order
        )

    def __hash__(self) -> int:
        return hash((self.indexes, self.x_start_order, self.y_start_order, self.y_center_order))

    def __replace__(self, /, **changes: Any) -> Self:
        indexes = changes.pop("indexes", self.indexes)
        x_start_order = changes.pop("x_start_order", self.x_start_order)
        y_start_order = changes.pop("y_start_order", self.y_start_order)
        y_center_order = changes.pop("y_center_order", self.y_center_order)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(indexes, x_start_order, y_start_order, y_center_order)


class LayoutGeometry:
    __slots__ = ("boxes", "x_centers", "y_centers", "heights", "marks", "row_ids")

    boxes: numpy.ndarray
    x_centers: numpy.ndarray
    y_centers: numpy.ndarray
    heights: numpy.ndarray
    marks: numpy.ndarray
    row_ids: numpy.ndarray

    __fields__: ClassVar[tuple[str, ...]] = (
        "boxes",
        "x_centers",
        "y_centers",
        "heights",
        "marks",
        "row_ids",
    )
    __match_args__ = ("boxes", "x_centers", "y_centers", "heights", "marks", "row_ids")

    def __init__(
        self,
        boxes: numpy.ndarray,
        x_centers: numpy.ndarray,
        y_centers: numpy.ndarray,
        heights: numpy.ndarray,
        marks: numpy.ndarray,
        row_ids: numpy.ndarray,
    ) -> None:
        self.boxes = boxes
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.heights = heights
        self.marks = marks
        self.row_ids = row_ids

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"boxes={self.boxes!r}, "
            f"x_centers={self.x_centers!r}, "
            f"y_centers={self.y_centers!r}, "
            f"heights={self.heights!r}, "
            f"marks={self.marks!r}, "
            f"row_ids={self.row_ids!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.boxes == other.boxes
            and self.x_centers == other.x_centers
            and self.y_centers == other.y_centers
            and self.heights == other.heights
            and self.marks == other.marks
            and self.row_ids == other.row_ids
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        boxes = changes.pop("boxes", self.boxes)
        x_centers = changes.pop("x_centers", self.x_centers)
        y_centers = changes.pop("y_centers", self.y_centers)
        heights = changes.pop("heights", self.heights)
        marks = changes.pop("marks", self.marks)
        row_ids = changes.pop("row_ids", self.row_ids)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(boxes, x_centers, y_centers, heights, marks, row_ids)

    @classmethod
    def create(cls, boxes: numpy.ndarray) -> LayoutGeometry:
        x_centers = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y_centers = (boxes[:, 1] + boxes[:, 3]) * 0.5
        return cls(
            boxes=boxes,
            x_centers=x_centers,
            y_centers=y_centers,
            heights=numpy.maximum(1.0, boxes[:, 3] - boxes[:, 1]),
            marks=numpy.zeros(len(boxes), dtype=numpy.bool_),
            row_ids=numpy.empty(len(boxes), dtype=numpy.int64),
        )

    def region(self, indexes: numpy.ndarray, parent: LayoutRegion | None = None) -> LayoutRegion:
        if parent is None:
            return LayoutRegion(
                indexes=indexes,
                x_start_order=indexes[numpy.argsort(self.boxes[indexes, 0], kind="stable")],
                y_start_order=indexes[numpy.argsort(self.boxes[indexes, 1], kind="stable")],
                y_center_order=indexes[numpy.argsort(-self.y_centers[indexes], kind="stable")],
            )
        self.marks[indexes] = True
        region = LayoutRegion(
            indexes=indexes,
            x_start_order=parent.x_start_order[self.marks[parent.x_start_order]],
            y_start_order=parent.y_start_order[self.marks[parent.y_start_order]],
            y_center_order=parent.y_center_order[self.marks[parent.y_center_order]],
        )
        self.marks[indexes] = False
        return region


def projection_gap_from_sorted(
    sorted_starts: numpy.ndarray,
    sorted_ends: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    previous_ends = numpy.maximum.accumulate(sorted_ends)[:-1]
    gaps = sorted_starts[1:] - previous_ends
    if not len(gaps):
        return None
    best_index = (
        len(gaps) - 1 - int(numpy.argmax(gaps[::-1])) if axis == 1 else int(numpy.argmax(gaps))
    )
    best_gap = float(gaps[best_index])
    best_cut = float((sorted_starts[best_index + 1] + previous_ends[best_index]) * 0.5)
    return (best_gap, best_cut) if best_gap >= minimum_gap else None


def best_projection_gap(
    boxes: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    starts = boxes[:, axis]
    ends = boxes[:, axis + 2]
    order = numpy.argsort(starts, kind="stable")
    return projection_gap_from_sorted(starts[order], ends[order], axis, minimum_gap)


def best_region_projection_gap(
    geometry: LayoutGeometry,
    region: LayoutRegion,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    order = region.x_start_order if axis == 0 else region.y_start_order
    return projection_gap_from_sorted(
        geometry.boxes[order, axis],
        geometry.boxes[order, axis + 2],
        axis,
        minimum_gap,
    )


def gutter_tolerating_contained_boxes(
    region_boxes: numpy.ndarray, minimum_gap: float
) -> float | None:
    count = len(region_boxes)
    if count < 8:
        return None
    left = float(region_boxes[:, 0].min())
    right = float(region_boxes[:, 2].max())
    if right - left <= minimum_gap * 2:
        return None
    positions = numpy.linspace(left, right, 256)
    crossing = interval_crossing_counts(region_boxes, positions)
    allowed = max(1, count // 20)
    quiet = crossing <= allowed
    if not quiet.any():
        return None
    padded = numpy.concatenate(([False], quiet, [False]))
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    margin = float(positions[1] - positions[0]) * 0.5 if len(positions) > 1 else 0.0
    runs = [
        (float(positions[run_start]) - margin, float(positions[run_end - 1]) + margin)
        for run_start, run_end in zip(edges[0::2], edges[1::2], strict=True)
        if float(positions[run_start]) > left and float(positions[run_end - 1]) < right
    ]
    if not runs:
        return None

    def unspanned(low: float, high: float) -> bool:
        spanning = (region_boxes[:, 0] < low) & (region_boxes[:, 2] > high)
        return not bool(spanning.any())

    def enclosed(low: float, high: float) -> int:
        inside = (region_boxes[:, 0] >= low) & (region_boxes[:, 2] <= high)
        return int(inside.sum())

    best: tuple[float, float] | None = None
    for index, (low, _high) in enumerate(runs):
        span_high = runs[index][1]
        for next_low, next_high in runs[index + 1 :]:
            if not unspanned(low, next_high) or enclosed(low, next_high) > allowed:
                break
            span_high = next_high
        if (
            unspanned(low, span_high)
            and enclosed(low, span_high) <= allowed
            and span_high - low >= minimum_gap
            and (best is None or span_high - low > best[1] - best[0])
        ):
            best = (low, span_high)
    if best is None:
        return None
    return (best[0] + best[1]) * 0.5


def interval_crossing_counts(boxes: numpy.ndarray, positions: numpy.ndarray) -> numpy.ndarray:
    sorted_starts = numpy.sort(boxes[:, 0])
    sorted_ends = numpy.sort(boxes[:, 2])
    return numpy.searchsorted(sorted_starts, positions, side="left") - numpy.searchsorted(
        sorted_ends, positions, side="right"
    )


def column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    return max(12.0, median_width * 0.05)


def narrow_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    median_height = finite_median(region_boxes[:, 3] - region_boxes[:, 1])
    return max(6.0, min(12.0, median_width * 0.08), median_height * 0.9)


def columnar_split_alignment(region_boxes: numpy.ndarray, cut: float) -> bool:
    centers = (region_boxes[:, 0] + region_boxes[:, 2]) * 0.5
    left = region_boxes[centers < cut]
    right = region_boxes[centers >= cut]
    if len(left) < 4 or len(right) < 4:
        return False

    def aligned(side: numpy.ndarray) -> bool:
        starts = side[:, 0]
        median_start = finite_median(starts)
        return float(numpy.mean(numpy.abs(starts - median_start) <= 3.0)) >= 0.6

    if not (aligned(left) and aligned(right)):
        return False
    left_width = finite_median(left[:, 2] - left[:, 0])
    right_width = finite_median(right[:, 2] - right[:, 0])
    return min(left_width, right_width) >= max(left_width, right_width) * 0.5


def narrow_projection_gap(
    region_boxes: numpy.ndarray,
) -> tuple[float, float] | None:
    narrow_minimum = narrow_column_gap_minimum(region_boxes)
    if narrow_minimum >= column_gap_minimum(region_boxes):
        return None
    narrow = best_projection_gap(region_boxes, 0, narrow_minimum)
    if narrow is None or not columnar_split_alignment(region_boxes, narrow[1]):
        return None
    return narrow


def peel_spanning_band(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    median_height: float,
    *,
    from_bottom: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    if len(indexes) < 4:
        return None
    region = boxes[indexes]
    if from_bottom:
        edges = region[:, 1]
        order = numpy.argsort(edges, kind="stable")
    else:
        edges = region[:, 3]
        order = numpy.argsort(-edges, kind="stable")
    limit = len(indexes) // 2
    taken = 0
    while taken < limit:
        band_edge = float(edges[order[taken]])
        band_end = taken
        while band_end < len(order) and abs(band_edge - float(edges[order[band_end]])) <= max(
            1.0, median_height * 0.5
        ):
            band_end += 1
        if band_end >= len(order):
            return None
        taken = band_end
        remainder_indexes = indexes[order[taken:]]
        if len(remainder_indexes) < 2:
            return None
        remainder = boxes[remainder_indexes]
        gutter = best_projection_gap(remainder, 0, column_gap_minimum(remainder))
        if gutter is None:
            gutter = narrow_projection_gap(remainder)
        if gutter is not None:
            if from_bottom:
                if not columnar_split_alignment(remainder, gutter[1]):
                    continue
                centers = (remainder[:, 0] + remainder[:, 2]) * 0.5
                left_count = int(numpy.count_nonzero(centers < gutter[1]))
                if min(left_count, len(remainder) - left_count) < 18:
                    continue
            return indexes[order[:taken]], remainder_indexes
    return None


def assign_row_bands(
    order: numpy.ndarray,
    centers: numpy.ndarray,
    tolerance: float,
    row_ids: numpy.ndarray,
) -> None:
    current_row = 0
    row_center = float(centers[order[0]])
    for position, raw_item in enumerate(order):
        item = int(raw_item)
        center = float(centers[item])
        if position and row_center - center > tolerance:
            current_row += 1
            row_center = center
        row_ids[item] = current_row


def row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    geometry = LayoutGeometry.create(boxes)
    return row_order_region(geometry, geometry.region(indexes))


def row_order_region(geometry: LayoutGeometry, region: LayoutRegion) -> numpy.ndarray:
    indexes = region.indexes
    if len(indexes) < 2:
        return indexes
    tolerance = max(1.0, finite_median(geometry.heights[indexes]) * 0.5)
    if not math.isfinite(tolerance):
        tolerance = 1.0
    assign_row_bands(
        region.y_center_order,
        geometry.y_centers,
        tolerance,
        geometry.row_ids,
    )
    return indexes[numpy.lexsort((geometry.boxes[indexes, 0], geometry.row_ids[indexes]))]


def partition_by_obstacles(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    used_obstacles: frozenset[int] = frozenset(),
) -> tuple[tuple[numpy.ndarray, ...], int] | None:
    if not obstacles or len(indexes) < 3:
        return None
    region = boxes[indexes]
    region_box = (
        float(numpy.min(region[:, 0])),
        float(numpy.min(region[:, 1])),
        float(numpy.max(region[:, 2])),
        float(numpy.max(region[:, 3])),
    )
    region_width = max(1.0, region_box[2] - region_box[0])
    region_height = max(1.0, region_box[3] - region_box[1])
    centers_x = (region[:, 0] + region[:, 2]) * 0.5
    centers_y = (region[:, 1] + region[:, 3]) * 0.5
    for current_obstacle_index, obstacle in enumerate(obstacles):
        if current_obstacle_index in used_obstacles:
            continue
        x0, y0, x1, y1 = obstacle
        obstacle_width = max(0.0, x1 - x0)
        obstacle_height = max(0.0, y1 - y0)
        if obstacle_width / region_width >= 0.70:
            groups = (
                indexes[centers_y > y1],
                indexes[(centers_y >= y0) & (centers_y <= y1)],
                indexes[centers_y < y0],
            )
        elif obstacle_height / region_height >= 0.70:
            groups = (
                indexes[centers_x < x0],
                indexes[(centers_x >= x0) & (centers_x <= x1)],
                indexes[centers_x > x1],
            )
        else:
            continue
        populated = tuple(group for group in groups if len(group))
        if len(populated) >= 2:
            return populated, current_obstacle_index
    return None


def xy_cut_regions(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    median_height: float,
    *,
    depth: int = 0,
    used_obstacles: frozenset[int] = frozenset(),
    geometry: LayoutGeometry | None = None,
    parent_region: LayoutRegion | None = None,
) -> list[numpy.ndarray]:
    if geometry is None:
        geometry = LayoutGeometry.create(boxes)
    current_region = geometry.region(indexes, parent_region)
    if len(indexes) <= 2 or depth >= 32:
        return [row_order_region(geometry, current_region)]

    def recurse(
        group: numpy.ndarray,
        used: frozenset[int] = used_obstacles,
    ) -> list[numpy.ndarray]:
        return xy_cut_regions(
            group,
            boxes,
            obstacles,
            median_height,
            depth=depth + 1,
            used_obstacles=used,
            geometry=geometry,
            parent_region=current_region,
        )

    obstacle_partition = partition_by_obstacles(
        indexes,
        boxes,
        obstacles,
        used_obstacles,
    )
    if obstacle_partition is not None:
        groups, used_obstacle = obstacle_partition
        next_used_obstacles = used_obstacles | {used_obstacle}
        return [region for group in groups for region in recurse(group, next_used_obstacles)]

    region_boxes = boxes[indexes]
    horizontal = best_region_projection_gap(
        geometry, current_region, 1, max(3.0, median_height * 0.90)
    )
    vertical = best_region_projection_gap(
        geometry, current_region, 0, column_gap_minimum(region_boxes)
    )
    if vertical is None:
        vertical = narrow_projection_gap(region_boxes)
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append((horizontal[0] / median_height * 1.15, 1, horizontal[1]))
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        tolerant_cut = gutter_tolerating_contained_boxes(
            region_boxes, column_gap_minimum(region_boxes)
        )
        if tolerant_cut is not None:
            centers_x = geometry.x_centers[indexes]
            left = indexes[centers_x < tolerant_cut]
            right = indexes[centers_x >= tolerant_cut]
            if len(left) and len(right):
                return [region for group in (left, right) for region in recurse(group)]
        peeled = peel_spanning_band(indexes, boxes, median_height)
        if peeled is not None:
            band, remainder = peeled
            return [
                row_order_region(geometry, geometry.region(band, current_region)),
                *recurse(remainder),
            ]
        peeled = peel_spanning_band(indexes, boxes, median_height, from_bottom=True)
        if peeled is not None:
            band, remainder = peeled
            return [
                *recurse(remainder),
                row_order_region(geometry, geometry.region(band, current_region)),
            ]
        return [row_order_region(geometry, current_region)]

    _, axis, cut = max(candidates, key=lambda item: item[0])
    centers = (geometry.x_centers if axis == 0 else geometry.y_centers)[indexes]
    first = indexes[centers < cut]
    second = indexes[centers >= cut]
    if not len(first) or not len(second):
        return [row_order_region(geometry, current_region)]
    ordered_groups = (second, first) if axis == 1 else (first, second)
    return [region for group in ordered_groups for region in recurse(group)]
