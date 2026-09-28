# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_layout_rules import LAYOUT_RULES
from core_pdf.impl.geometry import array_bbox
from core_pdf.impl.spatial import band_rows
from core_pdf.impl.types import (
    GeneratedRecord,
    RecordType,
    ReplaceFields,
    ReprFields,
)

XY_CUT = LAYOUT_RULES.xy_cut


class LayoutRegion(GeneratedRecord):
    indexes: numpy.ndarray
    x_start_order: numpy.ndarray
    y_start_order: numpy.ndarray
    y_center_order: numpy.ndarray


class LayoutGeometry(ReplaceFields, ReprFields, metaclass=RecordType, frozen=False):
    boxes: numpy.ndarray
    x_centers: numpy.ndarray
    y_centers: numpy.ndarray
    heights: numpy.ndarray
    marks: numpy.ndarray
    row_ids: numpy.ndarray

    __hash__ = None  # type: ignore[assignment]

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
    rules = XY_CUT
    count = len(region_boxes)
    if count < rules.gutter_min_boxes:
        return None
    left = float(region_boxes[:, 0].min())
    right = float(region_boxes[:, 2].max())
    if right - left <= minimum_gap * 2:
        return None
    positions = numpy.linspace(left, right, rules.gutter_samples)
    crossing = interval_crossing_counts(region_boxes, positions)
    allowed = max(1, count // rules.gutter_crossing_divisor)
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

    starts = region_boxes[:, 0]
    ends = region_boxes[:, 2]
    starts_before = [starts < low for low, _high in runs]
    starts_within = [starts >= low for low, _high in runs]
    ends_after = [ends > high for _low, high in runs]
    ends_within = [ends <= high for _low, high in runs]

    def fits(first: int, last: int) -> bool:
        if (starts_before[first] & ends_after[last]).any():
            return False
        return int((starts_within[first] & ends_within[last]).sum()) <= allowed

    best: tuple[float, float] | None = None
    for first, (low, _high) in enumerate(runs):
        last = first
        for following in range(first + 1, len(runs)):
            if not fits(first, following):
                break
            last = following
        span_high = runs[last][1]
        if (
            (last > first or fits(first, first))
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


def median_box_width(region_boxes: numpy.ndarray) -> float | None:
    if not len(region_boxes):
        return None
    return finite_median(region_boxes[:, 2] - region_boxes[:, 0])


def column_gap_minimum(region_boxes: numpy.ndarray, median_width: float | None = None) -> float:
    rules = XY_CUT
    if not len(region_boxes):
        return rules.column_gap_floor
    if median_width is None:
        median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    return max(rules.column_gap_floor, median_width * rules.column_gap_width_ratio)


def narrow_column_gap_minimum(
    region_boxes: numpy.ndarray, median_width: float | None = None
) -> float:
    rules = XY_CUT
    if not len(region_boxes):
        return rules.column_gap_floor
    if median_width is None:
        median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    median_height = finite_median(region_boxes[:, 3] - region_boxes[:, 1])
    return max(
        rules.narrow_gap_floor,
        min(rules.narrow_gap_cap, median_width * rules.narrow_gap_width_ratio),
        median_height * rules.narrow_gap_height_ratio,
    )


def columnar_split_alignment(region_boxes: numpy.ndarray, cut: float) -> bool:
    rules = XY_CUT
    centers = (region_boxes[:, 0] + region_boxes[:, 2]) * 0.5
    left = region_boxes[centers < cut]
    right = region_boxes[centers >= cut]
    minimum = rules.column_alignment_min_boxes
    if len(left) < minimum or len(right) < minimum:
        return False
    tolerance = rules.column_alignment_tolerance
    ratio = rules.column_alignment_ratio

    def aligned(side: numpy.ndarray) -> bool:
        starts = side[:, 0]
        median_start = finite_median(starts)
        return float(numpy.mean(numpy.abs(starts - median_start) <= tolerance)) >= ratio

    if not (aligned(left) and aligned(right)):
        return False
    left_width = finite_median(left[:, 2] - left[:, 0])
    right_width = finite_median(right[:, 2] - right[:, 0])
    return min(left_width, right_width) >= max(left_width, right_width) * rules.column_width_balance


def narrow_projection_gap(
    region_boxes: numpy.ndarray,
    column_minimum: float | None = None,
    median_width: float | None = None,
) -> tuple[float, float] | None:
    """column_minimum and median_width, when given, are column_gap_minimum and
    median_box_width of region_boxes, which the caller already has."""
    if median_width is None:
        median_width = median_box_width(region_boxes)
    narrow_minimum = narrow_column_gap_minimum(region_boxes, median_width)
    if column_minimum is None:
        column_minimum = column_gap_minimum(region_boxes, median_width)
    if narrow_minimum >= column_minimum:
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
    rules = XY_CUT
    if len(indexes) < rules.peel_min_boxes:
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
    band_tolerance = max(rules.peel_band_floor, median_height * rules.peel_band_ratio)
    while taken < limit:
        band_edge = float(edges[order[taken]])
        band_end = taken
        while (
            band_end < len(order)
            and abs(band_edge - float(edges[order[band_end]])) <= band_tolerance
        ):
            band_end += 1
        if band_end >= len(order):
            return None
        taken = band_end
        remainder_indexes = indexes[order[taken:]]
        if len(remainder_indexes) < 2:
            return None
        remainder = boxes[remainder_indexes]
        remainder_width = median_box_width(remainder)
        remainder_minimum = column_gap_minimum(remainder, remainder_width)
        gutter = best_projection_gap(remainder, 0, remainder_minimum)
        if gutter is None:
            gutter = narrow_projection_gap(remainder, remainder_minimum, remainder_width)
        if gutter is not None:
            if from_bottom:
                if not columnar_split_alignment(remainder, gutter[1]):
                    continue
                centers = (remainder[:, 0] + remainder[:, 2]) * 0.5
                left_count = int(numpy.count_nonzero(centers < gutter[1]))
                if min(left_count, len(remainder) - left_count) < rules.peel_min_column_boxes:
                    continue
            return indexes[order[:taken]], remainder_indexes
    return None


def assign_row_bands(
    order: numpy.ndarray,
    centers: numpy.ndarray,
    tolerance: float,
    row_ids: numpy.ndarray,
) -> None:
    bands = band_rows(centers[order].tolist(), tolerance, range(len(order)), linkage="anchor")
    for row, band in enumerate(bands):
        row_ids[order[band]] = row


def row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    geometry = LayoutGeometry.create(boxes)
    return row_order_region(geometry, geometry.region(indexes))


def row_order_region(geometry: LayoutGeometry, region: LayoutRegion) -> numpy.ndarray:
    indexes = region.indexes
    if len(indexes) < 2:
        return indexes
    rules = XY_CUT
    tolerance = max(
        rules.row_band_floor, finite_median(geometry.heights[indexes]) * rules.row_band_ratio
    )
    if not math.isfinite(tolerance):
        tolerance = rules.row_band_floor
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
    region_box = array_bbox(region)
    region_width = max(1.0, region_box[2] - region_box[0])
    region_height = max(1.0, region_box[3] - region_box[1])
    centers_x = (region[:, 0] + region[:, 2]) * 0.5
    centers_y = (region[:, 1] + region[:, 3]) * 0.5
    spanning = LAYOUT_RULES.reading_order.full_width_ratio
    for current_obstacle_index, obstacle in enumerate(obstacles):
        if current_obstacle_index in used_obstacles:
            continue
        x0, y0, x1, y1 = obstacle
        obstacle_width = max(0.0, x1 - x0)
        obstacle_height = max(0.0, y1 - y0)
        if obstacle_width / region_width >= spanning:
            groups = (
                indexes[centers_y > y1],
                indexes[(centers_y >= y0) & (centers_y <= y1)],
                indexes[centers_y < y0],
            )
        elif obstacle_height / region_height >= spanning:
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
    rules = XY_CUT
    if len(indexes) <= 2 or depth >= rules.max_depth:
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
    region_width = median_box_width(region_boxes)
    region_column_minimum = column_gap_minimum(region_boxes, region_width)
    horizontal = best_region_projection_gap(
        geometry,
        current_region,
        1,
        max(rules.horizontal_gap_floor, median_height * rules.horizontal_gap_ratio),
    )
    vertical = best_region_projection_gap(geometry, current_region, 0, region_column_minimum)
    if vertical is None:
        vertical = narrow_projection_gap(region_boxes, region_column_minimum, region_width)
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append(
            (horizontal[0] / median_height * rules.horizontal_preference, 1, horizontal[1])
        )
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        tolerant_cut = gutter_tolerating_contained_boxes(region_boxes, region_column_minimum)
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
