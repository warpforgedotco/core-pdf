# SPDX-License-Identifier: AGPL-3.0-only
"""Geometric page partitioning and local row ordering."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy

from core_pdf.impl.layout.spatial import SpatialIndex
from core_pdf.impl.runtime.array_views import finite_median


@dataclass(frozen=True, slots=True)
class internal_LayoutRegion:
    indexes: numpy.ndarray
    x_start_order: numpy.ndarray
    y_start_order: numpy.ndarray
    y_center_order: numpy.ndarray


@dataclass(slots=True)
class internal_LayoutGeometry:
    boxes: numpy.ndarray
    x_centers: numpy.ndarray
    y_centers: numpy.ndarray
    heights: numpy.ndarray
    marks: numpy.ndarray
    row_ids: numpy.ndarray

    @classmethod
    def create(cls, boxes: numpy.ndarray) -> internal_LayoutGeometry:
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

    def region(
        self, indexes: numpy.ndarray, parent: internal_LayoutRegion | None = None
    ) -> internal_LayoutRegion:
        if parent is None:
            return internal_LayoutRegion(
                indexes=indexes,
                x_start_order=indexes[numpy.argsort(self.boxes[indexes, 0], kind="stable")],
                y_start_order=indexes[numpy.argsort(self.boxes[indexes, 1], kind="stable")],
                y_center_order=indexes[numpy.argsort(-self.y_centers[indexes], kind="stable")],
            )
        self.marks[indexes] = True
        region = internal_LayoutRegion(
            indexes=indexes,
            x_start_order=parent.x_start_order[self.marks[parent.x_start_order]],
            y_start_order=parent.y_start_order[self.marks[parent.y_start_order]],
            y_center_order=parent.y_center_order[self.marks[parent.y_center_order]],
        )
        self.marks[indexes] = False
        return region


def internal_projection_gap_from_sorted(
    sorted_starts: numpy.ndarray,
    sorted_ends: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    previous_ends = numpy.maximum.accumulate(sorted_ends)[:-1]
    gaps = sorted_starts[1:] - previous_ends
    if not len(gaps):
        return None
    # PDF page space is bottom-left based.  For equal horizontal whitespace,
    # split at the uppermost gap first so a full-width header is detached before
    # column detection runs on the body below it.
    best_index = (
        len(gaps) - 1 - int(numpy.argmax(gaps[::-1])) if axis == 1 else int(numpy.argmax(gaps))
    )
    best_gap = float(gaps[best_index])
    best_cut = float((sorted_starts[best_index + 1] + previous_ends[best_index]) * 0.5)
    return (best_gap, best_cut) if best_gap >= minimum_gap else None


def internal_best_projection_gap(
    boxes: numpy.ndarray,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    starts = boxes[:, axis]
    ends = boxes[:, axis + 2]
    order = numpy.argsort(starts, kind="stable")
    return internal_projection_gap_from_sorted(starts[order], ends[order], axis, minimum_gap)


def internal_best_region_projection_gap(
    geometry: internal_LayoutGeometry,
    region: internal_LayoutRegion,
    axis: int,
    minimum_gap: float,
) -> tuple[float, float] | None:
    order = region.x_start_order if axis == 0 else region.y_start_order
    return internal_projection_gap_from_sorted(
        geometry.boxes[order, axis],
        geometry.boxes[order, axis + 2],
        axis,
        minimum_gap,
    )


def internal_gutter_tolerating_contained_boxes(
    region_boxes: numpy.ndarray, minimum_gap: float
) -> float | None:
    """Find a column boundary that almost nothing crosses.

    A projection gap has to be completely empty, so a single mark lying in the
    gutter hides it. That happens constantly on recognised pages: a centred
    page number, a speck of noise, a stray accent. On one patent, four boxes
    26 units wide -- against a median line width of 860 -- hid the boundary
    from 135 lines and interleaved both columns for the whole page.

    Coverage tells the story the projection cannot: it runs at fifty-odd boxes
    across each column and collapses to one or two between them. So look for
    where the fewest boxes cross rather than where none do, and require the
    collapse to be near total, so ordinary text never splits down the middle.
    """
    count = len(region_boxes)
    if count < 8:
        return None
    left = float(region_boxes[:, 0].min())
    right = float(region_boxes[:, 2].max())
    if right - left <= minimum_gap * 2:
        return None
    positions = numpy.linspace(left, right, 256)
    # Count intervals crossing each sample without materializing a 256 x N
    # boolean matrix. Strict inequalities match the former broadcast exactly:
    # starts below the position minus ends at or below the position.
    crossing = internal_interval_crossing_counts(region_boxes, positions)
    # "Almost nothing" has to stay a small share of the region, or a page with
    # few lines would split on a single crossing.
    allowed = max(1, count // 20)
    quiet = crossing <= allowed
    if not quiet.any():
        return None
    # Take the widest quiet run, ignoring the margins outside the text.
    padded = numpy.concatenate(([False], quiet, [False]))
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    # A run is measured between sampled points, so it stops short of the real
    # gutter by up to one step at each end. The sample count is fixed while the
    # region is not, so the wider the region the coarser that step and the more
    # every gutter in it is understated -- on a three-column page a true 17pt
    # gutter measured 11.98 and lost to a 12pt floor. The boundary lies between
    # the last quiet sample and the first busy one, so credit half a step.
    margin = float(positions[1] - positions[0]) * 0.5 if len(positions) > 1 else 0.0
    runs = [
        (float(positions[run_start]) - margin, float(positions[run_end - 1]) + margin)
        for run_start, run_end in zip(edges[0::2], edges[1::2], strict=True)
        if float(positions[run_start]) > left and float(positions[run_end - 1]) < right
    ]
    if not runs:
        return None

    def unspanned(low: float, high: float) -> bool:
        """Does nothing reach across this span into the text on both sides?

        A box poking in from one side is the last word of a line, not a
        heading: only one that starts left of the gutter and ends right of it
        joins the columns together.
        """
        spanning = (region_boxes[:, 0] < low) & (region_boxes[:, 2] > high)
        return not bool(spanning.any())

    def enclosed(low: float, high: float) -> int:
        """How much text sits wholly inside this span?

        Nothing spans a stretch drawn from the gutter on one side of a column to
        the gutter on the other: that column's own lines start and end within it
        and so cross neither edge. ``unspanned`` therefore cannot tell a mark
        left in the gutter from an entire column, and on a three-column page it
        will merge both gutters into one span and cut the middle column in half.
        What is enclosed separates them -- a speck is a handful of boxes, a
        column is most of the region.
        """
        inside = (region_boxes[:, 0] >= low) & (region_boxes[:, 2] <= high)
        return int(inside.sum())

    # A mark sitting in the gutter interrupts the quiet stretch without ending
    # it, so join neighbouring runs while everything crossing the join stays
    # inside the result. A box reaching into the columns on either side stops
    # the merge, which is what keeps ordinary text from splitting mid-line, and
    # so does a join that would swallow a column whole.
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


def internal_interval_crossing_counts(
    boxes: numpy.ndarray, positions: numpy.ndarray
) -> numpy.ndarray:
    """Count horizontal box intervals that strictly cross each position."""
    sorted_starts = numpy.sort(boxes[:, 0])
    sorted_ends = numpy.sort(boxes[:, 2])
    return numpy.searchsorted(sorted_starts, positions, side="left") - numpy.searchsorted(
        sorted_ends, positions, side="right"
    )


def internal_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    """Return the smallest horizontal gap that counts as a column gutter.

    Scaling this by element height, as the row threshold rightly does, is a
    category error: it judges a horizontal distance by a vertical one. It
    survives at line level only because a line's height is close to its text
    size, and breaks completely once the boxes are blocks, where a 141pt tall
    column demanded a 211pt gutter to separate from its neighbour 26pt away.

    A gutter is a horizontal distance, so scale it by one: the width of a
    typical box in the region. The floor keeps ordinary word spacing from
    qualifying on pages whose boxes are narrow.
    """
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    return max(12.0, median_width * 0.05)


def internal_narrow_column_gap_minimum(region_boxes: numpy.ndarray) -> float:
    """Return the narrower gutter minimum a well-aligned split may use.

    Narrow columns draw proportionally narrow gutters: a two-column
    small-print callout with 113pt columns separates them by 11pt, which
    the flat 12pt floor rejects, and the columns interleave line by line.
    A narrower gap is only trustworthy alongside the margin-alignment
    evidence checked by ``internal_columnar_split_alignment`` -- ragged
    recognized text offers same-width gaps purely by accident.

    The absolute floor only means something at native page scale; scanned
    pages measure several units per point, so anchor the floor to the
    region's own line height as well -- no real gutter is narrower than
    the text it separates is tall.
    """
    if not len(region_boxes):
        return 12.0
    median_width = finite_median(region_boxes[:, 2] - region_boxes[:, 0])
    median_height = finite_median(region_boxes[:, 3] - region_boxes[:, 1])
    return max(6.0, min(12.0, median_width * 0.08), median_height * 0.9)


def internal_columnar_split_alignment(region_boxes: numpy.ndarray, cut: float) -> bool:
    """Report whether both sides of a vertical cut read as real columns.

    Genuine columns hang their lines from a shared left margin, so most
    boxes on each side start within a few points of that side's median
    start. Accidental gaps through ragged text leave scattered starts and
    fail this immediately. Twin text columns are also close in width;
    a skinny aligned strip beside a wide one is a marker rail -- claim
    numbers, list bullets, row labels -- whose gap to the body text is
    not a gutter, so require the sides to balance as well.
    """
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


def internal_narrow_projection_gap(
    region_boxes: numpy.ndarray,
) -> tuple[float, float] | None:
    """Find a sub-12pt column gutter backed by margin-alignment evidence."""
    narrow_minimum = internal_narrow_column_gap_minimum(region_boxes)
    if narrow_minimum >= internal_column_gap_minimum(region_boxes):
        return None
    narrow = internal_best_projection_gap(region_boxes, 0, narrow_minimum)
    if narrow is None or not internal_columnar_split_alignment(region_boxes, narrow[1]):
        return None
    return narrow


def internal_peel_spanning_band(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    median_height: float,
    *,
    from_bottom: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """Split an edge row band off a region that admits no projection cut.

    A projection gap only counts when nothing straddles it, so a single
    full-width line -- a heading centred over two columns is the usual case --
    hides the gutter beneath it from the whole region. There is no horizontal
    gap either, because between two columns every row band holds text. The
    region then falls back to row order and the columns interleave for good.

    Such an element cannot take part in a column split, so remove it and let
    the caller re-test what is left. A heading is often set over several lines,
    and a banner may carry a subtitle, so keep taking bands off the edge while
    the columns stay hidden -- stopping once half the region is gone, since
    past that the region is not a heading above columns at all.

    Spanners sit under columns as readily as over them -- a banner or footer
    across the page bottom hides the gutter exactly the same way -- so the
    caller may peel from either edge; it emits a bottom band after the
    remainder instead of before it.

    Returning None when no amount of peeling helps leaves regions that are
    genuinely unsplittable exactly as they were.
    """
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
        # Take everything sharing the outermost remaining band, so a heading
        # set across a line is removed whole rather than a word at a time.
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
        gutter = internal_best_projection_gap(remainder, 0, internal_column_gap_minimum(remainder))
        if gutter is None:
            gutter = internal_narrow_projection_gap(remainder)
        if gutter is not None:
            # Bottom peeling is speculative in a way top peeling is not:
            # receipts and invoices end in totals rows whose removal exposes a
            # label/value gap that reads row-major. Demand real column
            # evidence -- aligned margins, balanced widths, and a paragraph's
            # worth of lines on each side -- before trusting a gutter
            # uncovered from below.
            if from_bottom:
                if not internal_columnar_split_alignment(remainder, gutter[1]):
                    continue
                centers = (remainder[:, 0] + remainder[:, 2]) * 0.5
                left_count = int(numpy.count_nonzero(centers < gutter[1]))
                # A paragraph column runs a couple dozen lines; the aligned
                # key/value panels of invoices and receipts do not.
                if min(left_count, len(remainder) - left_count) < 18:
                    continue
            return indexes[order[:taken]], remainder_indexes
    return None


def internal_assign_row_bands(
    order: numpy.ndarray,
    centers: numpy.ndarray,
    tolerance: float,
    row_ids: numpy.ndarray,
) -> None:
    """Write a row id per box, walking centers top-to-bottom.

    Band by how far each box sits from the row already being built, rather
    than by rounding its absolute position to a grid. Rounding puts the
    boundary at an arbitrary offset, so two cells a point apart can land in
    different rows while two a whole line apart share one -- which scrambles
    the reading order of any table whose cells vary in height.
    """
    current_row = 0
    row_center = float(centers[order[0]])
    for position, raw_item in enumerate(order):
        item = int(raw_item)
        center = float(centers[item])
        if position and row_center - center > tolerance:
            current_row += 1
            row_center = center
        row_ids[item] = current_row


def internal_row_order_indexes(indexes: numpy.ndarray, boxes: numpy.ndarray) -> numpy.ndarray:
    """Order boxes into reading order: row bands top-to-bottom, then left-to-right."""
    geometry = internal_LayoutGeometry.create(boxes)
    return internal_row_order_region(geometry, geometry.region(indexes))


def internal_row_order_region(
    geometry: internal_LayoutGeometry, region: internal_LayoutRegion
) -> numpy.ndarray:
    indexes = region.indexes
    if len(indexes) < 2:
        return indexes
    tolerance = max(1.0, finite_median(geometry.heights[indexes]) * 0.5)
    if not math.isfinite(tolerance):
        tolerance = 1.0
    internal_assign_row_bands(
        region.y_center_order,
        geometry.y_centers,
        tolerance,
        geometry.row_ids,
    )
    return indexes[numpy.lexsort((geometry.boxes[indexes, 0], geometry.row_ids[indexes]))]


def internal_obstacle_partition(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    obstacle_index: SpatialIndex[int] | None = None,
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
    obstacle_indexes: Iterable[int] = (
        obstacle_index.intersecting(region_box)
        if obstacle_index is not None
        else range(len(obstacles))
    )
    for raw_obstacle_index in obstacle_indexes:
        current_obstacle_index = int(raw_obstacle_index)
        if current_obstacle_index in used_obstacles:
            continue
        obstacle = obstacles[current_obstacle_index]
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


def internal_xy_cut_regions(
    indexes: numpy.ndarray,
    boxes: numpy.ndarray,
    obstacles: tuple[tuple[float, float, float, float], ...],
    median_height: float,
    *,
    depth: int = 0,
    obstacle_index: SpatialIndex[int] | None = None,
    used_obstacles: frozenset[int] = frozenset(),
    geometry: internal_LayoutGeometry | None = None,
    parent_region: internal_LayoutRegion | None = None,
) -> list[numpy.ndarray]:
    if geometry is None:
        geometry = internal_LayoutGeometry.create(boxes)
    current_region = geometry.region(indexes, parent_region)
    if len(indexes) <= 2 or depth >= 32:
        return [internal_row_order_region(geometry, current_region)]

    def recurse(
        group: numpy.ndarray,
        used: frozenset[int] = used_obstacles,
    ) -> list[numpy.ndarray]:
        return internal_xy_cut_regions(
            group,
            boxes,
            obstacles,
            median_height,
            depth=depth + 1,
            obstacle_index=obstacle_index,
            used_obstacles=used,
            geometry=geometry,
            parent_region=current_region,
        )

    obstacle_partition = internal_obstacle_partition(
        indexes,
        boxes,
        obstacles,
        obstacle_index,
        used_obstacles,
    )
    if obstacle_partition is not None:
        groups, used_obstacle = obstacle_partition
        next_used_obstacles = used_obstacles | {used_obstacle}
        return [region for group in groups for region in recurse(group, next_used_obstacles)]

    region_boxes = boxes[indexes]
    horizontal = internal_best_region_projection_gap(
        geometry, current_region, 1, max(3.0, median_height * 0.90)
    )
    vertical = internal_best_region_projection_gap(
        geometry, current_region, 0, internal_column_gap_minimum(region_boxes)
    )
    if vertical is None:
        vertical = internal_narrow_projection_gap(region_boxes)
    candidates: list[tuple[float, int, float]] = []
    if horizontal is not None:
        candidates.append((horizontal[0] / median_height * 1.15, 1, horizontal[1]))
    if vertical is not None:
        candidates.append((vertical[0] / median_height, 0, vertical[1]))
    if not candidates:
        tolerant_cut = internal_gutter_tolerating_contained_boxes(
            region_boxes, internal_column_gap_minimum(region_boxes)
        )
        if tolerant_cut is not None:
            centers_x = geometry.x_centers[indexes]
            left = indexes[centers_x < tolerant_cut]
            right = indexes[centers_x >= tolerant_cut]
            if len(left) and len(right):
                return [region for group in (left, right) for region in recurse(group)]
        peeled = internal_peel_spanning_band(indexes, boxes, median_height)
        if peeled is not None:
            band, remainder = peeled
            return [
                internal_row_order_region(geometry, geometry.region(band, current_region)),
                *recurse(remainder),
            ]
        peeled = internal_peel_spanning_band(indexes, boxes, median_height, from_bottom=True)
        if peeled is not None:
            band, remainder = peeled
            return [
                *recurse(remainder),
                internal_row_order_region(geometry, geometry.region(band, current_region)),
            ]
        return [internal_row_order_region(geometry, current_region)]

    _, axis, cut = max(candidates, key=lambda item: item[0])
    centers = (geometry.x_centers if axis == 0 else geometry.y_centers)[indexes]
    first = indexes[centers < cut]
    second = indexes[centers >= cut]
    if not len(first) or not len(second):
        return [internal_row_order_region(geometry, current_region)]
    ordered_groups = (second, first) if axis == 1 else (first, second)
    return [region for group in ordered_groups for region in recurse(group)]
