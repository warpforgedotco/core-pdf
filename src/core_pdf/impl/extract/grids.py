# SPDX-License-Identifier: AGPL-3.0-only
"""Ruled-grid geometry, table construction, and OCR grid tasks."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from statistics import fmean
from typing import Any

import numpy

from core_pdf.impl.extract.contracts import CapturedPage, ObservationBatch, ObservationSource
from core_pdf.impl.extract.ocr.types import internal_OcrTask, internal_pixel_box_to_page_box
from core_pdf.impl.extract.table_cleanup import internal_cell_text
from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.output import Table, TableCell
from core_pdf.impl.render.model import RasterImage
from core_pdf.impl.runtime.array_views import finite_median

# Shared vector ruling geometry.

AXIS_TOLERANCE = 1.5


TABLE_REGION_GAP = 22.0  # loosened to allow adjacent table regions with modest gaps to merge


def internal_line_coordinate_columns(
    lines: Any,
) -> tuple[
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
]:
    """Materialize ``(x0, y0, x1, y1)`` columns for captured lines."""
    values = tuple(lines)
    coordinates = numpy.fromiter(
        (value for line in values for value in (line.x0, line.y0, line.x1, line.y1)),
        dtype=numpy.float64,
        count=len(values) * 4,
    ).reshape((-1, 4))
    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]


class internal_DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def internal_axis_segments(
    capture: CapturedPage,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    lines = capture.grid_lines
    if not lines:
        empty = numpy.empty((0, 3), dtype=numpy.float32)
        return empty, empty

    # Fold the captured coordinates into columns, then classify and normalize
    # all segments with array operations.
    x0, y0, x1, y1 = internal_line_coordinate_columns(lines)
    horizontal_mask = (numpy.abs(y1 - y0) <= AXIS_TOLERANCE) & (
        numpy.abs(x1 - x0) >= page_width * 0.02
    )
    vertical_mask = (
        ~horizontal_mask
        & (numpy.abs(x1 - x0) <= AXIS_TOLERANCE)
        & (numpy.abs(y1 - y0) >= page_height * 0.015)
    )
    horizontal = numpy.column_stack(
        (numpy.minimum(x0, x1), numpy.maximum(x0, x1), (y0 + y1) * 0.5)
    )[horizontal_mask].astype(numpy.float32, copy=False)
    vertical = numpy.column_stack(((x0 + x1) * 0.5, numpy.minimum(y0, y1), numpy.maximum(y0, y1)))[
        vertical_mask
    ].astype(numpy.float32, copy=False)
    return horizontal.reshape((-1, 3)), vertical.reshape((-1, 3))


def internal_grid_components(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    if len(horizontal) < 2 or len(vertical) < 2:
        return ()
    vertical_x = vertical[:, 0]
    vertical_y0 = vertical[:, 1]
    vertical_y1 = vertical[:, 2]
    pairs: list[tuple[int, int]] = []
    for h_index, segment in enumerate(horizontal):
        matching = numpy.flatnonzero(
            (vertical_x >= float(segment[0]) - AXIS_TOLERANCE)
            & (vertical_x <= float(segment[1]) + AXIS_TOLERANCE)
            & (vertical_y1 >= float(segment[2]) - AXIS_TOLERANCE)
            & (vertical_y0 <= float(segment[2]) + AXIS_TOLERANCE)
        )
        pairs.extend((h_index, int(v_index)) for v_index in matching)
    if not pairs:
        return ()
    disjoint = internal_DisjointSet(len(horizontal) + len(vertical))
    for h_index, v_index in pairs:
        disjoint.union(h_index, len(horizontal) + v_index)
    grouped_h: dict[int, list[int]] = defaultdict(list)
    grouped_v: dict[int, list[int]] = defaultdict(list)
    for index in sorted({h_index for h_index, internal_v_index in pairs}):
        grouped_h[disjoint.find(index)].append(index)
    for index in sorted({v_index for internal_h_index, v_index in pairs}):
        grouped_v[disjoint.find(len(horizontal) + index)].append(index)
    return tuple(
        (horizontal[grouped_h[root]], vertical[grouped_v[root]])
        for root in grouped_h.keys() & grouped_v.keys()
    )


def internal_split_grid_component(
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
) -> tuple[tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]], ...]:
    """Split a connected ruled component into vertically separated table regions."""
    positions = numpy.unique(horizontal[:, 2])
    if len(positions) < 3:
        return ((horizontal, vertical),)
    group_breaks = numpy.empty(len(positions), dtype=numpy.bool_)
    group_breaks[0] = True
    group_breaks[1:] = numpy.diff(positions) > TABLE_REGION_GAP
    position_groups = numpy.cumsum(group_breaks) - 1
    horizontal_groups = position_groups[numpy.searchsorted(positions, horizontal[:, 2])]
    regions = []
    for group_index in range(int(position_groups[-1]) + 1):
        group_positions = positions[position_groups == group_index]
        y0, y1 = group_positions[0], group_positions[-1]
        region_horizontal = horizontal[horizontal_groups == group_index]
        region_vertical = vertical[
            (vertical[:, 1] <= y1 + AXIS_TOLERANCE) & (vertical[:, 2] >= y0 - AXIS_TOLERANCE)
        ]
        if len(region_horizontal) >= 2 and len(region_vertical) >= 2:
            regions.append((region_horizontal, region_vertical))
    return tuple(regions) or ((horizontal, vertical),)


# Ruled-grid table construction.


def internal_cluster_positions(values: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    if values.size == 0:
        return numpy.empty(0, dtype=numpy.float32)
    ordered = numpy.sort(values.astype(numpy.float32, copy=False))
    breaks = numpy.empty(len(ordered), dtype=numpy.bool_)
    breaks[0] = True
    breaks[1:] = numpy.diff(ordered) > AXIS_TOLERANCE
    groups = numpy.cumsum(breaks) - 1
    counts = numpy.bincount(groups)
    sums = numpy.bincount(groups, weights=ordered)
    clustered = (sums / counts).astype(numpy.float32)
    if len(clustered) < 2:
        return clustered
    keep = numpy.empty(len(clustered), dtype=numpy.bool_)
    keep[0] = True
    keep[1:] = numpy.diff(clustered) >= 2.0
    return clustered[keep]


def internal_merge_collinear_segments(
    segments: numpy.ndarray[Any, Any],
    *,
    coordinate: int,
    start: int,
    end: int,
) -> numpy.ndarray[Any, Any]:
    if len(segments) < 2:
        return segments
    order = numpy.lexsort((segments[:, start], segments[:, coordinate]))
    sorted_segs = segments[order]
    diff_coord = (
        numpy.abs(sorted_segs[1:, coordinate] - sorted_segs[:-1, coordinate]) <= AXIS_TOLERANCE
    )
    overlap = sorted_segs[1:, start] <= (sorted_segs[:-1, end] + AXIS_TOLERANCE * 2.0)
    can_merge = diff_coord & overlap
    if not numpy.any(can_merge):
        return sorted_segs.astype(numpy.float32, copy=False)

    merged: list[list[float]] = []
    for values in sorted_segs:
        current = [float(value) for value in values]
        if merged:
            previous = merged[-1]
            if (
                abs(current[coordinate] - previous[coordinate]) <= AXIS_TOLERANCE
                and current[start] <= previous[end] + AXIS_TOLERANCE * 2.0
            ):
                previous[start] = min(previous[start], current[start])
                previous[end] = max(previous[end], current[end])
                previous[coordinate] = (previous[coordinate] + current[coordinate]) * 0.5
                continue
        merged.append(current)
    return numpy.asarray(merged, dtype=numpy.float32).reshape((-1, 3))


def internal_merge_grid_cells(
    rows: list[list[TableCell]],
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
    x_edges: numpy.ndarray[Any, Any],
    y_edges: numpy.ndarray[Any, Any],
) -> list[tuple[TableCell, ...]]:
    """Collapse grid cells across absent rules into row/column-spanning cells."""
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    if row_count == 0 or column_count == 0:
        return []

    def boundary_present(
        position: float,
        start: float,
        end: float,
        segments: numpy.ndarray[Any, Any],
        coordinate_indexes: tuple[int, int, int],
    ) -> bool:
        position_index, start_index, end_index = coordinate_indexes
        return bool(
            numpy.any(
                (numpy.abs(segments[:, position_index] - position) <= AXIS_TOLERANCE)
                & (segments[:, start_index] <= start + AXIS_TOLERANCE)
                & (segments[:, end_index] >= end - AXIS_TOLERANCE)
            )
        )

    def vertical_boundary_present(x: float, y0: float, y1: float) -> bool:
        return boundary_present(
            x,
            y0,
            y1,
            vertical,
            (0, 1, 2),
        )

    def horizontal_boundary_present(y: float, x0: float, x1: float) -> bool:
        return boundary_present(
            y,
            x0,
            x1,
            horizontal,
            (2, 0, 1),
        )

    disjoint = internal_DisjointSet(row_count * column_count)
    for row in range(row_count):
        y0, y1 = float(y_edges[row + 1]), float(y_edges[row])
        for column in range(column_count - 1):
            if not vertical_boundary_present(float(x_edges[column + 1]), y0, y1):
                disjoint.union(row * column_count + column, row * column_count + column + 1)
    for row in range(row_count - 1):
        y = float(y_edges[row + 1])
        for column in range(column_count):
            x0, x1 = float(x_edges[column]), float(x_edges[column + 1])
            if not horizontal_boundary_present(y, x0, x1):
                disjoint.union(row * column_count + column, (row + 1) * column_count + column)

    members: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in range(row_count):
        for column in range(column_count):
            members[disjoint.find(row * column_count + column)].append((row, column))
    merged: list[list[TableCell]] = [[] for _ in range(row_count)]
    for cells in members.values():
        min_row = min(row for row, internal_column in cells)
        max_row = max(row for row, internal_column in cells)
        min_column = min(column for internal_row, column in cells)
        max_column = max(column for internal_row, column in cells)
        source_cells = [rows[row][column] for row, column in cells]
        text = " ".join(cell.text for cell in source_cells if cell.text).strip()
        boxes = [cell.bbox for cell in source_cells if cell.bbox is not None]
        bbox = bbox_union(boxes)
        merged[min_row].append(
            TableCell(
                row=min_row,
                column=min_column,
                text=text,
                row_span=max_row - min_row + 1,
                column_span=max_column - min_column + 1,
                bbox=bbox,
            )
        )
    return [tuple(sorted(row, key=lambda cell: cell.column)) for row in merged]


def internal_table_from_component(
    order: int,
    horizontal: numpy.ndarray[Any, Any],
    vertical: numpy.ndarray[Any, Any],
    observations: ObservationBatch,
) -> Table | None:
    x_edges = internal_cluster_positions(vertical[:, 0])
    y_edges = internal_cluster_positions(horizontal[:, 2])[::-1]
    columns = len(x_edges) - 1
    row_count = len(y_edges) - 1
    if columns < 2 or row_count < 1 or columns * row_count > 1_000:
        return None
    x0, x1 = float(x_edges[0]), float(x_edges[-1])
    y0, y1 = float(y_edges[-1]), float(y_edges[0])
    cell_observations: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_indexes = numpy.flatnonzero(observations.visible)
    candidate_boxes = observations.bbox[candidate_indexes]
    center_xs = ((candidate_boxes[:, 0] + candidate_boxes[:, 2]) * 0.5).tolist()
    center_ys = ((candidate_boxes[:, 1] + candidate_boxes[:, 3]) * 0.5).tolist()
    x_edge_list = x_edges.tolist()
    negated_y_edges = (-y_edges).tolist()
    for index, center_x, center_y in zip(candidate_indexes.tolist(), center_xs, center_ys):
        if not (x0 <= center_x <= x1 and y0 <= center_y <= y1):
            continue
        column = bisect_right(x_edge_list, center_x) - 1
        row = bisect_right(negated_y_edges, -center_y) - 1
        if column == columns:
            column -= 1
        if row == row_count:
            row -= 1
        if 0 <= row < row_count and 0 <= column < columns:
            cell_observations[(row, column)].append(index)
    populated = sum(bool(value) for value in cell_observations.values())
    if populated < 2:
        return None
    density = populated / max(1, columns * row_count)
    # Wide, sparsely populated ruled grids are usually decorative form/layout
    # geometry rather than tables.  Keep narrow sparse tables supported while
    # requiring broad grids to contain enough text to justify table structure.
    # Relax wide-grid rejection slightly: allow wider grids with moderate density.
    # Previously rejected grids with >=6 columns and density < 0.5. Loosen to
    # only reject very wide grids (>=8 columns) with density < 0.4 to capture
    # legitimate tables that are sparse but meaningful.
    # Loosen rejection for wide grids: only reject very wide grids with low density.
    # Increase column threshold and lower density threshold to accept sparser wide tables.
    # Loosen rejection for wide grids further: accept slightly sparser wide tables.
    if columns >= 10 and density < 0.30:
        return None
    rows: list[list[TableCell]] = []
    for row in range(row_count):
        cells = []
        for column in range(columns):
            bbox = (
                float(x_edges[column]),
                float(y_edges[row + 1]),
                float(x_edges[column + 1]),
                float(y_edges[row]),
            )
            cells.append(
                TableCell(
                    row=row,
                    column=column,
                    text=internal_cell_text(
                        observations,
                        cell_observations[(row, column)],
                    ),
                    bbox=bbox,
                )
            )
        rows.append(cells)
    merged_rows = internal_merge_grid_cells(rows, horizontal, vertical, x_edges, y_edges)
    return Table(
        order=order,
        rows=tuple(tuple(row) for row in merged_rows),
        bbox=(x0, y0, x1, y1),
        confidence=1.0,
    )


# Raster ruling detection and cell-level OCR task construction.

internal_GRID_DARK_THRESHOLD = 160
internal_GRID_LINE_MIN_FRACTION = 0.30
internal_GRID_LINE_CLUSTER_PX = 5
internal_GRID_MIN_LINES = 4
internal_GRID_MIN_SPAN_FRACTION = 0.25
internal_GRID_CELL_MIN_PX = 6
internal_GRID_CELL_INSET_PX = 2
internal_GRID_CELL_MIN_INK = 0.003
internal_GRID_MAX_CELLS = 1200
internal_GRID_MIN_CELLS = 12
internal_GRID_CELL_MIN_CONFIDENCE = 50.0
internal_PSM_SINGLE_LINE = 7


def internal_close_row_gaps(mask: numpy.ndarray, gap: int) -> numpy.ndarray:
    """Bridge horizontal gaps up to ``gap`` pixels inside each row.

    Scanned rulings drop out along their length, so a rule reads as many
    short runs. Closing (dilate then erode along the row) reconnects them
    without thickening genuine text into false rules.
    """
    if gap <= 0:
        return mask
    window = gap + 1
    padded = numpy.zeros((mask.shape[0], mask.shape[1] + 2 * window), dtype=numpy.int32)
    padded[:, window:-window] = mask
    sums = numpy.cumsum(padded, axis=1)
    dilated = (sums[:, 2 * window :] - sums[:, : -2 * window]) > 0
    padded2 = numpy.zeros_like(padded)
    padded2[:, window:-window] = dilated
    sums2 = numpy.cumsum(padded2, axis=1)
    return (sums2[:, 2 * window :] - sums2[:, : -2 * window]) >= (2 * window)


def internal_longest_true_runs(mask: numpy.ndarray) -> numpy.ndarray:
    """Return the longest consecutive True run per row of a boolean matrix."""
    height, width = mask.shape
    separated = numpy.zeros((height, width + 2), dtype=numpy.int8)
    separated[:, 1:-1] = mask
    flat = separated.ravel()
    deltas = numpy.diff(flat)
    starts = numpy.flatnonzero(deltas == 1)
    ends = numpy.flatnonzero(deltas == -1)
    longest = numpy.zeros(height, dtype=numpy.int64)
    if len(starts):
        numpy.maximum.at(longest, starts // (width + 2), ends - starts)
    return longest


def internal_cluster_line_positions(
    positions: numpy.ndarray,
    tolerance: int = internal_GRID_LINE_CLUSTER_PX,
) -> list[int]:
    clusters: list[list[int]] = []
    for position in positions.tolist():
        if clusters and position - clusters[-1][-1] <= tolerance:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [int(round(fmean(cluster))) for cluster in clusters]


internal_GRID_STRIP_MIN_FRACTION = 0.5
internal_GRID_STRIP_GAP_PX = 12
internal_GRID_MAX_SKEW = 0.02
internal_GRID_DETECT_POOL = 3


def internal_estimate_ruling_skew(dark: numpy.ndarray) -> float:
    """Estimate page skew from horizontal rule offsets between page thirds."""
    height, width = dark.shape
    strip = width // 3
    if strip < 50:
        return 0.0
    gap = internal_GRID_STRIP_GAP_PX
    left_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, :strip], gap))
    right_runs = internal_longest_true_runs(internal_close_row_gaps(dark[:, -strip:], gap))
    minimum = strip * internal_GRID_STRIP_MIN_FRACTION
    left_lines = internal_cluster_line_positions(numpy.flatnonzero(left_runs >= minimum))
    right_lines = internal_cluster_line_positions(numpy.flatnonzero(right_runs >= minimum))
    if len(left_lines) < 3 or len(right_lines) < 3:
        return 0.0
    baseline = width - strip
    offsets = []
    for line in left_lines:
        nearest = min(right_lines, key=lambda candidate: abs(candidate - line))
        if abs(nearest - line) <= baseline * internal_GRID_MAX_SKEW:
            offsets.append(nearest - line)
    if len(offsets) < 3:
        return 0.0
    return finite_median(numpy.asarray(offsets, dtype=numpy.float64)) / baseline


def internal_vertical_shear(dark: numpy.ndarray, slope: float) -> numpy.ndarray:
    """Shift each column vertically by ``-slope * x`` to straighten h-rules."""
    height, width = dark.shape
    shifts = numpy.round(slope * numpy.arange(width)).astype(numpy.int64)
    rows = numpy.arange(height)[:, None] + shifts[None, :]
    numpy.clip(rows, 0, height - 1, out=rows)
    return numpy.take_along_axis(dark, rows, axis=0)


def internal_detect_ruling_grid(
    image: RasterImage,
) -> tuple[list[int], list[int], numpy.ndarray, float] | None:
    """Find a ruled table grid; return edges, source samples, and measured skew."""
    array = numpy.asarray(image.array())
    if array.ndim == 3 and array.shape[2] >= 3:
        color = array[:, :, :3]
    elif array.ndim == 3 and array.shape[2] == 1:
        color = array[:, :, 0]
    elif array.ndim == 2:
        color = array
    else:
        return None
    height, width = color.shape[:2]
    if height < 100 or width < 100:
        return None
    pool = internal_GRID_DETECT_POOL
    pooled_height = height // pool
    pooled_width = width // pool
    cropped = color[: pooled_height * pool, : pooled_width * pool]
    if cropped.ndim == 3:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool, 3).min(axis=(1, 3, 4))
            < internal_GRID_DARK_THRESHOLD
        )
    else:
        pooled = (
            cropped.reshape(pooled_height, pool, pooled_width, pool).min(axis=(1, 3))
            < internal_GRID_DARK_THRESHOLD
        )
    pooled_height, pooled_width = pooled.shape
    slope = internal_estimate_ruling_skew(pooled)
    straight = internal_vertical_shear(pooled, slope) if slope else pooled
    straight_columns = internal_vertical_shear(pooled.T, -slope) if slope else pooled.T
    gap = max(1, internal_GRID_STRIP_GAP_PX // pool)
    row_runs = internal_longest_true_runs(internal_close_row_gaps(straight, gap))
    column_runs = internal_longest_true_runs(internal_close_row_gaps(straight_columns, gap))
    y_lines = internal_cluster_line_positions(
        numpy.flatnonzero(row_runs >= pooled_width * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    x_lines = internal_cluster_line_positions(
        numpy.flatnonzero(column_runs >= pooled_height * internal_GRID_LINE_MIN_FRACTION),
        tolerance=max(1, internal_GRID_LINE_CLUSTER_PX // pool),
    )
    if len(y_lines) < internal_GRID_MIN_LINES or len(x_lines) < internal_GRID_MIN_LINES:
        return None
    if (
        y_lines[-1] - y_lines[0] < pooled_height * internal_GRID_MIN_SPAN_FRACTION
        or x_lines[-1] - x_lines[0] < pooled_width * internal_GRID_MIN_SPAN_FRACTION
    ):
        return None
    scaled_x = [line * pool + pool // 2 for line in x_lines]
    scaled_y = [line * pool + pool // 2 for line in y_lines]
    return scaled_x, scaled_y, color, slope


def internal_grid_cell_tasks(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
    source_samples: numpy.ndarray,
    slope: float,
) -> tuple[internal_OcrTask, ...]:
    """Build one single-line OCR task per populated ruled cell."""
    if (len(x_lines) - 1) * (len(y_lines) - 1) > internal_GRID_MAX_CELLS:
        return ()
    inset = max(internal_GRID_CELL_INSET_PX, int(round(task.resolution / 40)))
    height, width = source_samples.shape[:2]
    tasks: list[internal_OcrTask] = []
    for row_start, row_end in zip(y_lines, y_lines[1:]):
        if row_end - row_start < internal_GRID_CELL_MIN_PX + 2 * inset:
            continue
        for column_start, column_end in zip(x_lines, x_lines[1:]):
            if column_end - column_start < internal_GRID_CELL_MIN_PX + 2 * inset:
                continue
            center_x = (column_start + column_end) * 0.5
            center_y = (row_start + row_end) * 0.5
            row_shift = int(round(slope * center_x))
            column_shift = -int(round(slope * center_y))
            top = max(0, row_start + inset + row_shift)
            bottom = min(height, row_end - inset + row_shift)
            left = max(0, column_start + inset + column_shift)
            right = min(width, column_end - inset + column_shift)
            if bottom - top < internal_GRID_CELL_MIN_PX or right - left < (
                internal_GRID_CELL_MIN_PX
            ):
                continue
            cell = source_samples[top:bottom, left:right]
            if cell.ndim == 3:
                ink_ratio = float(
                    numpy.count_nonzero(cell.min(axis=2) < internal_GRID_DARK_THRESHOLD)
                ) / (cell.shape[0] * cell.shape[1])
            else:
                ink_ratio = float(numpy.count_nonzero(cell < internal_GRID_DARK_THRESHOLD)) / (
                    cell.shape[0] * cell.shape[1]
                )
            if ink_ratio < internal_GRID_CELL_MIN_INK:
                continue
            tasks.append(
                internal_OcrTask(
                    mode=internal_PSM_SINGLE_LINE,
                    image=task.image,
                    rectangle=(left, top, right - left, bottom - top),
                    page_box=task.page_box,
                    resolution=task.resolution,
                    minimum_confidence=internal_GRID_CELL_MIN_CONFIDENCE,
                )
            )
    return tuple(tasks)


def internal_grid_region_page_box(
    task: internal_OcrTask,
    x_lines: list[int],
    y_lines: list[int],
) -> tuple[float, float, float, float]:
    return internal_pixel_box_to_page_box(
        (x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]),
        task.image.width,
        task.image.height,
        task.page_box,
    )


internal_GRID_MIN_ROWS = 8
internal_GRID_MIN_COLUMNS = 5
internal_GRID_MAX_ROW_HEIGHT_DEVIATION = 0.4
internal_GRID_MAX_STRADDLE_RATIO = 0.25


def internal_grid_is_regular_table(
    grid: tuple[list[int], list[int], numpy.ndarray, float],
    prior: ObservationBatch,
    task: internal_OcrTask,
) -> bool:
    """Distinguish a data table's grid from a form's boxed fields."""
    x_lines, y_lines, _source_samples, slope = grid
    if len(y_lines) - 1 < internal_GRID_MIN_ROWS or len(x_lines) - 1 < internal_GRID_MIN_COLUMNS:
        return False
    heights = numpy.diff(numpy.asarray(y_lines, dtype=numpy.float64))
    median_height = finite_median(heights)
    if median_height <= 0.0:
        return False
    deviation = finite_median(numpy.abs(heights - median_height)) / median_height
    if deviation > internal_GRID_MAX_ROW_HEIGHT_DEVIATION:
        return False
    if not len(prior):
        return True
    page_x0, _page_y0, page_x1, page_y1 = task.page_box
    page_width = page_x1 - page_x0
    scale = task.image.width / max(1e-6, page_width)
    interior = x_lines[1:-1]
    if not interior:
        return True
    grid_box = internal_grid_region_page_box(task, x_lines, y_lines)
    inside = 0
    straddling = 0
    slack = max(2.0, task.image.width * 0.002)
    for box in prior.bbox:
        center_x = float(box[0] + box[2]) * 0.5
        center_y = float(box[1] + box[3]) * 0.5
        if not (grid_box[0] <= center_x <= grid_box[2] and grid_box[1] <= center_y <= grid_box[3]):
            continue
        inside += 1
        pixel_y = (page_y1 - center_y) * scale
        pixel_x0 = (float(box[0]) - page_x0) * scale + slope * pixel_y
        pixel_x1 = (float(box[2]) - page_x0) * scale + slope * pixel_y
        if any(pixel_x0 < line - slack and pixel_x1 > line + slack for line in interior):
            straddling += 1
    if inside < 8:
        return True
    return straddling <= inside * internal_GRID_MAX_STRADDLE_RATIO


def internal_grid_row_observations(
    observations: ObservationBatch,
) -> ObservationBatch:
    """Merge cell reads into one observation per grid row, left to right."""
    if not len(observations):
        return observations
    heights = observations.bbox[:, 3] - observations.bbox[:, 1]
    tolerance = max(2.0, finite_median(heights) * 0.6)
    order = numpy.argsort(-(observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5)
    rows: list[list[int]] = []
    row_center = 0.0
    for index in order.tolist():
        center = float(observations.bbox[index, 1] + observations.bbox[index, 3]) * 0.5
        if rows and abs(row_center - center) <= tolerance:
            rows[-1].append(index)
        else:
            rows.append([index])
            row_center = center
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    for row in rows:
        ordered = sorted(row, key=lambda index: float(observations.bbox[index, 0]))
        texts.append(" ".join(observations.text[index].strip() for index in ordered))
        row_boxes = observations.bbox[ordered]
        boxes.append(
            (
                float(row_boxes[:, 0].min()),
                float(row_boxes[:, 1].min()),
                float(row_boxes[:, 2].max()),
                float(row_boxes[:, 3].max()),
            )
        )
        confidences.append(float(numpy.mean(observations.confidence[ordered])))
    return ObservationBatch.from_columns(
        texts,
        boxes,
        source=ObservationSource.OCR,
        confidence=confidences,
        sequence=range(len(texts)),
        rotation=(0 for _ in texts),
        font_size=(box[3] - box[1] for box in boxes),
        line_break_before=(True for _ in texts),
    )
