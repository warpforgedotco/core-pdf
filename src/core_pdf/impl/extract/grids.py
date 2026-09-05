# SPDX-License-Identifier: AGPL-3.0-only
"""Vector ruled-grid geometry and native table construction."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import Any

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch, PageAnalysis
from core_pdf.impl.extract.table_cleanup import internal_cell_text
from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.output.model import Table, TableCell

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
    capture: PageAnalysis,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    page_width = capture.width
    page_height = capture.height
    lines = capture.program.lines
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
