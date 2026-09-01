# SPDX-License-Identifier: AGPL-3.0-only
"""Ruled-grid table construction."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import Any

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.extract.table_cleanup import internal_cell_text, internal_CellTextMemo
from core_pdf.impl.layout.grids import AXIS_TOLERANCE, internal_DisjointSet
from core_pdf.impl.layout.spatial import SpatialIndex
from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.output import Table, TableCell


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
    vertical_index = SpatialIndex(
        (
            (
                index,
                (
                    float(segment[0]) - AXIS_TOLERANCE,
                    float(segment[1]) - AXIS_TOLERANCE,
                    float(segment[0]) + AXIS_TOLERANCE,
                    float(segment[2]) + AXIS_TOLERANCE,
                ),
            )
            for index, segment in enumerate(vertical)
        )
    )
    horizontal_index = SpatialIndex(
        (
            (
                index,
                (
                    float(segment[0]) - AXIS_TOLERANCE,
                    float(segment[2]) - AXIS_TOLERANCE,
                    float(segment[1]) + AXIS_TOLERANCE,
                    float(segment[2]) + AXIS_TOLERANCE,
                ),
            )
            for index, segment in enumerate(horizontal)
        )
    )

    def boundary_present(
        position: float,
        start: float,
        end: float,
        segments: numpy.ndarray[Any, Any],
        index: SpatialIndex,
        coordinate_indexes: tuple[int, int, int],
        query: tuple[float, float, float, float],
    ) -> bool:
        position_index, start_index, end_index = coordinate_indexes
        return any(
            abs(float(segments[segment_index, position_index]) - position) <= AXIS_TOLERANCE
            and float(segments[segment_index, start_index]) <= start + AXIS_TOLERANCE
            and float(segments[segment_index, end_index]) >= end - AXIS_TOLERANCE
            for segment_index in index.intersecting(query)
        )

    def vertical_boundary_present(x: float, y0: float, y1: float) -> bool:
        return boundary_present(
            x,
            y0,
            y1,
            vertical,
            vertical_index,
            (0, 1, 2),
            (x - AXIS_TOLERANCE, y0 - AXIS_TOLERANCE, x + AXIS_TOLERANCE, y1 + AXIS_TOLERANCE),
        )

    def horizontal_boundary_present(y: float, x0: float, x1: float) -> bool:
        return boundary_present(
            y,
            x0,
            x1,
            horizontal,
            horizontal_index,
            (2, 0, 1),
            (x0 - AXIS_TOLERANCE, y - AXIS_TOLERANCE, x1 + AXIS_TOLERANCE, y + AXIS_TOLERANCE),
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
    observation_index: SpatialIndex[int] | None = None,
    *,
    cell_text_cache: internal_CellTextMemo | None = None,
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
    if observation_index is not None:
        candidate_indexes = [
            int(index) for index in observation_index.intersecting((x0, y0, x1, y1))
        ]
    else:
        visible = observations.visible.tolist()
        candidate_indexes = [index for index in range(len(observations)) if visible[index]]
    # Compute centers vectorized (keeping the bbox dtype's rounding) and search
    # edges with bisect on plain lists; scalar numpy searchsorted costs a full
    # dispatch per call.
    candidate_boxes = observations.bbox[candidate_indexes]
    center_xs = ((candidate_boxes[:, 0] + candidate_boxes[:, 2]) * 0.5).tolist()
    center_ys = ((candidate_boxes[:, 1] + candidate_boxes[:, 3]) * 0.5).tolist()
    x_edge_list = x_edges.tolist()
    negated_y_edges = (-y_edges).tolist()
    for index, center_x, center_y in zip(candidate_indexes, center_xs, center_ys):
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
                        cell_text_cache,
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
