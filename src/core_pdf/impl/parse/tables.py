# SPDX-License-Identifier: AGPL-3.0-only
"""Grid and stream table detection."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from statistics import fmean
from typing import Any, cast

import numpy

from core_pdf.impl.layout.spatial import (
    SpatialIndex,
)
from core_pdf.impl.model.geometry import (
    bbox_union,
    horizontal_overlap_ratio,
    interval_overlap,
    overlap_ratio_min,
    union_bbox,
)
from core_pdf.impl.parse.grid_geometry import (
    AXIS_TOLERANCE,
    internal_axis_segments,
    internal_DisjointSet,
    internal_grid_components,
    internal_split_grid_component,
)
from core_pdf.impl.parse.model import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
)
from core_pdf.impl.runtime.array_views import finite_median
from core_pdf.impl.structured.model import (
    Table,
    TableAssociatedText,
    TableCell,
    TableColumnBand,
    TableRowBand,
)
from core_pdf.impl.text import (
    collapse_character_spaced,
    collapse_ws,
    is_leader_run,
    strip_edge_leaders,
)

COLUMN_TOLERANCE = 14.0  # loosen tolerance for column edge alignment to reduce split tables
TABLE_MERGE_GAP = 36.0  # further increased to allow modestly wider table merges (conservative)
internal_CHART_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d[\d,./%\-]*|\d[\d,./%\-]*\s+\d+)$")


def internal_chart_cell_texts(text: str) -> tuple[str, ...]:
    """Split dense OCR axis/value lines while keeping prose intact."""
    tokens = tuple(part for part in text.split() if part)
    numeric_count = sum(bool(internal_CHART_NUMERIC_TOKEN.fullmatch(part)) for part in tokens)
    if len(tokens) >= 4 and numeric_count >= 3:
        return tokens
    return (text,)


def extract_chart_table(capture: CapturedPage, observations: ObservationBatch) -> Table | None:
    """Represent OCR text recovered from vector artwork as one chart region.

    Vector charts frequently paint labels and values without table ruling.  The
    normal table detector correctly ignores them, but downstream parsers then
    lose the association between the recovered labels and values.  A compact
    synthetic row gives consumers a structured region while leaving ordinary
    pages untouched.
    """
    if (capture.evidence.uncovered_vector_area or 0.0) < 20_000.0:
        return None
    ocr_indexes = numpy.flatnonzero(observations.source == int(ObservationSource.OCR))
    if len(ocr_indexes) < 3:
        return None

    cells: list[TableCell] = []
    boxes: list[tuple[float, float, float, float]] = []
    seen: set[str] = set()
    column = 0
    for index in sorted(ocr_indexes, key=lambda item: observations.bbox[item, 0]):
        text = observations.text[int(index)].strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in observations.bbox[int(index)]),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        parts = internal_chart_cell_texts(text)
        if len(parts) == 1:
            boxes.append(box)
            cells.append(TableCell(row=0, column=column, text=text, bbox=box))
            column += 1
            continue
        width = (box[2] - box[0]) / len(parts)
        for offset, part in enumerate(parts):
            part_box = (box[0] + width * offset, box[1], box[0] + width * (offset + 1), box[3])
            boxes.append(part_box)
            cells.append(TableCell(row=0, column=column, text=part, bbox=part_box))
            column += 1
    if len(cells) < 3 or not boxes:
        return None
    row_tolerance = max(6.0, float(capture.page.height) * 0.008)
    row_groups: list[tuple[float, list[TableCell]]] = []
    for cell in sorted(
        cells,
        key=lambda item: (-(item.bbox or (0, 0, 0, 0))[1], item.column),
    ):
        cell_box = cell.bbox or (0.0, 0.0, 0.0, 0.0)
        center_y = (cell_box[1] + cell_box[3]) / 2
        group = next(
            (
                candidate
                for candidate in row_groups
                if abs(candidate[0] - center_y) <= row_tolerance
            ),
            None,
        )
        if group is None:
            row_groups.append((center_y, [cell]))
        else:
            group[1].append(cell)
    rows = tuple(
        tuple(
            sorted(
                (replace(cell, row=row_index) for cell in group),
                key=lambda item: item.column,
            )
        )
        for row_index, (internal_center_y, group) in enumerate(row_groups)
    )
    return Table(
        order=-1,
        rows=rows,
        bbox=bbox_union(boxes),
        confidence=0.35,
        metadata={"source": "chart-ocr", "synthetic": True},
    )


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


internal_CellTextMemo = dict[tuple[int, ...], str]


def internal_cell_text(
    observations: ObservationBatch,
    indexes: list[int],
    cache: internal_CellTextMemo | None = None,
) -> str:
    cache_key = tuple(indexes)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    boxes = observations.bbox[indexes]
    centers = ((boxes[:, 1] + boxes[:, 3]) * 0.5).tolist()
    lefts = boxes[:, 0].tolist()
    sequences = observations.sequence[indexes].tolist()
    ordered = sorted(
        range(len(indexes)),
        key=lambda position: (-centers[position], lefts[position], sequences[position]),
    )
    parts = []
    for position in ordered:
        part = collapse_ws(observations.text[indexes[position]])
        if part and not is_leader_run(part):
            parts.append(part)
    result = " ".join(parts)
    if cache is not None:
        cache[cache_key] = result
    return result


def internal_clean_table_cell_leader_runs(text: str) -> str:
    """Drop leader/fill punctuation runs from a table cell.

    Dot and dash leaders are page furniture (ToC fillers, reference-list
    separators, dashed cell rules) that reference text omits.  A cell made up
    entirely of such characters, or a cell ending in a long run of them, is
    stripped so the cell matches the reference reading order.
    """
    if not text:
        return text
    if is_leader_run(text):
        return ""
    if all(ch in "\u25cf\u25e6" for ch in text if not ch.isspace()):
        return ""
    return collapse_ws(strip_edge_leaders(text))


internal_TABLE_SPACED_DIGIT_SEQUENCE_RE = re.compile(r"[\d/.,]+(?: +[\d/.,]+)+")
internal_TABLE_SPACED_DIGIT_ADJACENCY_RE = re.compile(r"\d +\d")


def internal_repair_table_cell_spaced_digits(text: str) -> str:
    """Rejoin letter-spaced numeric/date runs inside a table cell.

    Tracked (letter-spaced) digits split a value such as ``10/19/21`` into
    ``10 /1 9`` and the split ``1 9`` no longer matches the reference
    ``19``.  Rejoin any space-separated run of digit/slash tokens when a
    space separates two digits, which is the tracking signature.  Plain
    ratios such as ``40 / 20`` rejoin into ``40/20`` without changing the
    token multiset because the slash keeps the digit groups apart.
    """
    if not text:
        return text
    if ":" in text or "," in text:
        return text

    def rejoin(match: re.Match[str]) -> str:
        sequence = match.group(0)
        if "/" not in sequence:
            return sequence
        if sum(ch.isdigit() for ch in sequence) < 3:
            return sequence
        if not internal_TABLE_SPACED_DIGIT_ADJACENCY_RE.search(sequence):
            return sequence
        joined = re.sub(r" +", "", sequence)
        groups = [group for group in re.split(r"[/.]", joined) if group]
        if len(groups) < 2 or any(len(group) > 4 for group in groups):
            return sequence
        return joined

    return internal_TABLE_SPACED_DIGIT_SEQUENCE_RE.sub(rejoin, text)


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


def internal_text_rows(observations: ObservationBatch) -> list[list[int]]:
    visible_flags = observations.visible.tolist()
    rotations = observations.rotation.tolist()
    visible = [
        index
        for index, text in enumerate(observations.text)
        if visible_flags[index] and text.strip() and rotations[index] == 0
    ]
    if not visible:
        return []
    # Unbox the sort/grouping columns once; per-element numpy indexing in sort
    # keys costs a scalar box per access.
    bbox = observations.bbox
    all_centers = ((bbox[:, 1] + bbox[:, 3]) * 0.5).tolist()
    all_lefts = bbox[:, 0].tolist()
    all_heights = (bbox[:, 3] - bbox[:, 1]).tolist()
    sequences = observations.sequence.tolist()
    ordered = sorted(
        visible,
        key=lambda index: (-all_centers[index], all_lefts[index], sequences[index]),
    )
    rows: list[list[int]] = []
    centers: list[float] = []
    heights: list[float] = []
    for index in ordered:
        center = all_centers[index]
        height = max(1.0, all_heights[index])
        if rows and abs(center - centers[-1]) <= max(2.0, min(height, heights[-1]) * 0.5):
            count = len(rows[-1])
            rows[-1].append(index)
            centers[-1] = (centers[-1] * count + center) / (count + 1)
            heights[-1] = (heights[-1] * count + height) / (count + 1)
        else:
            rows.append([index])
            centers.append(center)
            heights.append(height)
    return [sorted(row, key=lambda index: (all_lefts[index], sequences[index])) for row in rows]


def internal_row_centers(observations: ObservationBatch, rows: list[list[int]]) -> list[float]:
    """Mean vertical centre of each row, computed once per observation batch."""
    bbox = observations.bbox
    centers = ((bbox[:, 1] + bbox[:, 3]) * 0.5).tolist()
    return [sum(centers[index] for index in row) / len(row) for row in rows]


def internal_aligned_column_clusters(
    observations: ObservationBatch,
    rows: list[list[int]],
    page_width: float,
    *,
    minimum_rows: int = 2,
) -> list[list[tuple[int, int]]]:
    tolerance = max(COLUMN_TOLERANCE, min(24.0, page_width * 0.04))
    all_lefts = observations.bbox[:, 0].tolist()
    all_widths = (observations.bbox[:, 2] - observations.bbox[:, 0]).tolist()
    sequences = observations.sequence.tolist()
    positions = [
        (all_lefts[index], row_index, index) for row_index, row in enumerate(rows) for index in row
    ]
    positions.sort(key=lambda item: (item[0], item[1], sequences[item[2]]))
    clusters: list[list[tuple[int, int]]] = []
    means: list[float] = []
    for x, row_index, index in positions:
        if clusters and abs(x - means[-1]) <= tolerance:
            count = len(clusters[-1])
            clusters[-1].append((row_index, index))
            means[-1] = (means[-1] * count + x) / (count + 1)
        else:
            clusters.append([(row_index, index)])
            means.append(x)
    candidates = []
    for cluster in clusters:
        row_support = {row_index for row_index, internal_index in cluster}
        widths = [all_widths[index] for internal_row_index, index in cluster]
        alphanumeric = sum(
            any(character.isalnum() for character in observations.text[index])
            for internal_row_index, index in cluster
        )
        if (
            len(row_support) >= minimum_rows
            and finite_median(numpy.asarray(widths, dtype=numpy.float32)) <= page_width * 0.48
            and alphanumeric * 2 >= len(cluster)
        ):
            candidates.append(cluster)
    return candidates


def internal_split_support_rows(
    observations: ObservationBatch,
    rows: list[list[int]],
    indexes: list[int],
    *,
    minimum_rows: int = 3,
    row_centers: list[float] | None = None,
) -> list[list[int]]:
    if not indexes:
        return []
    if row_centers is None:
        row_centers = internal_row_centers(observations, rows)
    groups = [[indexes[0]]]
    for index in indexes[1:]:
        previous = groups[-1][-1]
        previous_height = max(
            float(observations.bbox[item, 3] - observations.bbox[item, 1])
            for item in rows[previous]
        )
        current_height = max(
            float(observations.bbox[item, 3] - observations.bbox[item, 1]) for item in rows[index]
        )
        allowed_gap = max(55.0, max(previous_height, current_height) * 4.0)
        if row_centers[previous] - row_centers[index] > allowed_gap:
            groups.append([])
        groups[-1].append(index)
    return [group for group in groups if len(group) >= minimum_rows]


def internal_stream_table(
    order: int,
    observations: ObservationBatch,
    rows: list[list[int]],
    support: list[int],
    columns: list[list[tuple[int, int]]],
    *,
    minimum_rows: int = 3,
    cell_text_cache: internal_CellTextMemo | None = None,
    row_centers: list[float] | None = None,
) -> Table | None:
    support_set = set(support)
    columns = [
        column
        for column in columns
        if len({row_index for row_index, internal_index in column}.intersection(support_set))
        >= minimum_rows
    ]
    if len(columns) < 2:
        return None
    all_x0 = observations.bbox[:, 0].tolist()
    all_y0 = observations.bbox[:, 1].tolist()
    all_x1 = observations.bbox[:, 2].tolist()
    all_y1 = observations.bbox[:, 3].tolist()
    column_centers = numpy.asarray(
        [
            finite_median(
                numpy.asarray(
                    [all_x0[index] for internal_row_index, index in column],
                    dtype=numpy.float32,
                )
            )
            for column in columns
        ],
        dtype=numpy.float32,
    )
    column_order = numpy.argsort(column_centers)
    column_centers = column_centers[column_order]
    columns = [columns[int(index)] for index in column_order]
    if row_centers is None:
        row_centers = internal_row_centers(observations, rows)
    top = row_centers[support[0]]
    bottom = row_centers[support[-1]]
    selected = [
        row
        for row, center in zip(rows, row_centers, strict=True)
        if bottom - 1.0 <= center <= top + 1.0
    ]
    if len(selected) < minimum_rows:
        return None
    edges = numpy.empty(len(columns) + 1, dtype=numpy.float32)
    edges[1:-1] = (column_centers[:-1] + column_centers[1:]) * 0.5
    edges[0] = min(all_x0[index] for row in selected for index in row)
    edges[-1] = max(
        all_x1[index]
        for column in columns
        for row_index, index in column
        if support[0] <= row_index <= support[-1]
    )
    if numpy.any(numpy.diff(edges) <= 2.0):
        return None

    edge_list = edges.tolist()
    right_edge_limit = edge_list[-1] + COLUMN_TOLERANCE
    column_count = len(columns)
    table_rows: list[tuple[TableCell, ...]] = []
    populated = 0
    numeric_by_column = [0] * column_count
    text_lengths = 0
    for row_index, row in enumerate(selected):
        cells: list[list[int]] = [[] for internal_column in columns]
        for index in row:
            x0 = all_x0[index]
            x1 = all_x1[index]
            x_center = (x0 + x1) * 0.5
            column = bisect_right(edge_list, x_center) - 1
            if not (0 <= column < column_count):
                # Fallback to max interval overlap or left-edge proximity
                best_col = 0
                max_ov = -1.0
                for c_idx in range(column_count):
                    ov = interval_overlap(x0, x1, edge_list[c_idx], edge_list[c_idx + 1])
                    if ov > max_ov:
                        max_ov = ov
                        best_col = c_idx
                if max_ov > 0.0:
                    column = best_col
                elif column < 0 and x0 <= right_edge_limit:
                    column = 0
            if 0 <= column < column_count and x0 <= right_edge_limit:
                cells[column].append(index)
        texts = [internal_cell_text(observations, cell, cell_text_cache) for cell in cells]
        if not any(texts):
            continue
        populated += sum(bool(text) for text in texts)
        numeric_cells = [internal_numeric_cell(text) for text in texts]
        for column, is_numeric in enumerate(numeric_cells):
            numeric_by_column[column] += int(is_numeric)
        text_lengths += sum(len(text) for text in texts)
        y0 = min(all_y0[index] for index in row)
        y1 = max(all_y1[index] for index in row)
        table_rows.append(
            tuple(
                TableCell(
                    row=len(table_rows),
                    column=column,
                    text=text,
                    bbox=(edge_list[column], y0, edge_list[column + 1], y1),
                )
                for column, text in enumerate(texts)
            )
        )
    if len(table_rows) < minimum_rows or populated < minimum_rows * 2:
        return None
    density = populated / (len(table_rows) * len(columns))
    average_text = text_lengths / max(1, populated)
    minimum_density = 0.75 if minimum_rows == 2 else 0.35
    if density < minimum_density:
        return None
    numeric_total = sum(numeric_by_column)
    filled_texts = [cell.text.strip() for row in table_rows for cell in row if cell.text.strip()]
    long_text_cells = sum(len(text) > 18 for text in filled_texts)
    sentence_like_cells = sum(
        any(mark in text for mark in (". ", ", ", "; ", ": ")) for text in filled_texts
    )
    character_spaced_cells = sum(internal_character_spaced_cell(text) for text in filled_texts)
    if (
        minimum_rows >= 3
        and len(table_rows) >= 5
        and len(columns) >= 4
        and numeric_total <= 1
        and filled_texts
        and long_text_cells / len(filled_texts) >= 0.35
        and sentence_like_cells / len(filled_texts) >= 0.20
    ):
        return None
    if (
        len(columns) >= 6
        and filled_texts
        and numeric_total / len(filled_texts) < 0.12
        and character_spaced_cells / len(filled_texts) >= 0.50
    ):
        return None
    if minimum_rows == 2:
        if max(numeric_by_column, default=0) < 1 and average_text > 24.0:
            return None
    elif max(numeric_by_column, default=0) < 3 and average_text > 12.0:
        return None
    bbox = (
        float(edges[0]),
        min(cell.bbox[1] for row in table_rows for cell in row if cell.bbox is not None),
        float(edges[-1]),
        max(cell.bbox[3] for row in table_rows for cell in row if cell.bbox is not None),
    )
    return Table(
        order=order,
        rows=tuple(table_rows),
        bbox=bbox,
        confidence=0.75,
        metadata={
            "source": "stream",
            "rows": len(table_rows),
            "columns": len(columns),
            "density": round(density, 4),
            "average_text": round(average_text, 2),
            "numeric_cells": numeric_total,
        },
    )


def internal_compact_stream_table(
    order: int,
    observations: ObservationBatch,
    rows: list[list[int]],
    page_width: float,
    *,
    cell_text_cache: internal_CellTextMemo | None = None,
) -> Table | None:
    """Recover compact tables whose rows are interleaved with nearby prose."""
    candidates = [
        row
        for row in rows
        if len(row) >= 3
        and max(float(observations.bbox[index, 2]) for index in row) <= page_width * 0.55
        and sum(len(observations.text[index].strip()) for index in row) <= 110
    ]
    if len(candidates) < 4:
        return None

    anchor_rows = [row for row in candidates if len(row) == 3]
    if len(anchor_rows) < 2:
        return None
    anchors = numpy.median(
        numpy.asarray(
            [[float(observations.bbox[index, 0]) for index in row] for row in anchor_rows],
            dtype=numpy.float32,
        ),
        axis=0,
    )
    if numpy.any(numpy.diff(anchors) < 30.0):
        return None

    table_rows: list[tuple[TableCell, ...]] = []
    numeric_cells = 0
    for row in candidates:
        cell_indexes: list[list[int]] = [[] for _ in anchors]
        for index in row:
            column = int(numpy.argmin(numpy.abs(anchors - observations.bbox[index, 0])))
            cell_indexes[column].append(index)
        texts = [
            internal_cell_text(observations, indexes, cell_text_cache) for indexes in cell_indexes
        ]
        if not any(texts):
            continue
        numeric_cells += sum(
            internal_numeric_cell(text) or any(character.isdigit() for character in text)
            for text in texts
        )
        y0 = min(float(observations.bbox[index, 1]) for index in row)
        y1 = max(float(observations.bbox[index, 3]) for index in row)
        edges = [
            min(float(observations.bbox[index, 0]) for index in row),
            *(float((anchors[column] + anchors[column + 1]) * 0.5) for column in range(2)),
            max(float(observations.bbox[index, 2]) for index in row),
        ]
        table_rows.append(
            tuple(
                TableCell(
                    row=len(table_rows),
                    column=column,
                    text=text,
                    bbox=(edges[column], y0, edges[column + 1], y1),
                )
                for column, text in enumerate(texts)
            )
        )
    if len(table_rows) < 4 or numeric_cells < 2:
        return None
    boxes = [cell.bbox for row in table_rows for cell in row if cell.bbox is not None]
    if not boxes:
        return None
    return Table(
        order=order,
        rows=tuple(table_rows),
        bbox=bbox_union(boxes),
        confidence=0.7,
        metadata={"source": "stream", "compact": True},
    )


def internal_numeric_cell(text: str) -> bool:
    alphanumeric = sum(character.isalnum() for character in text)
    digits = sum(character.isdigit() for character in text)
    return bool(digits and digits * 2 >= max(1, alphanumeric))


def internal_character_spaced_cell(text: str) -> bool:
    tokens = [token for token in text.split() if any(character.isalpha() for character in token)]
    if len(tokens) < 4:
        return False
    single_character = sum(len(token) == 1 for token in tokens)
    return single_character / len(tokens) >= 0.50


def internal_collapse_character_spaced_cell(text: str) -> str:
    """Collapse glyph-separated prose captured as a table cell."""
    return collapse_character_spaced(text, min_tokens=8, single_char_ratio=0.80)


def internal_stream_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
    start_order: int,
    *,
    cell_text_cache: internal_CellTextMemo | None = None,
) -> tuple[Table, ...]:
    horizontal = (observations.rotation % 180) == 0
    horizontal_count = int(numpy.count_nonzero(horizontal & observations.visible))
    visible_count = int(numpy.count_nonzero(observations.visible))
    if horizontal_count >= 4 and horizontal_count * 5 >= visible_count * 2:
        observations = observations.select(horizontal & observations.visible)
    rows = internal_text_rows(observations)
    row_centers = internal_row_centers(observations, rows)
    tables: list[Table] = []
    for minimum_rows in (3, 2):
        columns = internal_aligned_column_clusters(
            observations,
            rows,
            float(capture.page.width),
            minimum_rows=minimum_rows,
        )
        if len(columns) < 2:
            continue
        row_columns: dict[int, set[int]] = defaultdict(set)
        for column_index, column in enumerate(columns):
            for row_index, internal_index in column:
                row_columns[row_index].add(column_index)
        pair_counts: Counter[tuple[int, int]] = Counter()
        for present in row_columns.values():
            pair_counts.update(combinations(sorted(present), 2))
        disjoint = internal_DisjointSet(len(columns))
        for pair, count in pair_counts.items():
            if count >= minimum_rows:
                disjoint.union(*pair)
        components: dict[int, list[int]] = defaultdict(list)
        for column_index in range(len(columns)):
            components[disjoint.find(column_index)].append(column_index)

        for component in components.values():
            if len(component) < 2 or len(component) > 20:
                continue
            required = max(1 if minimum_rows == 2 else 2, (len(component) + 1) // 2)
            support = sorted(
                row_index
                for row_index, present in row_columns.items()
                if len(present.intersection(component)) >= required
            )
            for group in internal_split_support_rows(
                observations,
                rows,
                support,
                minimum_rows=minimum_rows,
                row_centers=row_centers,
            ):
                table = internal_stream_table(
                    start_order + len(tables),
                    observations,
                    rows,
                    group,
                    [columns[index] for index in component],
                    minimum_rows=minimum_rows,
                    cell_text_cache=cell_text_cache,
                    row_centers=row_centers,
                )
                if table is not None:
                    tables.append(table)
    if not tables:
        compact = internal_compact_stream_table(
            start_order,
            observations,
            rows,
            float(capture.page.width),
            cell_text_cache=cell_text_cache,
        )
        if compact is not None:
            tables.append(compact)
    unique: list[Table] = []
    for table in tables:
        if any(
            table.bbox is not None
            and existing.bbox is not None
            and overlap_ratio_min(table.bbox, existing.bbox) >= 0.8
            for existing in unique
        ):
            continue
        unique.append(table)
    return tuple(unique)


def internal_table_quality(table: Table) -> tuple[int, int, float, int, int]:
    rows = len(table.rows)
    columns = max((len(row) for row in table.rows), default=0)
    populated = sum(bool(cell.text.strip()) for row in table.rows for cell in row)
    density = populated / max(1, rows * columns)
    # Allow more columns to be considered valid for table quality (up to 16)
    return (int(2 <= columns <= 16), populated, density, rows, -columns)


def internal_table_column_bounds(table: Table) -> tuple[tuple[float, float], ...]:
    bounds: list[list[float | None]] = []
    for row in table.rows:
        for cell in row:
            if cell.bbox is None:
                continue
            while len(bounds) <= cell.column:
                bounds.append([None, None])
            left, right = bounds[cell.column]
            bounds[cell.column][0] = cell.bbox[0] if left is None else min(left, cell.bbox[0])
            bounds[cell.column][1] = cell.bbox[2] if right is None else max(right, cell.bbox[2])
    return tuple(
        (float(left), float(right))
        for left, right in bounds
        if left is not None and right is not None
    )


def internal_table_column_alignment(left: Table, right: Table) -> float:
    left_bounds = internal_table_column_bounds(left)
    right_bounds = internal_table_column_bounds(right)
    if len(left_bounds) != len(right_bounds) or not left_bounds:
        return 0.0
    overlaps = []
    for (left_start, left_end), (right_start, right_end) in zip(
        left_bounds, right_bounds, strict=True
    ):
        intersection = interval_overlap(left_start, left_end, right_start, right_end)
        union = max(left_end, right_end) - min(left_start, right_start)
        overlaps.append(intersection / max(1.0, union))
    return fmean(overlaps)


def internal_same_semantic_header(
    left: tuple[TableCell, ...], right: tuple[TableCell, ...]
) -> bool:
    left_text = tuple(cell.text.strip().casefold() for cell in left if cell.text.strip())
    right_text = tuple(cell.text.strip().casefold() for cell in right if cell.text.strip())
    return (
        len(left_text) >= 2
        and left_text == right_text
        and internal_semantic_header_row(left)
        and internal_semantic_header_row(right)
    )


def internal_merge_adjacent_tables(tables: list[Table]) -> list[Table]:
    ordered = sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3])
    merged: list[Table] = []
    for table in ordered:
        if not merged or table.bbox is None or merged[-1].bbox is None:
            merged.append(table)
            continue
        previous = merged[-1]
        previous_bbox = previous.bbox
        table_bbox = table.bbox
        if previous_bbox is None or table_bbox is None:
            merged.append(table)
            continue
        previous_columns = max((len(row) for row in previous.rows), default=0)
        columns = max((len(row) for row in table.rows), default=0)
        vertical_gap = previous_bbox[1] - table_bbox[3]
        # Relax adjacent-table merge conditions slightly to allow merging
        # of tables with minor horizontal overlap or slightly differing column
        # counts. This reduces false splits where a single logical table is
        # broken into two adjacent segments.
        if (
            columns != previous_columns
            or not 2 <= columns <= 16
            or horizontal_overlap_ratio(previous_bbox, table_bbox) < 0.6
            or internal_table_column_alignment(previous, table) < 0.55
            or not -5.0 <= vertical_gap <= TABLE_MERGE_GAP
        ):
            merged.append(table)
            continue
        continuation_rows = table.rows
        if (
            previous.rows
            and table.rows
            and internal_same_semantic_header(previous.rows[0], table.rows[0])
        ):
            continuation_rows = table.rows[1:]
        combined_rows: list[tuple[TableCell, ...]] = []
        for row in (*previous.rows, *continuation_rows):
            combined_rows.append(
                tuple(
                    TableCell(
                        row=len(combined_rows),
                        column=cell.column,
                        text=cell.text,
                        row_span=cell.row_span,
                        column_span=cell.column_span,
                        bbox=cell.bbox,
                    )
                    for cell in row
                )
            )
        merged[-1] = Table(
            order=previous.order,
            rows=tuple(combined_rows),
            bbox=union_bbox(previous_bbox, table_bbox),
            confidence=min(previous.confidence or 1.0, table.confidence or 1.0),
            title=previous.title or table.title,
            caption=table.caption or previous.caption,
            metadata=previous.metadata,
        )
    return merged


def internal_semantic_header_row(row: tuple[TableCell, ...]) -> bool:
    populated = [cell for cell in row if cell.text.strip()]
    if not populated:
        return False
    if len(populated) == 1:
        return populated[0].column_span > 1
    numeric = sum(internal_numeric_cell(cell.text) for cell in populated)
    return numeric == 0 and len(populated) >= 2


def internal_numeric_density(table: Table) -> float:
    cells = [cell for row in table.rows for cell in row if cell.text.strip()]
    return sum(internal_numeric_cell(cell.text) for cell in cells) / max(1, len(cells))


def internal_structured_stream_table(table: Table) -> bool:
    """Identify stream tables with enough structured values to preserve."""
    if table.metadata.get("source") != "stream":
        return False
    numeric_cells = table.metadata.get("numeric_cells", 0)
    return internal_numeric_density(table) >= 0.10 or (
        isinstance(numeric_cells, int) and numeric_cells >= 2
    )


def internal_split_semantic_table(table: Table) -> tuple[Table, ...]:
    """Split long grid regions at repeated section-header rows."""
    if len(table.rows) < 6 or internal_numeric_density(table) < 0.3:
        return (table,)
    boundaries = [
        index
        for index, row in enumerate(table.rows[1:], start=1)
        if (
            internal_semantic_header_row(row)
            and index >= 2
            and index + 1 < len(table.rows)
            and any(internal_numeric_cell(cell.text) for cell in table.rows[index + 1])
        )
    ]
    if not boundaries:
        return (table,)
    signatures = {
        tuple(index for index, cell in enumerate(table.rows[index]) if cell.text.strip())
        for index in boundaries
    }
    labels = {
        " ".join(item.text for item in table.rows[index] if item.text.strip()).casefold()
        for index in boundaries
        if any(item.text.strip() for item in table.rows[index])
    }
    if len(table.rows) > 8 and len(boundaries) > 1 and len(signatures) == 1 and len(labels) == 1:
        return (table,)
    starts = [0, *boundaries]
    segments: list[Table] = []
    for segment_index, start in enumerate(starts):
        end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(table.rows)
        rows = table.rows[start:end]
        if len(rows) < 2:
            continue
        boxes = [cell.bbox for row in rows for cell in row if cell.bbox is not None]
        bbox = bbox_union(boxes)
        segments.append(
            Table(
                order=table.order + segment_index,
                rows=rows,
                bbox=bbox,
                confidence=table.confidence,
                title=table.title if segment_index == 0 else None,
                caption=table.caption if end == len(table.rows) else None,
                metadata=table.metadata,
            )
        )
    return tuple(segments) or (table,)


def internal_table_character_spaced_prose(table: Table) -> bool:
    if table.metadata.get("source") != "stream":
        return False
    columns = max((len(row) for row in table.rows), default=0)
    # Raise the column threshold so only wider multi-column prose is filtered.
    if columns < 8:
        return False
    filled_texts = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled_texts:
        return False
    numeric_cells = sum(internal_numeric_cell(text) for text in filled_texts)
    character_spaced_cells = sum(internal_character_spaced_cell(text) for text in filled_texts)
    average_cell_length = sum(len(text) for text in filled_texts) / len(filled_texts)
    # Tighten the character-spaced fraction to 0.6 to avoid filtering legitimate
    # tables that have some character-spaced cells.
    return (
        numeric_cells / len(filled_texts) < 0.12
        and character_spaced_cells / len(filled_texts) >= 0.60
    ) or (
        len(table.rows) >= 40
        and average_cell_length < 8.0
        and numeric_cells / len(filled_texts) < 0.20
    )


def internal_table_has_grid_shape(table: Table) -> bool:
    """Report whether a table's rows actually use its columns.

    This is the positive form of :func:`internal_table_is_single_column_prose`.
    A grid inferred from alignment or from rules can describe either a real
    table or a column of prose, and what separates them is whether the rows
    divide: a table's rows hold several cells because there are several
    columns to fill, while prose yields one cell per row spanning the width.

    Deciding this on shape alone keeps it free of assumptions about what a
    table contains, so it holds for a schedule of numbers and a grid of
    sentences alike.
    """
    rows = [row for row in table.rows if row]
    if len(rows) < 2:
        return False
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    if columns < 2:
        return False
    divided_rows = sum(1 for row in rows if len(row) >= 2)
    return divided_rows * 2 > len(rows)


def internal_table_is_single_column_prose(table: Table) -> bool:
    """Report a detected grid that is really a column of flowing text.

    A grid inferred from whitespace alignment can latch onto ordinary prose:
    paragraph lines all start at the same margin, so a column boundary appears
    to run down the page. What gives it away is shape rather than content --
    the rows hold one cell each, spanning most of the inferred width, because
    there was never a second column to divide them.

    Judging this on shape keeps it free of assumptions about what a table
    contains; a genuine table has rows that actually use its columns. It
    applies whichever way the grid was inferred: rules drawn on a page are as
    happy to be underlines and dividers as they are to be a table border.
    """
    rows = [row for row in table.rows if row]
    if len(rows) < 3:
        return False
    columns = max((cell.column + cell.column_span for row in rows for cell in row), default=0)
    if columns < 2:
        return False
    single_cell_rows = sum(1 for row in rows if len(row) == 1)
    return single_cell_rows * 2 > len(rows)


internal_STREAM_PROSE_LONG_CELL_CHARACTERS = 25
internal_STREAM_PROSE_LONG_CELL_RATIO = 0.6
internal_STREAM_PROSE_NUMERIC_CELL_RATIO = 0.15
internal_STREAM_WORD_GRID_MIN_COLUMNS = 8
internal_STREAM_WORD_GRID_MIN_ROWS = 12
internal_STREAM_WORD_GRID_NUMERIC_RATIO = 0.2
internal_STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS = 14
internal_STREAM_SPARSE_PROSE_MAX_DENSITY = 0.68
internal_STREAM_SPARSE_PROSE_LONG_RATIO = 0.25
internal_STREAM_SPARSE_PROSE_MAX_COLUMNS = 6


def internal_stream_table_reads_like_prose(table: Table) -> bool:
    """Report a stream table whose cells are sentences rather than values.

    Whitespace alignment finds parallel text columns -- two-column papers,
    side-by-side lists, label/description pairs -- as readily as it finds
    tables. Rendering those row-major interleaves the columns and destroys
    the reading order, which costs far more than the table was worth. Real
    borderless tables carry short data cells and a numeric backbone; a
    candidate dominated by long cells with almost no short numeric ones is
    flowing text and should stay in the normal layout.

    Judged after wrapped-row and text-column merging, because the raw
    detection holds word-level fragments whose lengths say nothing about
    the prose that emerges once the cells are assembled.
    """
    filled = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if not filled:
        return True
    long_cells = sum(1 for text in filled if len(text) > internal_STREAM_PROSE_LONG_CELL_CHARACTERS)
    numeric_cells = sum(
        1
        for text in filled
        if len(text) <= internal_STREAM_PROSE_LONG_CELL_CHARACTERS
        and any(character.isdigit() for character in text)
    )
    if (
        long_cells >= len(filled) * internal_STREAM_PROSE_LONG_CELL_RATIO
        and numeric_cells < len(filled) * internal_STREAM_PROSE_NUMERIC_CELL_RATIO
    ):
        return True
    # Side-by-side lists picked up as one table are sparse -- each row only
    # populates the columns its list reaches -- and their cells run long,
    # because list entries are phrases. A genuine table with empty cells
    # (a financial grid) stays short-celled, a dense table with long cells
    # (definition tables) stays dense, and a wide word matrix (roadmaps)
    # carries more columns than side-by-side lists ever produce, so
    # requiring all three signals rejects the parallel lists alone.
    total_cells = sum(len(row) for row in table.rows)
    narrow = (
        max((len(row) for row in table.rows), default=0) <= internal_STREAM_SPARSE_PROSE_MAX_COLUMNS
    )
    if (
        total_cells
        and narrow
        and len(filled) < total_cells * internal_STREAM_SPARSE_PROSE_MAX_DENSITY
        and long_cells >= len(filled) * internal_STREAM_SPARSE_PROSE_LONG_RATIO
    ):
        return True
    # Word-fragment grids: whitespace alignment over justified prose yields a
    # row per text line and a column per word rail. Cell statistics match a
    # genuine word-y table (short alphabetic cells), but a real one is small
    # -- a body of text produces dozens of rows, and a real table that wide
    # and tall carries numbers.
    columns = max((len(row) for row in table.rows), default=0)
    populated_rows = sum(1 for row in table.rows if any(cell.text.strip() for cell in row))
    if (
        columns >= internal_STREAM_WORD_GRID_MIN_COLUMNS
        and populated_rows >= internal_STREAM_WORD_GRID_MIN_ROWS
        and numeric_cells < len(filled) * internal_STREAM_WORD_GRID_NUMERIC_RATIO
    ):
        lengths = sorted(len(text) for text in filled)
        median_length = lengths[len(lengths) // 2]
        if median_length <= internal_STREAM_WORD_GRID_MEDIAN_CELL_CHARACTERS:
            return True
    return False


def internal_merge_stream_text_columns(table: Table) -> Table:
    """Merge word-aligned columns that form two wrapped text columns."""
    columns = max((len(row) for row in table.rows), default=0)
    if (
        table.metadata.get("source") != "stream"
        or columns < 6
        or columns % 2
        or len(table.rows) < 4
        or internal_numeric_density(table) >= 0.25
    ):
        return table
    group_size = columns // 2
    merged_rows: list[tuple[TableCell, ...]] = []
    for row_index, row in enumerate(table.rows):
        merged: list[TableCell] = []
        for group in range(2):
            cells = row[group * group_size : (group + 1) * group_size]
            text = " ".join(cell.text for cell in cells if cell.text).strip()
            boxes = [cell.bbox for cell in cells if cell.bbox is not None]
            if not text and row_index == 0:
                continue
            bbox = bbox_union(boxes)
            merged.append(
                TableCell(
                    row=row_index,
                    column=len(merged),
                    text=text,
                    column_span=group_size if row_index == 0 and len(merged) == 0 else 1,
                    bbox=bbox,
                )
            )
        if merged:
            merged_rows.append(tuple(merged))
    return replace(table, rows=tuple(merged_rows), metadata={**table.metadata, "merged": True})


internal_LOGICAL_ROW_GAP_RATIO = 0.10
internal_LOGICAL_ROW_MIN_GAP = 0.5
internal_LOGICAL_ROW_MIN_ROWS = 4
internal_LOGICAL_ROW_MAX_NUMERIC_RATIO = 0.40
internal_LOGICAL_ROW_MIN_TALL_RATIO = 3.0
internal_LOGICAL_ROW_MIN_COLUMNS = 5


def internal_merge_wrapped_cell_rows(table: Table) -> Table:
    """Group per-line stream rows into the logical rows their cells span.

    Whitespace alignment sees one row per line of text, but a borderless
    table's cells wrap independently: a logical row is as tall as its
    longest cell, and the shorter cells beside it leave the rest of that
    height blank. Emitting the per-line rows reads across all columns at
    each line, interleaving the wrapped fragments of every cell -- the
    content is all present and every word is in the wrong place.

    A cell that continues past the next line's top holds its logical row
    open, so accumulate rows while the next row starts at or above the
    running bottom of the group. Wrapped lines inside a cell touch (the
    leading gap is a fraction of a line), while a genuine row boundary
    clears the cell padding, which the ratio distinguishes.
    """
    if table.metadata.get("source") != "stream" or len(table.rows) < internal_LOGICAL_ROW_MIN_ROWS:
        return table
    # A numeric table records one datum per line, so its lines are its rows;
    # only descriptive tables wrap a cell across several of them. Line
    # spacing alone cannot tell the two apart -- a tightly set list of names
    # separates its records by less than a descriptive table's leading.
    filled = [cell.text.strip() for row in table.rows for cell in row if cell.text.strip()]
    if filled:
        numeric = sum(
            1
            for text in filled
            if len(text) <= internal_STREAM_PROSE_LONG_CELL_CHARACTERS
            and any(character.isdigit() for character in text)
        )
        if numeric >= len(filled) * internal_LOGICAL_ROW_MAX_NUMERIC_RATIO:
            return table
    if max((len(row) for row in table.rows), default=0) < internal_LOGICAL_ROW_MIN_COLUMNS:
        return table
    extents: list[tuple[float, float]] = []
    heights: list[float] = []
    for row in table.rows:
        boxes = [cell.bbox for cell in row if cell.bbox is not None]
        if not boxes:
            return table
        extents.append((max(box[3] for box in boxes), min(box[1] for box in boxes)))
        heights.extend(box[3] - box[1] for box in boxes)
    if not heights:
        return table
    # Merge only where a cell is demonstrably several lines tall. Spacing
    # alone is too weak a signal: a tightly set table of one-line records
    # separates its rows by less than a wrapped cell's leading, so inferring
    # wrapping from gaps regroups records that were already rows.
    median_height = finite_median(numpy.asarray(heights, dtype=numpy.float32))
    if max(heights) < median_height * internal_LOGICAL_ROW_MIN_TALL_RATIO:
        return table
    tolerance = max(
        internal_LOGICAL_ROW_MIN_GAP,
        median_height * internal_LOGICAL_ROW_GAP_RATIO,
    )
    groups: list[list[int]] = []
    running_bottom = 0.0
    for index, (top, bottom) in enumerate(extents):
        if groups and top >= running_bottom - tolerance:
            groups[-1].append(index)
            running_bottom = min(running_bottom, bottom)
        else:
            groups.append([index])
            running_bottom = bottom
    if len(groups) == len(table.rows) or len(groups) < 2:
        return table
    columns = max((len(row) for row in table.rows), default=0)
    merged_rows: list[tuple[TableCell, ...]] = []
    for row_index, group in enumerate(groups):
        cells: list[TableCell] = []
        for column in range(columns):
            parts: list[str] = []
            cell_boxes: list[tuple[float, float, float, float]] = []
            span = 1
            for index in group:
                row = table.rows[index]
                if column >= len(row):
                    continue
                cell = row[column]
                if cell.text.strip():
                    parts.append(cell.text.strip())
                if cell.bbox is not None:
                    cell_boxes.append(cell.bbox)
                span = max(span, cell.column_span)
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    text=" ".join(parts),
                    column_span=span,
                    bbox=bbox_union(cell_boxes),
                )
            )
        merged_rows.append(tuple(cells))
    return replace(
        table,
        rows=tuple(merged_rows),
        metadata={**table.metadata, "logical_rows": True},
    )


def internal_merge_wrapped_stream_rows(table: Table) -> Table:
    """Merge continuation lines in dense, text-only stream tables."""
    if (
        table.metadata.get("source") != "stream"
        or len(table.rows) < 8
        or max((len(row) for row in table.rows), default=0) < 5
        or table.metadata.get("numeric_cells", 0) > 2
    ):
        return table
    merged: list[list[TableCell]] = []
    for row in table.rows:
        cells = list(row)
        if merged and cells and not cells[0].text.strip():
            previous = merged[-1]
            for index, cell in enumerate(cells):
                if index >= len(previous) or not cell.text.strip():
                    continue
                target = previous[index]
                boxes = [box for box in (target.bbox, cell.bbox) if box is not None]
                previous[index] = replace(
                    target,
                    text=" ".join(part for part in (target.text, cell.text) if part).strip(),
                    bbox=bbox_union(boxes),
                )
            continue
        merged.append(cells)
    if len(merged) == len(table.rows):
        return table
    return replace(
        table,
        rows=tuple(
            tuple(replace(cell, row=index) for cell in row) for index, row in enumerate(merged)
        ),
    )


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


def internal_detect_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
    *,
    cell_text_cache: internal_CellTextMemo | None = None,
) -> tuple[Table, ...]:
    horizontal, vertical = internal_axis_segments(capture)
    horizontal = internal_merge_collinear_segments(horizontal, coordinate=2, start=0, end=1)
    vertical = internal_merge_collinear_segments(vertical, coordinate=0, start=1, end=2)
    components = internal_grid_components(horizontal, vertical)
    ruled_tables: list[Table] = []
    # Only the ruled-grid loop consults this index, and a page with fewer than two
    # segments on either axis yields no components at all.  Building it over every
    # visible observation before that check is wasted on any page without a ruled grid.
    if components:
        visible_indices = numpy.flatnonzero(observations.visible)
        observation_index = SpatialIndex(
            (int(idx), observations.bbox[idx]) for idx in visible_indices
        )
        for component in components:
            for component_part in internal_split_grid_component(*component):
                table = internal_table_from_component(
                    len(ruled_tables),
                    *component_part,
                    observations,
                    observation_index,
                    cell_text_cache=cell_text_cache,
                )
                if table is not None:
                    ruled_tables.append(table)
    ruled = tuple(ruled_tables)
    tables = list(ruled)
    for stream in internal_stream_tables(
        capture,
        observations,
        len(tables),
        cell_text_cache=cell_text_cache,
    ):
        conflicts = [
            table
            for table in tables
            if stream.bbox is not None
            and table.bbox is not None
            and overlap_ratio_min(stream.bbox, table.bbox) >= 0.5
        ]
        if conflicts and internal_table_quality(stream) < max(
            map(internal_table_quality, conflicts)
        ):
            continue
        for conflict in conflicts:
            tables.remove(conflict)
        merged_stream = internal_merge_wrapped_stream_rows(
            internal_merge_stream_text_columns(stream)
        )
        if internal_stream_table_reads_like_prose(merged_stream):
            continue
        # Logical-row grouping runs after the prose gate, which was tuned
        # against per-line cells: merged cells are longer by construction and
        # would read as prose to it.
        tables.append(internal_merge_wrapped_cell_rows(merged_stream))
    tables = [
        segment
        for table in internal_merge_adjacent_tables(tables)
        for segment in internal_split_semantic_table(table)
        if not internal_table_character_spaced_prose(segment)
        and not internal_table_is_single_column_prose(segment)
    ]
    for order, table in enumerate(tables):
        if table.order != order:
            tables[order] = replace(table, order=order)
    tables = [
        replace(
            table,
            rows=tuple(
                tuple(
                    replace(
                        cell,
                        text=internal_repair_table_cell_spaced_digits(
                            internal_collapse_character_spaced_cell(
                                internal_clean_table_cell_leader_runs(cell.text)
                            )
                        ),
                    )
                    for cell in row
                )
                for row in table.rows
            ),
        )
        for table in tables
    ]
    return tuple(sorted(tables, key=lambda table: -(table.bbox or (0.0, 0.0, 0.0, 0.0))[3]))


def internal_annotate_table_associations(
    table: Table,
    observations: ObservationBatch,
    text_rows: list[list[int]],
    *,
    cell_text_cache: internal_CellTextMemo | None = None,
) -> Table:
    """Annotate spanning rows and nearby aligned text without changing cells."""
    title = table.title
    caption = table.caption
    if len(table.rows) >= 2:
        first = tuple(cell for cell in table.rows[0] if cell.text.strip())
        second = tuple(cell for cell in table.rows[1] if cell.text.strip())
        if len(first) == 1 and len(second) >= 2:
            cell = first[0]
            kind = "caption" if ":" in cell.text else "title"
            associated = TableAssociatedText(cell.text, cell.bbox, kind=kind)
            if kind == "caption":
                caption = associated
            else:
                title = associated
    if table.bbox is None or title is not None:
        return replace(table, title=title, caption=caption)
    x0, _y0, x1, y1 = table.bbox
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for row in text_rows:
        boxes = [observations.bbox[index] for index in row]
        row_x0 = min(float(box[0]) for box in boxes)
        row_x1 = max(float(box[2]) for box in boxes)
        row_y0 = min(float(box[1]) for box in boxes)
        overlap = interval_overlap(x0, x1, row_x0, row_x1)
        if overlap / max(1.0, min(x1 - x0, row_x1 - row_x0)) < 0.60:
            continue
        gap = row_y0 - y1
        if 0.0 <= gap <= 36.0:
            candidates.append((gap, tuple(row)))
    if candidates:
        _gap, title_row = min(candidates, key=lambda item: item[0])
        text = internal_cell_text(observations, list(title_row), cell_text_cache)
        if text:
            title = TableAssociatedText(
                text,
                (
                    min(float(observations.bbox[index, 0]) for index in title_row),
                    min(float(observations.bbox[index, 1]) for index in title_row),
                    max(float(observations.bbox[index, 2]) for index in title_row),
                    max(float(observations.bbox[index, 3]) for index in title_row),
                ),
                kind="title",
            )
    if title is table.title and caption is table.caption:
        return table
    return replace(table, title=title, caption=caption)


def internal_table_with_bands(table: Table) -> Table:
    """Materialize table row and column semantics at the extraction boundary."""
    associated = {
        text.kind: text.text.casefold() for text in (table.title, table.caption) if text is not None
    }
    first_grid = next(
        (
            index
            for index, row in enumerate(table.rows)
            if any(cell.text.strip() for cell in row)
            and not any(
                value in associated.values()
                for value in (cell.text.strip().casefold() for cell in row if cell.text.strip())
            )
        ),
        None,
    )
    row_bands: list[TableRowBand] = []
    for index, row in enumerate(table.rows):
        boxes = [cell.bbox for cell in row if cell.bbox is not None]
        texts = tuple(cell.text.strip().casefold() for cell in row if cell.text.strip())
        kind = "blank"
        if texts:
            if any(value in associated.values() for value in texts):
                kind = "title" if texts[0] == associated.get("title") else "caption"
            elif first_grid is not None and index == first_grid and len(texts) >= 2:
                kind = "header"
            else:
                kind = "body"
        row_bands.append(TableRowBand(index=index, bbox=bbox_union(boxes), kind=kind))

    column_count = max(
        (cell.column + cell.column_span for row in table.rows for cell in row),
        default=0,
    )
    column_boxes: list[list[tuple[float, float, float, float]]] = [[] for _ in range(column_count)]
    for row in table.rows:
        for cell in row:
            if cell.bbox is None:
                continue
            for index in range(
                max(cell.column, 0),
                min(cell.column + cell.column_span, column_count),
            ):
                column_boxes[index].append(cell.bbox)
    column_bands = tuple(
        TableColumnBand(index=index, bbox=bbox_union(boxes))
        for index, boxes in enumerate(column_boxes)
    )
    return replace(table, row_bands=tuple(row_bands), column_bands=column_bands)


def extract_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
) -> tuple[Table, ...]:
    """Run the complete table stage and return one annotated table product."""
    # Dense schematic wiring creates large false ruled tables. Vector decoders
    # already supply text geometry, and normal layout preserves those labels
    # without another geometric pass.
    if capture.evidence.vector_text_trusted or capture.evidence.stroked_vector_text.trusted:
        return ()
    cell_text_cache: internal_CellTextMemo = {}
    tables = internal_detect_tables(
        capture,
        observations,
        cell_text_cache=cell_text_cache,
    )
    chart_table = extract_chart_table(capture, observations)
    if chart_table is not None:
        tables = (*tables, chart_table)
    if not tables:
        return ()
    text_rows = internal_text_rows(observations)
    return tuple(
        internal_table_with_bands(
            internal_annotate_table_associations(
                replace(table, order=order) if table.order != order else table,
                observations,
                text_rows,
                cell_text_cache=cell_text_cache,
            )
        )
        for order, table in enumerate(tables)
    )
