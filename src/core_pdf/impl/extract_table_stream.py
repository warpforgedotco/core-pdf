# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import combinations

import numpy

from core_pdf.impl.array_views import finite_median
from core_pdf.impl.extract_table_cleanup import clean_stream_table
from core_pdf.impl.extract_table_core import (
    COLUMN_TOLERANCE,
    TableAnalysis,
    TableCandidate,
    TableContext,
    TableFacts,
    cell_text,
    digit_bearing_cell,
    numeric_cell,
)
from core_pdf.impl.geometry import bbox_union, interval_overlap, overlap_ratio_min
from core_pdf.impl.output_model import Table, TableCell
from core_pdf.impl.spatial import DisjointSet

STREAM_CONFLICT_OVERLAP = 0.5
STREAM_DUPLICATE_OVERLAP = 0.8

STREAM_SENTENCE_MIN_ROWS = 5
STREAM_SENTENCE_MIN_COLUMNS = 4
STREAM_SENTENCE_MAX_NUMERIC_CELLS = 1
STREAM_SENTENCE_LONG_CELL_CHARACTERS = 18
STREAM_SENTENCE_LONG_CELL_RATIO = 0.35
STREAM_SENTENCE_MARK_RATIO = 0.20
STREAM_SENTENCE_MARKS = (". ", ", ", "; ", ": ")

STREAM_SPACED_MIN_COLUMNS = 6
STREAM_SPACED_NUMERIC_RATIO = 0.12
STREAM_SPACED_CELL_RATIO = 0.50

STREAM_PAIR_MAX_AVERAGE_TEXT = 24.0
STREAM_MAX_AVERAGE_TEXT = 12.0
STREAM_MIN_NUMERIC_CELLS_PER_COLUMN = 3


def stream_rows_read_as_sentences(
    minimum_rows: int,
    row_count: int,
    column_count: int,
    numeric_total: int,
    filled_texts: tuple[str, ...],
) -> bool:
    if not (
        minimum_rows >= 3
        and row_count >= STREAM_SENTENCE_MIN_ROWS
        and column_count >= STREAM_SENTENCE_MIN_COLUMNS
        and numeric_total <= STREAM_SENTENCE_MAX_NUMERIC_CELLS
        and filled_texts
    ):
        return False
    long_text_cells = sum(len(text) > STREAM_SENTENCE_LONG_CELL_CHARACTERS for text in filled_texts)
    sentence_like_cells = sum(
        any(mark in text for mark in STREAM_SENTENCE_MARKS) for text in filled_texts
    )
    return (
        long_text_cells / len(filled_texts) >= STREAM_SENTENCE_LONG_CELL_RATIO
        and sentence_like_cells / len(filled_texts) >= STREAM_SENTENCE_MARK_RATIO
    )


def stream_rows_character_spaced(column_count: int, numeric_total: int, facts: TableFacts) -> bool:
    filled_texts = facts.filled_texts
    return bool(
        column_count >= STREAM_SPACED_MIN_COLUMNS
        and filled_texts
        and numeric_total / len(filled_texts) < STREAM_SPACED_NUMERIC_RATIO
        and facts.character_spaced_cells / len(filled_texts) >= STREAM_SPACED_CELL_RATIO
    )


def stream_text_outweighs_numbers(
    minimum_rows: int, numeric_by_column: list[int], average_text: float
) -> bool:
    if minimum_rows == 2:
        return max(numeric_by_column, default=0) < 1 and average_text > STREAM_PAIR_MAX_AVERAGE_TEXT
    return (
        max(numeric_by_column, default=0) < STREAM_MIN_NUMERIC_CELLS_PER_COLUMN
        and average_text > STREAM_MAX_AVERAGE_TEXT
    )


def split_support_rows(
    analysis: TableAnalysis,
    indexes: list[int],
    *,
    minimum_rows: int = 3,
) -> list[list[int]]:
    if not indexes:
        return []
    rows = analysis.text_rows
    row_centers = analysis.row_centers
    heights = analysis.coordinates.heights
    groups = [[indexes[0]]]
    for index in indexes[1:]:
        previous = groups[-1][-1]
        previous_height = max(heights[item] for item in rows[previous])
        current_height = max(heights[item] for item in rows[index])
        allowed_gap = max(55.0, max(previous_height, current_height) * 4.0)
        if row_centers[previous] - row_centers[index] > allowed_gap:
            groups.append([])
        groups[-1].append(index)
    return [group for group in groups if len(group) >= minimum_rows]


def stream_table(
    order: int,
    analysis: TableAnalysis,
    support: list[int],
    columns: list[list[tuple[int, int]]],
    *,
    minimum_rows: int = 3,
) -> TableCandidate | None:
    observations = analysis.observations
    rows = analysis.text_rows
    row_centers = analysis.row_centers
    coordinates = analysis.coordinates
    support_set = set(support)
    columns = [
        column
        for column in columns
        if len({row_index for row_index, _ in column}.intersection(support_set)) >= minimum_rows
    ]
    if len(columns) < 2:
        return None
    all_x0 = coordinates.x0
    all_y0 = coordinates.y0
    all_x1 = coordinates.x1
    all_y1 = coordinates.y1
    column_centers = numpy.asarray(
        [
            finite_median(
                numpy.asarray(
                    [all_x0[index] for _, index in column],
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
    for row in selected:
        cells: list[list[int]] = [[] for _ in columns]
        for index in row:
            x0 = all_x0[index]
            x1 = all_x1[index]
            x_center = (x0 + x1) * 0.5
            column = bisect_right(edge_list, x_center) - 1
            if not (0 <= column < column_count):
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
        texts = [cell_text(observations, cell) for cell in cells]
        if not any(texts):
            continue
        populated += sum(bool(text) for text in texts)
        numeric_cells = [numeric_cell(text) for text in texts]
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
    facts = TableFacts.from_rows(table_rows)
    if stream_rows_read_as_sentences(
        minimum_rows, len(table_rows), len(columns), numeric_total, facts.filled_texts
    ):
        return None
    if stream_rows_character_spaced(len(columns), numeric_total, facts):
        return None
    if stream_text_outweighs_numbers(minimum_rows, numeric_by_column, average_text):
        return None
    bbox = (
        float(edges[0]),
        min(cell.bbox[1] for row in table_rows for cell in row if cell.bbox is not None),
        float(edges[-1]),
        max(cell.bbox[3] for row in table_rows for cell in row if cell.bbox is not None),
    )
    return TableCandidate(
        Table(
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
        ),
        facts,
    )


def compact_stream_table(
    order: int,
    analysis: TableAnalysis,
    page_width: float,
) -> TableCandidate | None:
    observations = analysis.observations
    coordinates = analysis.coordinates
    all_x0 = coordinates.x0
    all_y0 = coordinates.y0
    all_x1 = coordinates.x1
    all_y1 = coordinates.y1
    candidates = [
        row
        for row in analysis.text_rows
        if len(row) >= 3
        and max(all_x1[index] for index in row) <= page_width * 0.55
        and sum(len(observations.text[index].strip()) for index in row) <= 110
    ]
    if len(candidates) < 4:
        return None

    anchor_rows = [row for row in candidates if len(row) == 3]
    if len(anchor_rows) < 2:
        return None
    anchors = numpy.median(
        numpy.asarray(
            [[all_x0[index] for index in row] for row in anchor_rows],
            dtype=numpy.float32,
        ),
        axis=0,
    )
    if numpy.any(numpy.diff(anchors) < 30.0):
        return None

    anchor_values: list[float] = anchors.tolist()
    anchor_midpoints: list[float] = ((anchors[:-1] + anchors[1:]) * 0.5).tolist()
    anchor_count = len(anchor_values)

    table_rows: list[tuple[TableCell, ...]] = []
    numeric_cells = 0
    for row in candidates:
        cell_indexes: list[list[int]] = [[] for _ in anchor_values]
        for index in row:
            x0_value = all_x0[index]
            column = min(range(anchor_count), key=lambda c: abs(anchor_values[c] - x0_value))
            cell_indexes[column].append(index)
        texts = [cell_text(observations, indexes) for indexes in cell_indexes]
        if not any(texts):
            continue
        numeric_cells += sum(digit_bearing_cell(text) for text in texts)
        y0 = min(all_y0[index] for index in row)
        y1 = max(all_y1[index] for index in row)
        edges = [
            min(all_x0[index] for index in row),
            *anchor_midpoints,
            max(all_x1[index] for index in row),
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
    return TableCandidate(
        Table(
            order=order,
            rows=tuple(table_rows),
            bbox=bbox_union(boxes),
            confidence=0.7,
            metadata={"source": "stream", "compact": True},
        )
    )


class StreamTableSource:
    __slots__ = ()

    def detect(self, context: TableContext, start_order: int) -> tuple[TableCandidate, ...]:
        analysis = context.analysis
        tables: list[TableCandidate] = []
        candidate_columns = analysis.candidate_columns
        for minimum_rows in (3, 2):
            columns = (
                [
                    column
                    for column in candidate_columns
                    if len({row_index for row_index, _ in column}) >= minimum_rows
                ]
                if minimum_rows > 2
                else candidate_columns
            )
            if len(columns) < 2:
                continue
            row_columns: dict[int, set[int]] = defaultdict(set)
            for column_index, column in enumerate(columns):
                for row_index, _ in column:
                    row_columns[row_index].add(column_index)
            pair_counts: Counter[tuple[int, int]] = Counter()
            for present in row_columns.values():
                pair_counts.update(combinations(sorted(present), 2))
            disjoint = DisjointSet(len(columns))
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
                for group in split_support_rows(analysis, support, minimum_rows=minimum_rows):
                    table = stream_table(
                        start_order + len(tables),
                        analysis,
                        group,
                        [columns[index] for index in component],
                        minimum_rows=minimum_rows,
                    )
                    if table is not None:
                        tables.append(table)
        if not tables:
            compact = compact_stream_table(start_order, analysis, context.capture.width)
            if compact is not None:
                tables.append(compact)
        unique: list[TableCandidate] = []
        for candidate in tables:
            box = candidate.table.bbox
            if any(
                box is not None
                and existing.table.bbox is not None
                and overlap_ratio_min(box, existing.table.bbox) >= STREAM_DUPLICATE_OVERLAP
                for existing in unique
            ):
                continue
            unique.append(candidate)
        return tuple(unique)

    def admit(
        self, accepted: list[TableCandidate], found: tuple[TableCandidate, ...]
    ) -> list[TableCandidate]:
        for stream in found:
            stream_box = stream.table.bbox
            conflicts = [
                candidate
                for candidate in accepted
                if stream_box is not None
                and candidate.table.bbox is not None
                and overlap_ratio_min(stream_box, candidate.table.bbox) >= STREAM_CONFLICT_OVERLAP
            ]
            if conflicts and stream.quality < max(conflict.quality for conflict in conflicts):
                continue
            if conflicts:
                accepted = [
                    candidate
                    for candidate in accepted
                    if not any(candidate is conflict for conflict in conflicts)
                ]
            cleaned = clean_stream_table(stream)
            if cleaned is not None:
                accepted.append(cleaned)
        return accepted
