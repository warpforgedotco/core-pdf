# SPDX-License-Identifier: AGPL-3.0-only
"""Chart, ruled-grid, and whitespace-inferred table extraction."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from typing import cast

import numpy

from core_pdf.impl.extract.contracts import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
)
from core_pdf.impl.extract.grids import (
    internal_axis_segments,
    internal_DisjointSet,
    internal_grid_components,
    internal_merge_collinear_segments,
    internal_split_grid_component,
    internal_table_from_component,
)
from core_pdf.impl.extract.table_cleanup import (
    internal_cell_text,
    internal_character_spaced_cell,
    internal_clean_table_cell_leader_runs,
    internal_collapse_character_spaced_cell,
    internal_merge_adjacent_tables,
    internal_merge_stream_text_columns,
    internal_merge_wrapped_cell_rows,
    internal_merge_wrapped_stream_rows,
    internal_numeric_cell,
    internal_repair_table_cell_spaced_digits,
    internal_split_semantic_table,
    internal_stream_table_reads_like_prose,
    internal_table_character_spaced_prose,
    internal_table_is_single_column_prose,
    internal_table_quality,
)
from core_pdf.impl.model.geometry import bbox_union, interval_overlap, overlap_ratio_min
from core_pdf.impl.output import Table, TableCell
from core_pdf.impl.runtime.array_views import finite_median

# Table-stage orchestration.

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


def internal_detect_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
) -> tuple[Table, ...]:
    horizontal, vertical = internal_axis_segments(capture)
    horizontal = internal_merge_collinear_segments(horizontal, coordinate=2, start=0, end=1)
    vertical = internal_merge_collinear_segments(vertical, coordinate=0, start=1, end=2)
    components = internal_grid_components(horizontal, vertical)
    ruled_tables: list[Table] = []
    if components:
        for component in components:
            for component_part in internal_split_grid_component(*component):
                table = internal_table_from_component(
                    len(ruled_tables),
                    *component_part,
                    observations,
                )
                if table is not None:
                    ruled_tables.append(table)
    ruled = tuple(ruled_tables)
    tables = list(ruled)
    for stream in internal_stream_tables(
        capture,
        observations,
        len(tables),
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


def extract_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
) -> tuple[Table, ...]:
    """Compatibility import for the table-stage coordinator."""
    from core_pdf.impl.extract.table_pipeline import extract_tables as run_table_pipeline

    return run_table_pipeline(capture, observations)


# Whitespace-aligned stream table inference.

COLUMN_TOLERANCE = 14.0  # loosen tolerance for column edge alignment to reduce split tables


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
        texts = [internal_cell_text(observations, cell) for cell in cells]
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
        texts = [internal_cell_text(observations, indexes) for indexes in cell_indexes]
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


def internal_stream_tables(
    capture: CapturedPage,
    observations: ObservationBatch,
    start_order: int,
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
