# SPDX-License-Identifier: AGPL-3.0-only
"""Orchestrate chart, ruled-grid, and whitespace-inferred table detection."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import cast

import numpy

from core_pdf.impl.extract.contracts import (
    CapturedPage,
    ObservationBatch,
    ObservationSource,
)
from core_pdf.impl.extract.table_cleanup import (
    internal_annotate_table_associations,
    internal_CellTextMemo,
    internal_clean_table_cell_leader_runs,
    internal_collapse_character_spaced_cell,
    internal_merge_adjacent_tables,
    internal_merge_stream_text_columns,
    internal_merge_wrapped_cell_rows,
    internal_merge_wrapped_stream_rows,
    internal_repair_table_cell_spaced_digits,
    internal_split_semantic_table,
    internal_stream_table_reads_like_prose,
    internal_table_character_spaced_prose,
    internal_table_is_single_column_prose,
    internal_table_quality,
    internal_table_with_bands,
)
from core_pdf.impl.extract.table_grid import (
    internal_merge_collinear_segments,
    internal_table_from_component,
)
from core_pdf.impl.extract.table_stream import internal_stream_tables, internal_text_rows
from core_pdf.impl.layout.grids import (
    internal_axis_segments,
    internal_grid_components,
    internal_split_grid_component,
)
from core_pdf.impl.layout.spatial import SpatialIndex
from core_pdf.impl.model.geometry import bbox_union, overlap_ratio_min
from core_pdf.impl.output import Table, TableCell

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
