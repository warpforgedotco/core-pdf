# SPDX-License-Identifier: AGPL-3.0-only
"""Augment native table candidates with recognized chart regions."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch, internal_bbox_tuple
from core_pdf.impl.extract.table_detection import (
    internal_detect_tables,
    internal_finalize_tables,
    internal_TableAnalysis,
)
from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.output.model import Table, TableCell
from core_pdf_ocr.impl.extract.contracts import ObservationSource, PageAnalysis

internal_CHART_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d[\d,./%\-]*|\d[\d,./%\-]*\s+\d+)$")


def extract_tables(capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
    """Detect native tables and chart associations from recognized observations."""
    evidence = capture.evidence
    if evidence.vector_text_trusted or evidence.stroked_vector_text.trusted:
        return ()
    analysis = internal_TableAnalysis.build(observations, capture.width)
    tables = internal_detect_tables(capture, analysis)
    chart_table = extract_chart_table(capture, observations)
    if chart_table is not None:
        tables = (*tables, chart_table)
    return internal_finalize_tables(tables, analysis)


def internal_chart_cell_texts(text: str) -> tuple[str, ...]:
    """Split dense OCR axis/value lines while keeping prose intact."""
    tokens = tuple(part for part in text.split() if part)
    numeric_count = sum(bool(internal_CHART_NUMERIC_TOKEN.fullmatch(part)) for part in tokens)
    if len(tokens) >= 4 and numeric_count >= 3:
        return tokens
    return (text,)


def internal_chart_cell_center_y(cell: TableCell) -> float:
    box = cell.bbox or (0.0, 0.0, 0.0, 0.0)
    return (box[1] + box[3]) / 2


def extract_chart_table(capture: PageAnalysis, observations: ObservationBatch) -> Table | None:
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
        box = internal_bbox_tuple(observations.bbox[int(index)])
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
    row_tolerance = max(6.0, capture.height * 0.008)
    row_groups: list[tuple[float, list[TableCell]]] = []
    # Grouping only compares against the open group, so the cells have to arrive
    # in the same order the comparison uses: descending row center, not bbox top.
    for cell in sorted(
        cells,
        key=lambda item: (-internal_chart_cell_center_y(item), item.column),
    ):
        center_y = internal_chart_cell_center_y(cell)
        if not row_groups or abs(row_groups[-1][0] - center_y) > row_tolerance:
            row_groups.append((center_y, [cell]))
        else:
            row_groups[-1][1].append(cell)
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
