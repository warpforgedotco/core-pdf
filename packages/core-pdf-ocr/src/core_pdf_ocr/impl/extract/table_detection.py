# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from copy import replace

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.extract.table_detection import (
    TableAnalysis,
    detect_tables,
    finalize_tables,
)
from core_pdf.impl.geometry import bbox_union, finite_rect, overlap_ratio_min_exact
from core_pdf.impl.output.model import Table, TableCell
from core_pdf.impl.types import Rectangle
from core_pdf_ocr.impl.extract.contracts import ObservationSource, PageAnalysis

CHART_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d[\d,./%\-]*|\d[\d,./%\-]*\s+\d+)$")
CHART_DUPLICATE_OVERLAP = 0.5


def extract_tables(capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
    evidence = capture.evidence
    if evidence.vector_text_trusted or evidence.stroked_vector_text.trusted:
        return ()
    analysis = TableAnalysis.build(observations, capture.width)
    tables = detect_tables(capture, analysis)
    chart_table = extract_chart_table(capture, observations)
    if chart_table is not None:
        tables = (*tables, chart_table)
    return finalize_tables(tables, analysis)


def chart_cell_texts(text: str) -> tuple[str, ...]:
    tokens = tuple(part for part in text.split() if part)
    numeric_count = sum(bool(CHART_NUMERIC_TOKEN.fullmatch(part)) for part in tokens)
    if len(tokens) >= 4 and numeric_count >= 3:
        return tokens
    return (text,)


def chart_cell_center_y(cell: TableCell) -> float:
    box = cell.bbox or (0.0, 0.0, 0.0, 0.0)
    return (box[1] + box[3]) / 2


def extract_chart_table(capture: PageAnalysis, observations: ObservationBatch) -> Table | None:
    if (capture.evidence.uncovered_vector_area or 0.0) < 20_000.0:
        return None
    ocr_indexes = numpy.flatnonzero(observations.source == int(ObservationSource.OCR))
    if len(ocr_indexes) < 3:
        return None

    cells: list[TableCell] = []
    boxes: list[tuple[float, float, float, float]] = []
    chart_observations: list[tuple[str, Rectangle]] = []
    for index in ocr_indexes:
        text = observations.text[int(index)].strip()
        box = finite_rect(observations.bbox[int(index)])
        if not text or box is None:
            continue
        chart_observations.append((text, box))

    seen: dict[str, list[Rectangle]] = {}
    for text, box in sorted(chart_observations, key=lambda item: item[1][0]):
        matching_boxes = seen.setdefault(text.casefold(), [])
        if any(
            overlap_ratio_min_exact(box, previous) >= CHART_DUPLICATE_OVERLAP
            for previous in matching_boxes
        ):
            continue
        matching_boxes.append(box)
        parts = chart_cell_texts(text)
        if len(parts) == 1:
            boxes.append(box)
            cells.append(TableCell(row=0, column=len(cells), text=text, bbox=box))
            continue
        width = (box[2] - box[0]) / len(parts)
        for offset, part in enumerate(parts):
            part_box = (box[0] + width * offset, box[1], box[0] + width * (offset + 1), box[3])
            boxes.append(part_box)
            cells.append(TableCell(row=0, column=len(cells), text=part, bbox=part_box))
    if len(cells) < 3:
        return None
    row_tolerance = max(6.0, capture.height * 0.008)
    row_groups: list[tuple[float, list[TableCell]]] = []
    for cell in sorted(
        cells,
        key=lambda item: (-chart_cell_center_y(item), item.column),
    ):
        center_y = chart_cell_center_y(cell)
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
        for row_index, (center_y_of, group) in enumerate(row_groups)
    )
    return Table(
        order=-1,
        rows=rows,
        bbox=bbox_union(boxes),
        confidence=0.35,
        metadata={"source": "chart-ocr", "synthetic": True},
    )
