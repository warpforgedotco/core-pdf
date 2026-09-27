# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from copy import replace
from typing import cast

import numpy

from core_pdf.impl import extract_contracts
from core_pdf.impl.extract_contracts import ObservationBatch, ObservationSource
from core_pdf.impl.extract_table_core import TableCandidate, TableContext
from core_pdf.impl.extract_table_detection import TableDetector
from core_pdf.impl.geometry import bbox_union, finite_rect, overlap_ratio_min_exact
from core_pdf.impl.output_model import Table, TableCell
from core_pdf.impl.spatial import band_rows
from core_pdf.impl.text import text_word_tokens
from core_pdf.impl.types import Rectangle
from core_pdf_ocr.impl.extract.contracts import PageAnalysis, PageEvidence

CHART_NUMERIC_TOKEN = re.compile(r"^[+-]?(?:\d[\d,./%\-]*|\d[\d,./%\-]*\s+\d+)$")
CHART_DUPLICATE_OVERLAP = 0.5


def vector_text_untrusted(capture: extract_contracts.PageAnalysis) -> bool:
    evidence = cast(PageEvidence, capture.evidence)
    return not (evidence.vector_text_trusted or evidence.stroked_vector_text.trusted)


class ChartTableSource:
    __slots__ = ()

    def detect(self, context: TableContext, _start_order: int) -> tuple[TableCandidate, ...]:
        capture = cast(PageAnalysis, context.capture)
        chart_table = extract_chart_table(capture, context.analysis.observations)
        return () if chart_table is None else (TableCandidate(chart_table),)

    def admit(
        self, accepted: list[TableCandidate], found: tuple[TableCandidate, ...]
    ) -> list[TableCandidate]:
        return [*accepted, *found]


OCR_TABLES = replace(
    TableDetector.native(),
    gate=vector_text_untrusted,
    supplements=(ChartTableSource(),),
)


def extract_tables(capture: PageAnalysis, observations: ObservationBatch) -> tuple[Table, ...]:
    return OCR_TABLES.extract(capture, observations)


def chart_cell_texts(text: str) -> tuple[str, ...]:
    tokens = text_word_tokens(text)
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
    ordered_cells = sorted(
        cells,
        key=lambda item: (-chart_cell_center_y(item), item.column),
    )
    row_groups = band_rows(
        [chart_cell_center_y(cell) for cell in ordered_cells],
        row_tolerance,
        range(len(ordered_cells)),
        linkage="anchor",
    )
    rows = tuple(
        tuple(
            sorted(
                (replace(ordered_cells[position], row=row_index) for position in group),
                key=lambda item: item.column,
            )
        )
        for row_index, group in enumerate(row_groups)
    )
    return Table(
        order=-1,
        rows=rows,
        bbox=bbox_union(boxes),
        confidence=0.35,
        metadata={"source": "chart-ocr", "synthetic": True},
    )
