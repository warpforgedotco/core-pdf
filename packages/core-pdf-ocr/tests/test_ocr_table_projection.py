# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from dataclasses import replace

import pytest

from core_pdf.impl._impl.output.model import Block, BlockKind, Table, TableCell, TextLine
from core_pdf_ocr import PdfDocument
from core_pdf_ocr.impl.extract import pipeline as extraction_pipeline
from core_pdf_ocr.impl.extract.contracts import (
    ObservationBatch,
    ObservationSource,
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    PageRoute,
    RecognitionResult,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline as recognition_pipeline
from core_pdf_ocr.impl.extract.table_reconcile import internal_project_text_and_tables
from tests.helpers.pdf_bytes import one_page_pdf


def internal_table_pdf() -> bytes:
    # Verified with Poppler 26.07.0 `pdftotext -layout` and `pdftoppm -r 150
    # -png -singlefile`, then Tesseract 5.5.2 `image.png stdout --psm 11`:
    # both retain the separate Total heading and 10 23 footer beside the table.
    content = (
        b"BT /F1 18 Tf 20 300 Td (Total) Tj ET\n"
        b"0.5 w 20 120 240 100 re S 140 120 m 140 220 l S 20 170 m 260 170 l S\n"
        b"BT /F1 10 Tf 30 190 Td (Total) Tj ET\n"
        b"BT /F1 10 Tf 170 190 Td (10) Tj ET\n"
        b"BT /F1 10 Tf 30 140 Td (CCPS) Tj ET\n"
        b"BT /F1 10 Tf 170 140 Td (23) Tj ET\n"
        b"BT /F1 10 Tf 250 20 Td (10 23) Tj ET\n"
    )
    return one_page_pdf(content, media_box=(0, 0, 300, 400))


def test_ocr_pdf_extraction_keeps_separate_heading_and_numbers_with_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(OcrPass("controlled-recognition", OcrPassScope.PAGE, 1.0, (11,)),),
    )
    recognized: list[PageAnalysis] = []

    def recognize(capture: PageAnalysis, *args: object, **kwargs: object) -> RecognitionResult:
        recognized.append(capture)
        native = capture.observations
        # Controlled OCR uses the externally verified text and captured PDF
        # positions. Grid detection, layout, projection and public output run normally.
        return RecognitionResult(
            ObservationBatch.from_columns(
                native.text,
                (
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    for box in native.bbox
                ),
                source=ObservationSource.OCR,
                confidence=(99.0,) * len(native),
                font_size=map(float, native.font_size),
                sequence=map(int, native.sequence),
                line_break_before=map(bool, native.line_break_before),
            )
        )

    monkeypatch.setattr(extraction_pipeline, "plan_page", lambda capture: plan)
    monkeypatch.setattr(recognition_pipeline, "recognize_page", recognize)
    with PdfDocument.open(io.BytesIO(internal_table_pdf())) as document:
        page = document.extract().pages[0]

    assert len(recognized) == 1
    assert [block.text for block in page.blocks] == ["Total", "10 23"]
    assert all(line.source == "ocr" for block in page.blocks for line in block.lines)
    assert len(page.tables) == 1
    assert tuple(tuple(cell.text for cell in row) for row in page.tables[0].rows) == (
        ("Total", "10"),
        ("CCPS", "23"),
    )


@pytest.mark.parametrize("table_has_geometry", [True, False])
def test_table_projection_requires_spatial_evidence_to_remove_short_ocr_blocks(
    table_has_geometry: bool,
) -> None:
    table = Table(
        0,
        rows=(
            (TableCell(0, 0, "Total"), TableCell(0, 1, "10")),
            (TableCell(1, 0, "CCPS"), TableCell(1, 1, "23")),
        ),
        bbox=(20.0, 120.0, 260.0, 220.0) if table_has_geometry else None,
        metadata={"source": "lattice"},
    )
    blocks = [
        Block(
            index,
            kind,
            (TextLine(text, bbox=box, source="ocr"),),
            bbox=box,
            provenance=("ocr",),
        )
        for index, (text, kind, box) in enumerate(
            (
                ("Total", BlockKind.HEADING, (20.0, 300.0, 70.0, 320.0)),
                ("10 23", BlockKind.PARAGRAPH, (250.0, 20.0, 290.0, 40.0)),
                ("Total", BlockKind.PARAGRAPH, (30.0, 185.0, 60.0, 200.0)),
            )
        )
    ]

    projected, tables = internal_project_text_and_tables(blocks, (table,))

    assert [block.text for block in projected] == (
        ["Total", "10 23"] if table_has_geometry else ["Total", "10 23", "Total"]
    )
    assert tables == (table,)
    assert [block.text for block in blocks] == ["Total", "10 23", "Total"]


def test_text_shared_by_separate_tables_does_not_cover_an_unrelated_ocr_block() -> None:
    first = Table(
        0,
        rows=((TableCell(0, 0, "Total"),),),
        bbox=(20.0, 120.0, 100.0, 150.0),
        metadata={"source": "lattice"},
    )
    second = replace(
        first,
        order=1,
        rows=((TableCell(0, 0, "due"),),),
        bbox=(150.0, 120.0, 230.0, 150.0),
    )
    block = Block(
        0,
        BlockKind.HEADING,
        (TextLine("Total due", source="ocr"),),
        bbox=(20.0, 300.0, 100.0, 320.0),
        provenance=("ocr",),
    )

    projected, tables = internal_project_text_and_tables([block], (first, second))

    assert projected == [block]
    assert tables == (first, second)
