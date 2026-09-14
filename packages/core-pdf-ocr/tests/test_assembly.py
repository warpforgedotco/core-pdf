from dataclasses import replace
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf.impl._impl.output.model import Table, TableCell
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract import pipeline
from core_pdf_ocr.impl.extract.contracts import PageRoute, RecognitionResult, WorkPlan
from core_pdf_ocr.impl.extract.table_reconcile import internal_remove_duplicate_tables


def chart(order: int, text: str, *, synthetic: bool = True) -> Table:
    return Table(
        order,
        rows=((TableCell(0, 0, text),),),
        bbox=(10.0, 10.0, 80.0, 40.0),
        metadata={"source": "chart-ocr" if synthetic else "stream", "synthetic": synthetic},
    )


def test_table_reconciliation_prefers_real_tables_and_preserves_survivor_order() -> None:
    duplicate = chart(0, "Revenue sales income profit")
    real = chart(1, "Revenue sales income profit", synthetic=False)
    separate = replace(chart(2, "Cost labor rent materials"), bbox=(100.0, 10.0, 180.0, 40.0))
    assert internal_remove_duplicate_tables((duplicate, real, separate)) == (real, separate)
    assert internal_remove_duplicate_tables((duplicate, duplicate)) == (duplicate,)


def test_ocr_assembly_reconciles_tables_after_using_original_layout_obstacles(
    monkeypatch: pytest.MonkeyPatch,
    text_pdf_bytes: bytes,
) -> None:
    duplicate = chart(0, "Revenue sales income profit")
    real = chart(1, "Revenue sales income profit", synthetic=False)
    tables = (duplicate, real)
    original_layout = pipeline.layout_blocks_with_evidence
    seen: list[tuple[tuple[float, float, float, float], ...]] = []

    def layout(observations: ObservationBatch, **kwargs: Any) -> Any:
        seen.append(kwargs["obstacles"])
        return original_layout(observations, **kwargs)

    monkeypatch.setattr(pipeline, "extract_tables", lambda *args: tables)
    monkeypatch.setattr(pipeline, "layout_blocks_with_evidence", layout)
    with PdfDocument(text_pdf_bytes) as document:
        extraction = pipeline.internal_PageExtraction(
            document.pages[0],
            plan=WorkPlan(PageRoute.NATIVE),
            recognition=RecognitionResult(ObservationBatch.empty()),
        )
        page = extraction.assembled_page(ExtractionScope())
    assert seen == [(duplicate.bbox, real.bbox)]
    assert len(page.tables) == 1
    assert page.tables[0].rows == real.rows
    assert page.tables[0].metadata["source"] == "stream"
    assert page.width == page.height == 200
    assert page.page_number == 1
    assert "Hello maintenance" in page.text
