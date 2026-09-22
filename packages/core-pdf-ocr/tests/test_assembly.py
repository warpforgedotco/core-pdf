from copy import replace
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.output.model import Table, TableCell
from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract import pipeline
from core_pdf_ocr.impl.extract.contracts import PageRoute, RecognitionResult, WorkPlan
from core_pdf_ocr.impl.extract.table_reconcile import remove_duplicate_tables


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
    assert remove_duplicate_tables((duplicate, real, separate)) == (real, separate)
    assert remove_duplicate_tables((duplicate, duplicate)) == (duplicate,)


@pytest.mark.parametrize("route", [PageRoute.NATIVE, PageRoute.OCR])
def test_ocr_assembly_reconciles_tables_after_using_original_layout_obstacles(
    monkeypatch: pytest.MonkeyPatch,
    text_pdf_bytes: bytes,
    route: PageRoute,
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
        extraction = pipeline.PageExtraction(
            document.pages[0],
            plan=WorkPlan(route),
            recognition=RecognitionResult(
                ObservationBatch.from_columns(
                    ("Recognized maintenance",),
                    ((20.0, 100.0, 160.0, 112.0),),
                    source=1,
                    confidence=(99.0,),
                    font_size=(12.0,),
                )
            ),
        )
        page = extraction.assembled_page(ExtractionScope())
    assert seen == [(duplicate.bbox, real.bbox)]
    assert len(page.tables) == 1
    assert page.tables[0].rows == real.rows
    assert page.tables[0].metadata["source"] == "stream"
    assert page.width == page.height == 200
    assert page.page_number == 1
    expected = "Hello maintenance" if route is PageRoute.NATIVE else "Recognized maintenance"
    assert expected in page.text


def test_companion_exports_resolve() -> None:
    import core_pdf_ocr

    for name in core_pdf_ocr.__all__:
        assert getattr(core_pdf_ocr, name).__module__ == "core_pdf_ocr.api.document"
