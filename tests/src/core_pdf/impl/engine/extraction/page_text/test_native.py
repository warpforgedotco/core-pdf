from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import build_page_extraction_result

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"


def test_native_extraction_returns_pdf_text_without_external_services() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        text = page.extract_text()

        assert text.strip()
        assert page.extraction_cache is not None
        assert "native_text" in page.extraction_cache
        assert "page_extraction_snapshot" not in page.extraction_cache


def test_structured_page_result_reports_native_route() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert result.text.strip()
        assert result.base_route in {"native_fast", "native_layout"}
        assert result.resolved_lines
