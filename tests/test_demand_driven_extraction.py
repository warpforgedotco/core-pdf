from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.parse import pipeline as parse_pipeline
from core_pdf.impl.parse.pipeline import PAGE_EXTRACTION_CACHE_KEY, page_extraction

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "SCORE-Bench"
    / "src"
    / "Employee_Health_Benefits_Assess-p006.pdf"
)


def test_repeated_extract_reuses_the_cached_table_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated extraction reuses both the emitted page and its table-stage result."""
    calls = 0

    def counted_extract_tables(
        *internal_args: object, **internal_kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(parse_pipeline, "extract_tables", counted_extract_tables)

    with PdfDocument.open(FIXTURE) as document:
        page = document.pages[0]
        first = page.extract()
        second = page.extract()
        extraction = page_extraction(page)
        cache = page.extraction_cache

    assert first is second
    assert first.tables == ()
    assert calls == 1
    assert cache is not None
    assert cache[PAGE_EXTRACTION_CACHE_KEY] is extraction
    assert extraction.internal_assembled_page is first
