from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.parse import pipeline as parse_pipeline
from core_pdf.impl.engine.parse.pipeline import ASSEMBLED_PAGE_CACHE_KEY

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
        cache = page.extraction_cache

    assert first is second
    assert first.tables == ()
    assert calls == 1
    assert cache is not None
    assert ASSEMBLED_PAGE_CACHE_KEY in cache
