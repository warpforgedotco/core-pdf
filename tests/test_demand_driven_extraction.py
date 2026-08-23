from __future__ import annotations

from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.parse import ParseReport
from core_pdf.impl.engine.parse import pipeline as parse_pipeline
from core_pdf.impl.engine.parse import tables as parse_tables
from core_pdf.impl.engine.parse.pipeline import ASSEMBLED_PAGE_CACHE_KEY

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "SCORE-Bench"
    / "src"
    / "Employee_Health_Benefits_Assess-p006.pdf"
)


def test_extract_text_uses_the_graph_text_view() -> None:
    with PdfDocument.open(FIXTURE) as document:
        text = document.extract().text
        cache = document.pages[0].extraction_cache

    assert text
    assert cache is not None
    assert ASSEMBLED_PAGE_CACHE_KEY in cache


def test_extract_tables_reconciles_with_the_emitted_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Table extraction reports the tables the emitted page keeps.

    This deliberately gives up the older contract that extract_tables() would
    not materialize layout or emit. A candidate table is only known to be a
    table once it has been checked against the blocks and found not to be a
    repeat of them, and that check needs the blocks -- so the cheap path could
    only ever return candidates. Reporting those left the two table APIs
    disagreeing on 53 of the 224 benchmark documents, always by reporting
    tables the document itself does not carry: 109 candidates against 23 the
    emitted page keeps.

    The cost, measured over 40 documents, is roughly 5% on a table-only
    extraction; capture dominates and both paths pay it.
    """
    calls = 0

    def counted_extract_tables(
        *internal_args: object, **internal_kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(parse_pipeline, "extract_tables", counted_extract_tables)
    monkeypatch.setattr(parse_tables, "extract_tables", counted_extract_tables)

    with PdfDocument.open(FIXTURE) as document:
        page = document.pages[0]
        tables = page.extract().table_view.tables
        emitted = page.extract().table_view.tables
        cache = page.extraction_cache

    assert tables == ()
    assert calls == 1
    assert len(tables) == len(emitted)
    assert cache is not None
    # The assembled page remains cached after the graph view is requested.
    assert ASSEMBLED_PAGE_CACHE_KEY in cache


def test_full_extract_populates_the_typed_parse_report() -> None:
    with PdfDocument.open(FIXTURE) as document:
        page = document.pages[0]
        extracted = page.extract()
        cache = page.extraction_cache

    assert extracted.text
    assert cache is not None
    report = cache["parse_report_v1"]
    assert isinstance(report, ParseReport)
    metrics = report.metrics
    assert metrics["route"] in {"native", "hybrid", "ocr"}
    assert isinstance(metrics["table_seconds"], (int, float))
