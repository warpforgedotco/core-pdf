from __future__ import annotations

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.extract import pipeline as parse_pipeline
from tests.helpers.paths import SCORE_BENCH

FIXTURE = SCORE_BENCH / "Employee_Health_Benefits_Assess-p006.pdf"


def test_repeated_extract_materializes_the_table_stage_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated direct extraction remains deterministic without retained page state."""
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

    assert first == second
    assert first.tables == ()
    assert calls == 2
