from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine import parse as pipeline

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "SCORE-Bench"
    / "src"
    / "Employee_Health_Benefits_Assess-p006.pdf"
)


def test_extract_text_does_not_materialize_tables_or_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_extract_tables(*internal_args: object, **internal_kwargs: object) -> object:
        raise AssertionError("text extraction should not materialize tables")

    def fail_emit(*internal_args: object, **internal_kwargs: object) -> object:
        raise AssertionError("text extraction should not emit a structured page")

    monkeypatch.setattr(pipeline, "extract_tables", fail_extract_tables)
    monkeypatch.setattr(pipeline, "emit_page", fail_emit)

    with PdfDocument.open(FIXTURE) as document:
        text = document.pages[0].extract_text()
        cache = document.pages[0].extraction_cache

    assert text
    assert cache is not None
    assert "emitted_page_v3" not in cache


def test_extract_tables_does_not_materialize_layout_or_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_extract_tables(
        *internal_args: object, **internal_kwargs: object
    ) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return ()

    def fail_layout(*internal_args: object, **internal_kwargs: object) -> object:
        raise AssertionError("table extraction should not materialize layout")

    def fail_emit(*internal_args: object, **internal_kwargs: object) -> object:
        raise AssertionError("table extraction should not emit a structured page")

    monkeypatch.setattr(pipeline, "extract_tables", counted_extract_tables)
    monkeypatch.setattr(pipeline, "layout_blocks", fail_layout)
    monkeypatch.setattr(pipeline, "emit_page", fail_emit)

    with PdfDocument.open(FIXTURE) as document:
        tables = document.pages[0].extract_tables()
        cache = document.pages[0].extraction_cache

    assert tables == []
    assert calls == 1
    assert cache is not None
    assert "emitted_page_v3" not in cache


def test_full_extract_still_populates_parse_metrics() -> None:
    with PdfDocument.open(FIXTURE) as document:
        page = document.pages[0]
        extracted = page.extract()
        cache = page.extraction_cache

    assert extracted.text
    assert cache is not None
    metrics = cast(dict[str, object], cache["parse_metrics"])
    assert metrics["route"] in {"native", "hybrid", "ocr"}
    assert isinstance(metrics["table_seconds"], (int, float))
