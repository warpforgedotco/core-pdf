from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.extraction.trace import TRACE_CACHE_KEY, extraction_stage


def test_extraction_stage_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CORE_PDF_TRACE", raising=False)
    cache: dict[object, object] = {}

    with extraction_stage(cache, "native"):
        pass

    assert TRACE_CACHE_KEY not in cache


def test_extraction_stage_accumulates_timings(monkeypatch) -> None:
    monkeypatch.setenv("CORE_PDF_TRACE", "1")
    cache: dict[object, object] = {}

    with extraction_stage(cache, "native"):
        pass
    with extraction_stage(cache, "native"):
        pass

    stages = cache[TRACE_CACHE_KEY]
    assert isinstance(stages, dict)
    typed_stages = cast(dict[str, float], stages)
    assert typed_stages["native"] >= 0.0
