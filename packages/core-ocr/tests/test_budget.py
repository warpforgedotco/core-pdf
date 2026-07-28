from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from core_ocr.impl import coordinator


def test_expired_render_budget_stops_before_loading_backend(monkeypatch) -> None:
    page = SimpleNamespace(extraction_cache={})

    def unexpected_backend_load():
        raise AssertionError("expired budget must not load Tesseract")

    monkeypatch.setattr(
        coordinator.TesseractCtypesBackend,
        "from_system",
        unexpected_backend_load,
    )

    coordinator.append_rendered_full_page_ocr_candidates(
        cast(Any, page),
        [],
        1.0,
        base_image=cast(Any, None),
        deadline=0.0,
    )

    assert page.extraction_cache["ocr_budget"] == {
        "exhausted": True,
        "candidate_count": 0,
    }
