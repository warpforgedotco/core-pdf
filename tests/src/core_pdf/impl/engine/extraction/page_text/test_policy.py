from __future__ import annotations

from typing import Any

import pytest

from core_pdf.impl.engine.extraction.page_text import policy


def test_classify_page_region_can_skip_dominant_image_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dominant_image_probe(page: Any) -> bool:
        raise AssertionError("dominant image probe should not run")

    monkeypatch.setattr(policy, "page_has_dominant_image", fail_dominant_image_probe)

    classification = policy.classify_page_region(
        "plain native text",
        page=object(),
        include_dominant_image=False,
    )

    assert classification.signals["dominant_image"] is False


def test_classify_page_region_checks_dominant_image_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def dominant_image_probe(page: Any) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(policy, "page_has_dominant_image", dominant_image_probe)

    classification = policy.classify_page_region("plain native text", page=object())

    assert calls == 1
    assert classification.signals["dominant_image"] is True
