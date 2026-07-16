from __future__ import annotations

from types import SimpleNamespace
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


def test_substantial_clean_native_table_outweighs_partial_ocr() -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(recommended_strategy="text_table")
    )
    native = "parameter volume temperature pressure value " * 70
    partial_ocr = "parameter volume temperature " * 50

    assert policy.should_preserve_substantial_text_table_native_text(
        page,
        native,
        partial_ocr,
    )


def test_noisy_native_table_can_yield_to_partial_ocr() -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(recommended_strategy="text_table")
    )
    native = "T @ B L E P ? R A M E T E R S } { " * 70
    partial_ocr = "table parameters volume temperature " * 50

    assert not policy.should_preserve_substantial_text_table_native_text(
        page,
        native,
        partial_ocr,
    )


def test_native_table_can_yield_to_materially_more_complete_ocr() -> None:
    page = SimpleNamespace(
        get_page_profile=lambda: SimpleNamespace(recommended_strategy="text_table")
    )
    native = "parameter volume temperature pressure value " * 60
    complete_ocr = "parameter volume temperature pressure value measurement " * 90

    assert not policy.should_preserve_substantial_text_table_native_text(
        page,
        native,
        complete_ocr,
    )
