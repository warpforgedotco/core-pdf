from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
from core_ocr.impl import policy


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


def test_native_alignment_skips_word_geometry_when_text_cannot_signal_a_table() -> None:
    def fail_word_geometry() -> None:
        raise AssertionError("word geometry should not be constructed")

    line = SimpleNamespace(
        text=lambda: "ordinary prose without digits",
        cached_text_and_words=fail_word_geometry,
    )

    assert policy.native_aligned_column_count([cast(Any, line)]) == 0


def test_table_signal_text_prefilter_has_no_false_negatives() -> None:
    randomizer = random.Random(11)
    alphabet = "abcXYZ0129 -:/.()"
    for _ in range(1_000):
        text = "".join(randomizer.choice(alphabet) for _ in range(randomizer.randrange(1, 80)))
        word_strings: list[str] = []
        current = ""
        for char in text:
            if char.isspace():
                if current:
                    word_strings.append(current)
                    current = ""
                continue
            if current and char.isalnum() != current[-1].isalnum():
                word_strings.append(current)
                current = ""
            current += char
        if current:
            word_strings.append(current)

        words = [SimpleNamespace(text=word) for word in word_strings]

        assert policy.layout_text_word_strings(text) == tuple(word_strings)
        digit_words = sum(1 for word in word_strings if any(char.isdigit() for char in word))
        alpha_words = sum(
            1
            for word in word_strings
            if not any(char.isdigit() for char in word) and any(char.isalpha() for char in word)
        )
        expected_numeric_signal = digit_words >= 2 or (
            digit_words >= 1 and len(word_strings) >= 4 and alpha_words <= 2
        )
        line = SimpleNamespace(text=lambda text=text: text)
        assert policy.layout_line_has_numeric_signal(cast(Any, line)) is expected_numeric_signal

        if policy.native_word_line_has_table_signal(words):
            assert policy.text_may_have_native_word_table_signal(text)


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
