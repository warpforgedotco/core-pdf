# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl._impl.model.text import (
    collapse_ws,
    compact_text,
    is_neutral_character,
    is_rtl_character,
    search_key,
    text_tokens,
    word_gap_threshold,
)


@pytest.mark.parametrize(
    ("space_width", "size", "expected"),
    [(1.0, 4.0, 0.75), (20.0, 10.0, 3.0), (1.0, 25.0, 2.0)],
)
def test_word_gap_uses_largest_spacing_size_or_minimum_threshold(
    space_width: float, size: float, expected: float
) -> None:
    assert word_gap_threshold(space_width, size) == pytest.approx(expected)


def test_whitespace_and_search_normalization_preserve_distinct_policies() -> None:
    text = " \tStraße\u00a0 12—!\n"
    assert collapse_ws(text) == "Straße 12—!"
    assert search_key(text) == "strasse 12—!"
    assert compact_text(text) == "strasse12"
    assert text_tokens("a (PDF) 42 b") == ("pdf", "42")


@pytest.mark.parametrize(
    ("character", "rtl", "neutral"),
    [
        ("א", True, False),
        ("ع", True, False),
        ("A", False, False),
        (";", False, True),
        ("€", False, True),
    ],
)
def test_character_direction_classification(character: str, rtl: bool, neutral: bool) -> None:
    assert is_rtl_character(character) is rtl
    assert is_neutral_character(character) is neutral
