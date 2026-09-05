# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl._impl.model.text import (
    collapse_character_spaced,
    collapse_leader_runs,
    collapse_ws,
    compact_text,
    is_leader_run,
    is_neutral_character,
    is_rtl_character,
    search_key,
    strip_edge_leaders,
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
    ("text", "expected"), [(" . — … ", True), (".", False), (" . page . ", False)]
)
def test_leader_detection_requires_multiple_punctuation_characters(
    text: str, expected: bool
) -> None:
    assert is_leader_run(text) is expected


def test_leader_cleanup_preserves_single_punctuation() -> None:
    assert collapse_leader_runs(" Chapter 1 . . . 12 ") == "Chapter 1 12"
    assert strip_edge_leaders(". . Chapter...") == "Chapter"
    assert strip_edge_leaders("Chapter.") == "Chapter."
    assert strip_edge_leaders("Chapter..") == "Chapter.."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("H e l l o\tw o r l d", "Hello world"),
        ("A B", "A B"),
        ("This is a title", "This is a title"),
    ],
)
def test_character_spacing_repair_requires_enough_single_character_tokens(
    text: str, expected: str
) -> None:
    assert collapse_character_spaced(text, min_tokens=4, single_char_ratio=0.8) == expected


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
