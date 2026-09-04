# SPDX-License-Identifier: AGPL-3.0-only
"""Native text spacing and normalization helpers."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.text import word_gap_threshold

NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
NO_SPACE_AFTER = frozenset("([{")


def gap_separator(left: str, right: str, gap: float, run: Any) -> str:
    if gap <= word_gap_threshold(run.space_width, run.font_size):
        return ""
    if not left or not right or left[-1].isspace() or right[0].isspace():
        return ""
    if right[0] in NO_SPACE_BEFORE or left[-1] in NO_SPACE_AFTER:
        return ""
    return " "


def can_merge_cross_font_word(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return (left[-1].isalnum() or left[-1] == "_") and (right[0].isalnum() or right[0] == "_")


def is_garbage_text(text: str) -> bool:
    if not text:
        return True
    for c in text:
        o = ord(c)
        if not (o < 32 or 0xE000 <= o <= 0xF8FF):
            return False
    return True


# Lone surrogates cannot survive encoding to UTF-8, so drop them here rather
# than letting them fail somewhere downstream.
NORMALIZE_EXTRACTED_TEXT_TABLE = dict.fromkeys(range(0xD800, 0xE000))


def normalize_extracted_text(text: str) -> str:
    if text.isascii():
        return text
    return text.translate(NORMALIZE_EXTRACTED_TEXT_TABLE)


__all__ = (
    "can_merge_cross_font_word",
    "gap_separator",
    "is_garbage_text",
    "normalize_extracted_text",
)
