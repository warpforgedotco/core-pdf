# SPDX-License-Identifier: AGPL-3.0-only
"""Native text spacing and normalization helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")
NO_SPACE_AFTER = frozenset("([{")


@lru_cache(maxsize=256)
def cached_encode_latin1(s: str) -> bytes:
    return s.encode("latin-1", "replace")


def gap_separator(left: str, right: str, gap: float, run: Any) -> str:
    threshold = run.space_width * 0.12
    font_threshold = run.font_size * 0.10
    if font_threshold > threshold:
        threshold = font_threshold
    if threshold < 1.0:
        threshold = 1.0
    if gap <= threshold:
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


@lru_cache(maxsize=4096)
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
    "NO_SPACE_AFTER",
    "NO_SPACE_BEFORE",
    "cached_encode_latin1",
    "can_merge_cross_font_word",
    "gap_separator",
    "is_garbage_text",
    "normalize_extracted_text",
)
