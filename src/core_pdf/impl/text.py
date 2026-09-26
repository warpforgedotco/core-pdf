# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from copy import replace

from core_pdf.impl.types import TextWord

NORMALIZE_TEXT_TABLE = dict.fromkeys(range(0xD800, 0xE000))
# Surrogates are rare in extracted text, and a search finds that out in C
# without building the copy translate always returns.
SURROGATE_RE = re.compile("[\ud800-\udfff]")


def normalize_extracted_text(text: str) -> str:
    if text.isascii() or SURROGATE_RE.search(text) is None:
        return text
    return text.translate(NORMALIZE_TEXT_TABLE)


WORD_GAP_SPACE_FACTOR = 0.15
WORD_GAP_SIZE_FACTOR = 0.08
WORD_GAP_MIN = 0.75

CONTENT_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(CONTENT_TOKEN_RE.findall(text))


def complete_text_covered(candidate: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    if not candidate:
        return False
    size = len(candidate)
    for start in range(len(reference) - size + 1):
        if reference[start : start + size] != candidate:
            continue
        if start and not reference[start - 1].isalnum():
            continue
        if start + size < len(reference) and not reference[start + size].isalnum():
            continue
        return True
    return False


def word_gap_threshold(space_width: float, size: float) -> float:
    return max(space_width * WORD_GAP_SPACE_FACTOR, size * WORD_GAP_SIZE_FACTOR, WORD_GAP_MIN)


def collapse_ws(text: str) -> str:
    return " ".join(text.split())


def text_word_tokens(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def reconcile_text_words(
    text: str,
    words: tuple[TextWord, ...],
) -> tuple[TextWord, ...]:
    tokens = text_word_tokens(text)
    if not tokens:
        return ()
    if not words:
        return tuple(TextWord(token) for token in tokens)
    if tuple(word.text for word in words) == tokens:
        return words
    if len(words) == len(tokens):
        return tuple(replace(word, text=token) for word, token in zip(words, tokens, strict=True))
    return tuple(TextWord(token) for token in tokens)


def search_key(text: str) -> str:
    return " ".join(text.casefold().split())


def is_rtl_character(character: str) -> bool:
    return any(
        start <= character <= end
        for start, end in (
            ("\u0590", "\u08ff"),
            ("\ufb1d", "\ufdff"),
            ("\ufe70", "\ufeff"),
        )
    )


def is_neutral_character(character: str) -> bool:
    return any(
        start <= character <= end
        for start, end in (
            ("\x00", "\x2f"),
            ("\x3a", "\x40"),
            ("\u2000", "\u206f"),
            ("\u20a0", "\u21ff"),
        )
    )


def compact_text(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def text_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (compact_text(part) for part in text.casefold().split())
        if len(token) >= 2
    )


__all__ = (
    "collapse_ws",
    "complete_text_covered",
    "compact_text",
    "content_tokens",
    "is_neutral_character",
    "is_rtl_character",
    "search_key",
    "text_tokens",
)
