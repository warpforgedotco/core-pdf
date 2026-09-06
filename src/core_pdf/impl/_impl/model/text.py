# SPDX-License-Identifier: AGPL-3.0-only
"""Shared text normalization and word-boundary rules, independent of processing stages."""

from __future__ import annotations

import re
from dataclasses import replace

from core_pdf.impl.types import TextWord

internal_NORMALIZE_TEXT_TABLE = dict.fromkeys(range(0xD800, 0xE000))


def normalize_extracted_text(text: str) -> str:
    """Drop lone surrogates, which cannot survive UTF-8 serialization."""
    return text if text.isascii() else text.translate(internal_NORMALIZE_TEXT_TABLE)


# Smallest horizontal gap that separates two words. Both the capture run merger
# and the layout line builder consult this so a gap that reads as a word break in
# one place reads the same in the other.
WORD_GAP_SPACE_FACTOR = 0.15
WORD_GAP_SIZE_FACTOR = 0.08
WORD_GAP_MIN = 0.75

internal_CONTENT_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def content_tokens(text: str) -> tuple[str, ...]:
    """Keep words, case, and every non-whitespace symbol for content comparisons."""
    return tuple(internal_CONTENT_TOKEN_RE.findall(text))


def complete_text_covered(candidate: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    """Require the complete ordered content without dropping surrounding operators."""
    if not candidate:
        return False
    size = len(candidate)
    for start in range(len(reference) - size + 1):
        if reference[start : start + size] != candidate:
            continue
        # A bare 5 is not equivalent to > 5, even though its word token matches.
        if start and not reference[start - 1].isalnum():
            continue
        if start + size < len(reference) and not reference[start + size].isalnum():
            continue
        return True
    return False


def word_gap_threshold(space_width: float, size: float) -> float:
    """Gap width above which two neighbouring runs are separate words."""
    return max(space_width * WORD_GAP_SPACE_FACTOR, size * WORD_GAP_SIZE_FACTOR, WORD_GAP_MIN)


def collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim the ends."""
    return " ".join(text.split())


def internal_text_word_tokens(text: str) -> tuple[str, ...]:
    """Return the non-whitespace word vocabulary used throughout extraction."""
    return tuple(text.split())


def internal_reconcile_text_words(
    text: str,
    words: tuple[TextWord, ...],
) -> tuple[TextWord, ...]:
    """Keep word text and geometry consistent after line-level normalization.

    Unchanged and one-to-one substituted words retain their boxes. If normalization
    changes word boundaries, geometry cannot be mapped safely and is left unknown.
    """
    tokens = internal_text_word_tokens(text)
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
    """Casefolded, whitespace-collapsed key for text matching and search."""
    return " ".join(text.casefold().split())


def is_rtl_character(character: str) -> bool:
    """True for characters in the right-to-left script ranges."""
    return any(
        start <= character <= end
        for start, end in (
            ("\u0590", "\u08ff"),
            ("\ufb1d", "\ufdff"),
            ("\ufe70", "\ufeff"),
        )
    )


def is_neutral_character(character: str) -> bool:
    """True for characters that take their direction from their neighbours."""
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
    """Casefolded alphanumeric-only form, for comparing text across sources."""
    return "".join(character.casefold() for character in text if character.isalnum())


def text_tokens(text: str) -> tuple[str, ...]:
    """Compacted tokens of two or more characters, for overlap and duplicate checks."""
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
