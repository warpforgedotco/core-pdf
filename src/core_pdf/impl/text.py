# SPDX-License-Identifier: AGPL-3.0-only
"""Shared text-normalization primitives used across parse, api, and compat layers."""

from __future__ import annotations


def collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim the ends."""
    return " ".join(text.split())


def search_key(text: str) -> str:
    """Casefolded, whitespace-collapsed key for text matching and search."""
    return " ".join(text.casefold().split())


def collapse_character_spaced(text: str, *, min_tokens: int, single_char_ratio: float) -> str:
    """Collapse glyph-per-token spacing back into words when it dominates a line.

    Some PDFs use unusually narrow character advances, so layout emits every
    glyph as a separate token while retaining larger word gaps as tabs. The
    thresholds restrict the repair to strongly character-spaced runs so genuine
    short labels and tables are left untouched.
    """
    tokens = text.split()
    if len(tokens) < min_tokens:
        return text
    if sum(len(token) == 1 for token in tokens) / len(tokens) < single_char_ratio:
        return text
    return text.replace(" ", "").replace("\t", " ")


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


__all__ = (
    "collapse_character_spaced",
    "collapse_ws",
    "is_neutral_character",
    "is_rtl_character",
    "search_key",
)
