# SPDX-License-Identifier: AGPL-3.0-only
"""Shared text-normalization primitives used across parse, api, and compat layers."""

from __future__ import annotations

import re

# Smallest horizontal gap that separates two words. Both the spec-level run merger
# and the layout line builder consult this so a gap that reads as a word break in
# one place reads the same in the other.
WORD_GAP_SPACE_FACTOR = 0.15
WORD_GAP_SIZE_FACTOR = 0.08
WORD_GAP_MIN = 0.75

# Leader and filler punctuation (ToC dot leaders, dashed rules, ellipsis fillers)
# that reference text omits. One character class for every stage that strips it.
LEADER_CHARS = frozenset(".-\u2013\u2014~\u2026")
internal_LEADER_CLASS = "[.\\-\u2013\u2014~\u2026]"
internal_LEADER_RUN_RE = re.compile(rf"(?:[ \t]*{internal_LEADER_CLASS}){{2,}}")
internal_TRAILING_LEADER_RE = re.compile(rf"(?<=\S)(?:[ \t]*{internal_LEADER_CLASS}){{3,}}[ \t]*$")
internal_LEADING_LEADER_RE = re.compile(rf"^(?:{internal_LEADER_CLASS}[ \t]+){{2,}}")


def word_gap_threshold(space_width: float, size: float) -> float:
    """Gap width above which two neighbouring runs are separate words."""
    return max(space_width * WORD_GAP_SPACE_FACTOR, size * WORD_GAP_SIZE_FACTOR, WORD_GAP_MIN)


def is_leader_run(text: str) -> bool:
    """True when the non-space content is nothing but leader punctuation."""
    nonspace = [ch for ch in text if not ch.isspace()]
    return len(nonspace) >= 2 and all(ch in LEADER_CHARS for ch in nonspace)


def strip_edge_leaders(text: str) -> str:
    """Drop a leader run at the end (3+) or the start (2+ spaced) of a fragment."""
    text = internal_TRAILING_LEADER_RE.sub("", text)
    return internal_LEADING_LEADER_RE.sub("", text)


def collapse_leader_runs(text: str) -> str:
    """Replace interior leader runs with a single space and collapse whitespace."""
    return collapse_ws(internal_LEADER_RUN_RE.sub(" ", text))


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
    "collapse_character_spaced",
    "collapse_ws",
    "compact_text",
    "is_neutral_character",
    "is_rtl_character",
    "search_key",
    "text_tokens",
)
