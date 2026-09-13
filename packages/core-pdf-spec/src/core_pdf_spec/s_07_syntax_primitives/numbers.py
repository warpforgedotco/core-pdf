# SPDX-License-Identifier: AGPL-3.0-only
"""PDF numeric tokens and the distinct syntax of indirect object identifiers."""

from __future__ import annotations

import math

from core_pdf_spec.exceptions import PdfParseError


def is_integer_token(token: bytes | memoryview) -> bool:
    raw = token.tobytes() if isinstance(token, memoryview) else token
    return bool(raw) and (raw[1:].isdigit() if raw[0] in (43, 45) else raw.isdigit())


def is_number_token(token: bytes | memoryview) -> bool:
    raw = token.tobytes() if isinstance(token, memoryview) else token
    if not raw:
        return False
    if raw.isdigit():
        return True
    digits = raw[1:] if raw[0] in (43, 45) else raw
    if digits.isdigit():
        return True
    before, dot, after = digits.partition(b".")
    return (
        bool(dot)
        and bool(before or after)
        and (not before or before.isdigit())
        and (not after or after.isdigit())
    )


def parse_integer_token(token: bytes | memoryview) -> int:
    """Convert a complete integer token, preserving Python's configured digit limit."""
    raw = token.tobytes() if isinstance(token, memoryview) else token
    if not is_integer_token(raw):
        raise PdfParseError("invalid PDF integer")
    try:
        return int(raw)
    except ValueError as error:
        raise PdfParseError("PDF integer exceeds implementation limits") from error


def parse_real_token(token: bytes | memoryview) -> float:
    """Convert a PDF number to a finite real; Annex C requires reporting limits."""
    raw = token.tobytes() if isinstance(token, memoryview) else token
    if not is_number_token(raw):
        raise PdfParseError("invalid PDF real number")
    value = float(raw)
    if not math.isfinite(value):
        raise PdfParseError("PDF real number exceeds implementation limits")
    return value


def parse_identifier_tokens(
    object_token: bytes | memoryview,
    generation_token: bytes | memoryview,
    *,
    canonical: bool = True,
) -> tuple[int, int]:
    """Validate the two tokens in ``X Y obj`` or ``X Y R``.

    ISO 32000-2:2020, 7.3.10, corrected by issue 379, requires digits without
    signs or redundant leading zeros. Earlier editions specify integer tokens.
    Object-stream pairs and xref fields have their own integer representations.
    """
    if canonical:
        for token in (object_token, generation_token):
            raw = token.tobytes() if isinstance(token, memoryview) else token
            if not raw.isdigit() or (len(raw) > 1 and raw[0] == 48):
                raise PdfParseError("invalid indirect object identifier")
    object_number = parse_integer_token(object_token)
    generation = parse_integer_token(generation_token)
    if object_number <= 0 or not 0 <= generation <= 65535:
        raise PdfParseError("invalid indirect object identifier")
    return object_number, generation


__all__ = (
    "is_integer_token",
    "is_number_token",
    "parse_identifier_tokens",
    "parse_integer_token",
    "parse_real_token",
)
