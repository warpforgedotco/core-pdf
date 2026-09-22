# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math

from core_pdf_spec.exceptions import PdfParseError


def is_integer_token(token: bytes | memoryview) -> bool:
    raw = token.tobytes() if isinstance(token, memoryview) else token
    return bool(raw) and (raw[1:].isdigit() if raw[0] in (43, 45) else raw.isdigit())


NUMBER_LEAD_BYTES = frozenset(b"+-.0123456789")


def is_number_token(token: bytes | memoryview) -> bool:
    raw = token.tobytes() if isinstance(token, memoryview) else token
    if not raw or raw[0] not in NUMBER_LEAD_BYTES:
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
    raw = token.tobytes() if isinstance(token, memoryview) else token
    if not is_integer_token(raw):
        raise PdfParseError("invalid PDF integer")
    try:
        return int(raw)
    except ValueError as error:
        raise PdfParseError("PDF integer exceeds implementation limits") from error


def parse_real_token(token: bytes | memoryview) -> float:
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
