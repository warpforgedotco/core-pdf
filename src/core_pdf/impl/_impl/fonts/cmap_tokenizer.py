"""Literal-string recovery selected by application CMap readers."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_syntax_primitives.scanning import read_literal_string
from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import (
    decode_cmap_token as decode_spec_cmap_token,
)


def internal_legacy_eol_pair(first: int, second: int) -> bool:
    return (first == 13 and second == 10) or (first == 10 and second == 13)


def decode_cmap_token(token: bytes) -> bytes:
    if not token.startswith(b"("):
        return decode_spec_cmap_token(token)
    raw = memoryview(token)
    if len(raw) < 2:
        raise ValueError("invalid PDF literal string")
    value, _ = read_literal_string(raw, 0, len(raw), eol_pair=internal_legacy_eol_pair)
    if value is None:
        raise ValueError("unterminated PDF literal string")
    return value
