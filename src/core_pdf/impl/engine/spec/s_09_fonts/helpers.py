# SPDX-License-Identifier: AGPL-3.0-only
"""Native font encoding and differences helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import PDFDOC_ENCODING_TABLE
from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode

EncodingFallback = Callable[[int], str]
LIGATURE_TEXT_OVERRIDES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def fallback_with_pdfdoc(b: int) -> str:
    return normalize_ligature_text(PDFDOC_ENCODING_TABLE[b])


def normalize_ligature_text(text: str) -> str:
    return LIGATURE_TEXT_OVERRIDES.get(text, text)


def internal_resolve_base_encoding(table: tuple[str, ...]) -> tuple[str, ...]:
    """Turn an Annex D.2 table into a decode table.

    Undefined codes below 040 keep their raw value, matching how the control
    range is treated everywhere else; above that they decode to nothing, since
    the encoding genuinely assigns them no glyph.
    """
    return tuple(
        normalize_ligature_text(text) if text else (chr(code) if code < 32 else "")
        for code, text in enumerate(table)
    )


STANDARD_ENCODING_TABLE = internal_resolve_base_encoding(STANDARD_ENCODING)
WIN_ANSI_ENCODING_TABLE = internal_resolve_base_encoding(WIN_ANSI_ENCODING)
MAC_ROMAN_ENCODING_TABLE = internal_resolve_base_encoding(MAC_ROMAN_ENCODING)


def build_decode_table(
    fallback_fn: EncodingFallback,
    differences: dict[int, str] | None = None,
) -> tuple[str, ...]:
    gtn = glyph_name_to_unicode
    if not differences:
        return tuple(fallback_fn(b) for b in range(256))
    table = [fallback_fn(b) for b in range(256)]
    for code, glyph_name in differences.items():
        mapped = gtn(glyph_name)
        if mapped == glyph_name and len(glyph_name) != 1:
            table[code] = ""
            continue
        # Expand ligatures here too, so a glyph reached through /Differences
        # or a built-in encoding reads the same as one reached through a base
        # encoding table.
        table[code] = normalize_ligature_text(mapped)
    return tuple(table)


@lru_cache(maxsize=256)
def cached_decode_table(
    key: str, differences_items: tuple[tuple[int, str], ...]
) -> tuple[str, ...]:
    differences = dict(differences_items)
    return build_decode_table(ENCODING_FALLBACKS.get(key, fallback_with_pdfdoc), differences)


def parse_differences(
    value: Any, resolve_name: Callable[[Any], str | None] | None = None
) -> dict[int, str]:
    differences: dict[int, str] = {}
    if value is None:
        return differences
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid encoding differences array")
    code = 0
    for item in value:
        if type(item) is int:
            if item < 0 or item > 255:
                continue
            code = item
            continue
        if resolve_name is not None:
            glyph_name = resolve_name(item)
        else:
            glyph_name = normalize_pdf_name(item)
        if glyph_name is None:
            continue
        if code < 0 or code > 255:
            continue
        differences[code] = glyph_name
        code += 1
    return differences


ENCODING_FALLBACKS: dict[str, EncodingFallback] = {
    "StandardEncoding": STANDARD_ENCODING_TABLE.__getitem__,
    # Type3 fonts use StandardEncoding when /Encoding is omitted.  Keep this
    # fallback separate from the parser's default so explicitly supplied
    # Differences can still override individual character codes.
    "Type3": STANDARD_ENCODING_TABLE.__getitem__,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_TABLE.__getitem__,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_TABLE.__getitem__,
}
