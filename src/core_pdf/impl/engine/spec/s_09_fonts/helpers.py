# SPDX-License-Identifier: AGPL-3.0-only
"""Native font encoding and differences helpers."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Callable

from core_pdf._vendor.fontTools.agl import UV2AGL
from core_pdf._vendor.fontTools.encodings.MacRoman import MacRoman
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.engine.spec.s_07_syntax.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf.impl.engine.spec.s_07_syntax.text_string import PDFDOC_ENCODING_TABLE
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


def unicode_for_glyph_name(glyph_name: str) -> str | None:
    """Resolve a glyph name, distinguishing an unknown name from valid text."""
    mapped = glyph_name_to_unicode(glyph_name)
    if not mapped or (mapped == glyph_name and len(glyph_name) != 1):
        return None
    return normalize_ligature_text(mapped)


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


def internal_glyph_name_for_unicode(text: str) -> str:
    """Return the conventional PostScript name for one encoded character."""
    if not text:
        return ".notdef"
    codepoint = ord(text)
    # AGLFN intentionally omits compatibility characters that PDF's Annex D
    # encodings still name. Keep their conventional PostScript spellings so
    # outline selection remains independent of Unicode recovery.
    legacy_names = {
        0x00B2: "twosuperior",
        0x00B3: "threesuperior",
        0x00B9: "onesuperior",
        0x03A9: "Omega",
        0xFB01: "fi",
        0xFB02: "fl",
    }
    return legacy_names.get(codepoint, UV2AGL.get(codepoint, f"uni{codepoint:04X}"))


def internal_normalize_glyph_names(values: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(name if name else ".notdef" for name in values)


STANDARD_ENCODING_GLYPH_NAMES = internal_normalize_glyph_names(list(StandardEncoding))

internal_mac_roman_glyph_names = list(MacRoman)
# fontTools' first 32 entries are the Mac glyph ordering rather than character
# codes. Annex D leaves them undefined and differs from the later Mac OS table
# at the four slots below.
internal_mac_roman_glyph_names[:32] = [".notdef"] * 32
internal_mac_roman_glyph_names[0x7F] = ".notdef"
internal_mac_roman_glyph_names[0xCA] = "space"
internal_mac_roman_glyph_names[0xDB] = "currency"
internal_mac_roman_glyph_names[0xF0] = ".notdef"
MAC_ROMAN_ENCODING_GLYPH_NAMES = internal_normalize_glyph_names(internal_mac_roman_glyph_names)

WIN_ANSI_ENCODING_GLYPH_NAMES = tuple(
    internal_glyph_name_for_unicode(text) for text in WIN_ANSI_ENCODING
)

BASE_ENCODING_GLYPH_NAMES: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_GLYPH_NAMES,
    "Type3": STANDARD_ENCODING_GLYPH_NAMES,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_GLYPH_NAMES,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_GLYPH_NAMES,
}


def base_encoding_glyph_names(key: str | None) -> tuple[str, ...]:
    """Return a complete code-to-name table without inventing undefined names."""
    return BASE_ENCODING_GLYPH_NAMES.get(key or "StandardEncoding", STANDARD_ENCODING_GLYPH_NAMES)


def build_decode_table(
    fallback_fn: EncodingFallback,
    differences: dict[int, str] | None = None,
) -> tuple[str, ...]:
    if not differences:
        return tuple(fallback_fn(b) for b in range(256))
    table = [fallback_fn(b) for b in range(256)]
    for code, glyph_name in differences.items():
        mapped = unicode_for_glyph_name(glyph_name)
        if mapped is None:
            if glyph_name.isdecimal():
                # Producer-made Type 3 encodings commonly use the character
                # code (or a producer's neighboring internal identifier) as
                # the CharProc name. It has no AGL meaning; PDF readers ignore
                # that failed difference and retain the inherited encoding.
                continue
            table[code] = ""
            continue
        # Expand ligatures here too, so a glyph reached through /Differences
        # or a built-in encoding reads the same as one reached through a base
        # encoding table.
        table[code] = mapped
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
