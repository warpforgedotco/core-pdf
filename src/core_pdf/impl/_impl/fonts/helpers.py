# SPDX-License-Identifier: AGPL-3.0-only
"""Native font encoding and differences helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from core_pdf.impl._impl.fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_syntax_primitives.text_string import PDFDOC_ENCODING_TABLE
from core_pdf_spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf_spec.s_09_fonts.helpers import (
    build_simple_encoding_glyph_names as spec_simple_encoding_glyph_names,
)


def strip_subset_tag(font_name: str) -> str:
    """Drop the ``ABCDEF+`` subset prefix (9.6.4) from a base font name."""
    return font_name.split("+", 1)[-1]


LIGATURE_TEXT_OVERRIDES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


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


def build_decode_table(
    key: str,
    differences: dict[int, str] | tuple[tuple[int, str], ...] | None = None,
) -> tuple[str, ...]:
    base = ENCODING_FALLBACKS.get(key, internal_PDFDOC_FALLBACK_TABLE)
    if not differences:
        return base
    table = list(base)
    items = differences.items() if isinstance(differences, dict) else differences
    for code, glyph_name in items:
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
            glyph_name = recover_pdf_name(item)
        if glyph_name is None:
            continue
        if code < 0 or code > 255:
            continue
        differences[code] = glyph_name
        code += 1
    return differences


# PDFDocEncoding is the only base whose entries need ligature expansion, so it
# is normalized once here rather than per lookup.
internal_PDFDOC_FALLBACK_TABLE: tuple[str, ...] = tuple(
    normalize_ligature_text(text) for text in PDFDOC_ENCODING_TABLE
)

ENCODING_FALLBACKS: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_TABLE,
    # Type3 fonts use StandardEncoding when /Encoding is omitted.  Keep this
    # fallback separate from the parser's default so explicitly supplied
    # Differences can still override individual character codes.
    "Type3": STANDARD_ENCODING_TABLE,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_TABLE,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_TABLE,
}


def build_simple_encoding_glyph_names(
    base_encoding: str | None,
    builtin_encoding: Mapping[int, str],
    differences: Mapping[int, str],
    *,
    authoritative_builtin: bool,
) -> tuple[str, ...]:
    """Keep the reader's out-of-range skipping and empty-name substitution."""
    return spec_simple_encoding_glyph_names(
        base_encoding,
        {
            int(code): name or ".notdef"
            for code, name in builtin_encoding.items()
            if 0 <= code < 256
        },
        {int(code): name or ".notdef" for code, name in differences.items() if 0 <= code < 256},
        authoritative_builtin=authoritative_builtin,
    )
