# SPDX-License-Identifier: AGPL-3.0-only

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
    BASE_ENCODING_GLYPH_NAMES,
    get_base_encoding_glyph_names,
)
from core_pdf_spec.s_09_fonts.helpers import (
    build_simple_encoding_glyph_names as spec_simple_encoding_glyph_names,
)
from core_pdf_spec.standards import SemanticContext


def strip_subset_tag(font_name: str) -> str:
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
    mapped = glyph_name_to_unicode(glyph_name)
    if not mapped or (mapped == glyph_name and len(glyph_name) != 1):
        return None
    return normalize_ligature_text(mapped)


def internal_resolve_base_encoding(table: tuple[str, ...]) -> tuple[str, ...]:
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
    *,
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    base = ENCODING_FALLBACKS.get(key, internal_PDFDOC_FALLBACK_TABLE)
    if context is not None and key in BASE_ENCODING_GLYPH_NAMES:
        names = get_base_encoding_glyph_names(key, context=internal_encoding_context(context))
        modern_names = BASE_ENCODING_GLYPH_NAMES[key]
        if names is not modern_names:
            base = tuple(
                (unicode_for_glyph_name(name) or "") if name != modern_names[code] else base[code]
                for code, name in enumerate(names)
            )
    if not differences:
        return base
    table = list(base)
    items = differences.items() if isinstance(differences, dict) else differences
    for code, glyph_name in items:
        mapped = unicode_for_glyph_name(glyph_name)
        if mapped is None:
            if glyph_name.isdecimal():
                continue
            table[code] = ""
            continue
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


internal_PDFDOC_FALLBACK_TABLE: tuple[str, ...] = tuple(
    normalize_ligature_text(text) for text in PDFDOC_ENCODING_TABLE
)

ENCODING_FALLBACKS: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_TABLE,
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
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    return spec_simple_encoding_glyph_names(
        base_encoding,
        {
            int(code): name or ".notdef"
            for code, name in builtin_encoding.items()
            if 0 <= code < 256
        },
        {int(code): name or ".notdef" for code, name in differences.items() if 0 <= code < 256},
        authoritative_builtin=authoritative_builtin,
        context=internal_encoding_context(context),
    )


def internal_encoding_context(context: SemanticContext | None) -> SemanticContext | None:
    if context is not None and (context.version is None or not context.version.recognized):
        return None
    return context
