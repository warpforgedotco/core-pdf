# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import unicodedata

from core_pdf.impl.engine.spec.s_09_fonts.glyphs import glyph_name_to_unicode

TEX_MATH_GLYPH_OVERRIDES: dict[str, dict[str, str]] = {
    "TeX_Times_Math_Italic": {
        "C14": "δ",
    },
    "TeX_Times_Math_Symbol": {
        "C14": "°",
    },
}
COMPUTER_MODERN_MATH_PREFIXES = (
    "CMEX",
    "CMMI",
    "CMMIB",
    "CMSY",
)
LIGATURE_GLYPH_TEXT = {
    "ff": "ff",
    "fi": "fi",
    "fl": "fl",
    "ffi": "ffi",
    "ffl": "ffl",
    "f_f": "ff",
    "f_i": "fi",
    "f_l": "fl",
    "f_f_i": "ffi",
    "f_f_l": "ffl",
}


def has_untrusted_unicode_semantics(text: str) -> bool:
    if not text:
        return True
    for ch in text:
        if ch == "\ufffd":
            return True
        if unicodedata.category(ch).startswith("C"):
            return True
    return False


def has_invalid_unicode_mapping(text: str) -> bool:
    """Return whether a mapping is an explicit failure sentinel."""
    return "\ufffd" in text or "\x00" in text


def should_prefer_glyph_name_mapping(
    current: str,
    mapped: str,
    *,
    authoritative: bool,
) -> bool:
    if not mapped or current == mapped:
        return False
    if authoritative:
        return True
    if has_untrusted_unicode_semantics(current):
        return True
    return unicodedata.normalize("NFKC", mapped) == current


def normalized_base_font_name(base_font_name: str | None) -> str | None:
    if base_font_name is None:
        return None
    return base_font_name.split("+", 1)[-1]


def build_glyph_decode_table(
    base_font_name: str | None, differences: dict[int, str]
) -> tuple[tuple[str, ...], bool] | None:
    normalized = normalized_base_font_name(base_font_name)
    if normalized is None:
        return None
    overrides = TEX_MATH_GLYPH_OVERRIDES.get(normalized, {})
    is_computer_modern_math = normalized.startswith(COMPUTER_MODERN_MATH_PREFIXES)
    if not overrides and not is_computer_modern_math and not differences:
        return None
    table: list[str | None] = [None] * 256
    has_mapping = False
    for code, glyph_name in differences.items():
        mapped = overrides.get(glyph_name)
        if mapped is None:
            mapped = LIGATURE_GLYPH_TEXT.get(glyph_name)
        if mapped is None:
            mapped = glyph_name_to_unicode(glyph_name)
            if mapped == glyph_name:
                mapped = None
        if mapped is not None and 0 <= code <= 255:
            table[code] = mapped
            has_mapping = True
    if not has_mapping:
        return None
    return tuple(ch or "" for ch in table), bool(overrides or is_computer_modern_math)


def replace_unicode_from_glyph_names(
    text: str,
    data: bytes,
    glyph_decode_table: tuple[str, ...],
    *,
    authoritative: bool,
    fallback_mapping: bool = False,
) -> str:
    if not text:
        if authoritative or fallback_mapping or all(glyph_decode_table[code] for code in data):
            return "".join(glyph_decode_table[code] for code in data)
        return text
    if len(text) != len(data):
        return text
    if len(data) == 1:
        mapped = glyph_decode_table[data[0]]
        if not mapped:
            return text
        if not (
            fallback_mapping
            or should_prefer_glyph_name_mapping(
                text,
                mapped,
                authoritative=authoritative,
            )
        ):
            return text
        return mapped
    out: list[str] | None = None
    for index, code in enumerate(data):
        mapped = glyph_decode_table[code]
        if not mapped:
            continue
        current = text[index]
        if current == mapped:
            continue
        if not (
            fallback_mapping
            or should_prefer_glyph_name_mapping(
                current,
                mapped,
                authoritative=authoritative,
            )
        ):
            continue
        if out is None:
            out = list(text)
        out[index] = mapped
    return text if out is None else "".join(out)
