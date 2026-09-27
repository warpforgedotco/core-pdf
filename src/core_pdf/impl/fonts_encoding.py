# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.fonts_glyphs import glyph_name_to_unicode
from core_pdf.impl.fonts_helpers import (
    recover_strip_subset_tag,
)

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


def normalized_base_font_name(base_font_name: str | None) -> str | None:
    if base_font_name is None:
        return None
    return recover_strip_subset_tag(base_font_name)


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
