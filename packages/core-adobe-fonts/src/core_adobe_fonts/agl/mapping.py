# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_adobe_fonts.agl.zapf_dingbats import ZAPF_DINGBATS_GLYPHS


def glyph_name_to_unicode(name: str, *, zapf_dingbats: bool = False) -> str:
    name = name.partition(".")[0]
    return "".join(
        glyph_component_to_unicode(part, zapf_dingbats=zapf_dingbats) for part in name.split("_")
    )


def glyph_component_to_unicode(name: str, *, zapf_dingbats: bool = False) -> str:
    if zapf_dingbats and name in ZAPF_DINGBATS_GLYPHS:
        return ZAPF_DINGBATS_GLYPHS[name]
    if name in GLYPH_DATA:
        return GLYPH_DATA[name]
    digits = name[3:] if name.startswith("uni") else name[1:] if name.startswith("u") else ""
    if not digits or digits.strip("0123456789ABCDEF"):
        return ""
    if name.startswith("uni"):
        if len(digits) % 4:
            return ""
        values = [int(digits[offset : offset + 4], 16) for offset in range(0, len(digits), 4)]
    elif 4 <= len(digits) <= 6:
        values = [int(digits, 16)]
    else:
        return ""
    if any(value > 0x10FFFF or 0xD800 <= value <= 0xDFFF for value in values):
        return ""
    return "".join(chr(value) for value in values)


__all__ = ["glyph_name_to_unicode", "glyph_component_to_unicode"]
