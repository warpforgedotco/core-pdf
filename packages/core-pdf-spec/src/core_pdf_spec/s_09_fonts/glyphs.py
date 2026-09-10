"""Adobe Glyph List name-to-Unicode mapping, without application repairs."""

from __future__ import annotations

from core_pdf_spec.s_09_fonts.data.core14 import GLYPH_DATA
from core_pdf_spec.s_09_fonts.data.zapf_dingbats import ZAPF_DINGBATS_GLYPHS


def glyph_name_to_unicode(name: str, *, zapf_dingbats: bool = False) -> str:
    """Apply AGL section 2, including empty results for unknown components."""
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
    if not digits or any(character not in "0123456789ABCDEF" for character in digits):
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
