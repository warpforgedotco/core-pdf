"""PDF font encoding names and dictionary semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from core_pdf._vendor.fontTools.agl import UV2AGL
from core_pdf._vendor.fontTools.encodings.MacRoman import MacRoman
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_09_fonts.data.base_encodings import (
    WIN_ANSI_ENCODING,
)


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
    return tuple(name or ".notdef" for name in values)


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
    "WinAnsiEncoding": WIN_ANSI_ENCODING_GLYPH_NAMES,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_GLYPH_NAMES,
}


def base_encoding_glyph_names(key: str) -> tuple[str, ...]:
    return BASE_ENCODING_GLYPH_NAMES[key]


def strip_subset_tag(font_name: str) -> str:
    """Strip the six uppercase letters and plus sign specified by PDF 9.6.4."""
    if len(font_name) > 7 and font_name[6] == "+" and all("A" <= c <= "Z" for c in font_name[:6]):
        return font_name[7:]
    return font_name


def parse_differences(
    value: Any, resolve_name: Callable[[Any], str | None] | None = None
) -> dict[int, str]:
    if value is None:
        return {}
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid encoding differences array")
    result: dict[int, str] = {}
    code: int | None = None
    for item in value:
        if type(item) is int:
            if not 0 <= item <= 255:
                raise ValueError("encoding difference code outside simple-font range")
            code = item
            continue
        name = (resolve_name or normalize_pdf_name)(item)
        if name is None or code is None or code > 255:
            raise ValueError("invalid encoding difference")
        result[code] = name
        code += 1
    return result


def build_simple_encoding_glyph_names(
    base_encoding: str | None,
    builtin_encoding: Mapping[int, str],
    differences: Mapping[int, str],
    *,
    authoritative_builtin: bool,
) -> tuple[str, ...]:
    """Layer one complete simple-font code-to-glyph-name encoding.

    Custom and Expert CFF encodings are sparse and authoritative: an
    absent code denotes ``.notdef`` rather than falling through to
    StandardEncoding. Explicit PDF /Differences are always the final layer.
    """
    names = (
        [".notdef"] * 256
        if authoritative_builtin
        else list(base_encoding_glyph_names(base_encoding or "StandardEncoding"))
    )
    for mapping in (builtin_encoding, differences):
        for code, name in mapping.items():
            if 0 <= code < 256:
                names[code] = name or ".notdef"
    return tuple(names)
