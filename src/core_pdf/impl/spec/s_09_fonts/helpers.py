"""ISO 32000-2 9.6: a simple font's base encoding and Differences."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.adobe_fonts.encodings import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
)
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext, require_recognized_version

BASE_ENCODING_GLYPH_NAMES: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_GLYPH_NAMES,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_GLYPH_NAMES,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_GLYPH_NAMES,
}
WIN_ANSI_PRE_1_3_GLYPH_NAMES: tuple[str, ...] = tuple(
    "bullet" if code in (0x80, 0x8E, 0x9E) else name
    for code, name in enumerate(WIN_ANSI_ENCODING_GLYPH_NAMES)
)
PDF_1_3 = PdfVersion(1, 3)


def symbol_character_code(codepoint: int) -> int:
    return codepoint & 0xFF if 0xF000 <= codepoint <= 0xF2FF else codepoint


def is_unicode_scalar(codepoint: int) -> bool:
    return 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF


@cites(
    "ISO 32000-2",
    "D.2",
    since="1.3",
    rule="WinAnsiEncoding 80, 8E and 9E are Euro, Zcaron and zcaron, not bullet",
)
def get_base_encoding_glyph_names(
    base_encoding: str, *, context: SemanticContext | None = None
) -> tuple[str, ...]:
    require_recognized_version(context, "font-encoding semantics require a recognized PDF version")
    if (
        base_encoding == "WinAnsiEncoding"
        and context is not None
        and context.version is not None
        and context.version < PDF_1_3
    ):
        return WIN_ANSI_PRE_1_3_GLYPH_NAMES
    return BASE_ENCODING_GLYPH_NAMES[base_encoding]


def build_simple_encoding_glyph_names(
    base_encoding: str | None,
    builtin_encoding: Mapping[int, str],
    differences: Mapping[int, str],
    *,
    authoritative_builtin: bool,
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    base_names = get_base_encoding_glyph_names(
        "StandardEncoding" if authoritative_builtin else base_encoding or "StandardEncoding",
        context=context,
    )
    if not authoritative_builtin and not builtin_encoding and not differences:
        return base_names
    names = [".notdef"] * 256 if authoritative_builtin else list(base_names)
    for mapping in (builtin_encoding, differences):
        for code, name in mapping.items():
            if type(code) is not int or not 0 <= code < 256 or not name:
                raise ValueError("invalid simple-font encoding entry")
            names[code] = name
    return tuple(names)


__all__ = [
    "BASE_ENCODING_GLYPH_NAMES",
    "get_base_encoding_glyph_names",
    "build_simple_encoding_glyph_names",
    "is_unicode_scalar",
    "symbol_character_code",
    "STANDARD_ENCODING_GLYPH_NAMES",
    "MAC_ROMAN_ENCODING_GLYPH_NAMES",
    "WIN_ANSI_ENCODING_GLYPH_NAMES",
]
