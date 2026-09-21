from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from core_adobe_fonts.encodings import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
)
from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf_spec.standards import PdfVersion, SemanticContext

BASE_ENCODING_GLYPH_NAMES: dict[str, tuple[str, ...]] = {
    "StandardEncoding": STANDARD_ENCODING_GLYPH_NAMES,
    "WinAnsiEncoding": WIN_ANSI_ENCODING_GLYPH_NAMES,
    "MacRomanEncoding": MAC_ROMAN_ENCODING_GLYPH_NAMES,
}


def get_base_encoding_glyph_names(
    base_encoding: str, *, context: SemanticContext | None = None
) -> tuple[str, ...]:
    if context is not None and (context.version is None or not context.version.recognized):
        raise PdfUnsupportedError("font-encoding semantics require a recognized PDF version")
    names = BASE_ENCODING_GLYPH_NAMES[base_encoding]
    if (
        base_encoding == "WinAnsiEncoding"
        and context is not None
        and context.version is not None
        and context.version < PdfVersion(1, 3)
    ):
        historical = list(names)
        for code in (0x80, 0x8E, 0x9E):
            historical[code] = "bullet"
        return tuple(historical)
    return names


def strip_subset_tag(font_name: str) -> str:
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
        name = (resolve_name or decoded_name)(item)
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
    context: SemanticContext | None = None,
) -> tuple[str, ...]:
    base_names = get_base_encoding_glyph_names(
        "StandardEncoding" if authoritative_builtin else base_encoding or "StandardEncoding",
        context=context,
    )
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
    "strip_subset_tag",
    "parse_differences",
    "build_simple_encoding_glyph_names",
    "STANDARD_ENCODING_GLYPH_NAMES",
    "MAC_ROMAN_ENCODING_GLYPH_NAMES",
    "WIN_ANSI_ENCODING_GLYPH_NAMES",
]
