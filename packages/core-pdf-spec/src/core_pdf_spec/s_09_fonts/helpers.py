"""PDF font encoding names and dictionary semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from core_pdf_spec._vendor.font_data.encoding_names import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name

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
            if type(code) is not int or not 0 <= code < 256 or not name:
                raise ValueError("invalid simple-font encoding entry")
            names[code] = name
    return tuple(names)


__all__ = [
    "BASE_ENCODING_GLYPH_NAMES",
    "base_encoding_glyph_names",
    "strip_subset_tag",
    "parse_differences",
    "build_simple_encoding_glyph_names",
    "STANDARD_ENCODING_GLYPH_NAMES",
    "MAC_ROMAN_ENCODING_GLYPH_NAMES",
    "WIN_ANSI_ENCODING_GLYPH_NAMES",
]
