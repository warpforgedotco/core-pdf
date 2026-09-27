# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts_cmap_resources import resolve_cmap_decoder, resolve_cmap_resource
from core_pdf.impl.fonts_cmap_tokenizer import CMapDecoder
from core_pdf.impl.fonts_glyphs import glyph_name_to_unicode
from core_pdf.impl.fonts_helpers import (
    build_decode_table,
    build_simple_encoding_glyph_names,
    recover_differences,
    recover_strip_subset_tag,
    unicode_for_glyph_name,
)
from core_pdf.impl.fonts_program_base import GlyphProgram
from core_pdf.impl.fonts_program_type1 import parse_type1_font_program_encoding
from core_pdf.impl.fonts_widths import recover_descendant
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfString
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_09_fonts.helpers import BASE_ENCODING_GLYPH_NAMES
from core_pdf_spec.standards import SemanticContext

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


def cid_system_info_string(value: object) -> str | None:
    if isinstance(value, PdfString):
        return value.data.decode("latin-1")
    normalized = recover_pdf_name(value)
    if normalized is not None:
        return normalized
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return None


def cid_system_info(font: dict[str, Any]) -> tuple[str | None, str | None]:
    descendant = recover_descendant(font)
    system_info = descendant.get("CIDSystemInfo") if descendant else None
    if not isinstance(system_info, dict):
        system_info = font.get("CIDSystemInfo")
    if not isinstance(system_info, dict):
        return None, None
    registry = cid_system_info_string(system_info.get("Registry"))
    ordering = cid_system_info_string(system_info.get("Ordering"))
    return registry, ordering


def named_cmap(base_encoding: str | None) -> CMapDecoder | None:
    if base_encoding is None:
        return None
    return resolve_cmap_decoder(base_encoding)


def builtin_font_encoding(
    font: dict[str, Any], program: GlyphProgram
) -> tuple[dict[int, str], bool]:
    builtin = program.font_builtin_encoding()
    if builtin is not None:
        return builtin
    descriptor = font.get("FontDescriptor")
    if not isinstance(descriptor, dict):
        return {}, False
    font_file = descriptor.get("FontFile")
    if isinstance(font_file, PdfStream):
        try:
            encoding = parse_type1_font_program_encoding(font_file.data)
            return encoding, bool(encoding)
        except PdfParseError, ValueError:
            return {}, False
    return {}, False


def parse_encoding(
    font: dict[str, Any], program: GlyphProgram
) -> tuple[CMapDecoder | None, str | None, dict[int, str], dict[int, str], bool]:
    cmap = None
    base_encoding = None
    base_encoding_explicit = False
    differences: dict[int, str] = {}
    subtype = recover_pdf_name(font.get("Subtype"))
    encoding_obj = font.get("Encoding")
    match encoding_obj:
        case PdfStream():
            try:
                cmap = CMapDecoder(
                    encoding_obj.data,
                    usecmap_resolver=resolve_cmap_resource,
                )
            except PdfParseError, ValueError:
                cmap = None
        case dict():
            base_encoding = recover_pdf_name(encoding_obj.get("BaseEncoding"))
            if base_encoding is None:
                base_encoding = "WinAnsiEncoding" if subtype == "TrueType" else "StandardEncoding"
            else:
                base_encoding_explicit = True
            differences_obj = encoding_obj.get("Differences")
            if differences_obj is not None and not isinstance(differences_obj, (list, tuple)):
                differences_obj = None
            differences = recover_differences(
                list(differences_obj) if isinstance(differences_obj, tuple) else differences_obj,
                recover_pdf_name,
            )
        case _:
            base_encoding = recover_pdf_name(encoding_obj)
            base_encoding_explicit = base_encoding is not None
            cmap = named_cmap(base_encoding)
    if base_encoding is None and subtype == "Type3":
        base_encoding = "StandardEncoding"
    builtin: dict[int, str] = {}
    builtin_authoritative = False
    if subtype in ("Type1", "MMType1") and not base_encoding_explicit:
        builtin, builtin_authoritative = builtin_font_encoding(font, program)
    if base_encoding is None and subtype in ("Type1", "MMType1"):
        base_encoding = "StandardEncoding"
    return cmap, base_encoding, differences, builtin, builtin_authoritative


class FontEncoding:
    __slots__ = (
        "is_cid_font",
        "cmap",
        "base_encoding",
        "differences",
        "encoding_differences",
        "simple_encoding_glyph_names",
        "encoding_decode_table",
        "glyph_decode_table",
        "glyph_decode_table_authoritative",
        "cid_registry",
        "cid_ordering",
    )

    is_cid_font: bool
    cmap: CMapDecoder | None
    base_encoding: str | None
    differences: dict[int, str]
    encoding_differences: dict[int, str]
    simple_encoding_glyph_names: tuple[str, ...]
    encoding_decode_table: tuple[str, ...]
    glyph_decode_table: tuple[str, ...] | None
    glyph_decode_table_authoritative: bool
    cid_registry: str | None
    cid_ordering: str | None

    def __init__(
        self,
        font: dict[str, Any],
        program: GlyphProgram,
        *,
        is_cid_font: bool,
        is_type3: bool,
        base_font_name: str | None,
        semantic_context: SemanticContext | None,
    ) -> None:
        (
            cmap,
            base_encoding,
            differences,
            builtin_encoding,
            builtin_encoding_authoritative,
        ) = parse_encoding(font, program)
        simple_encoding_glyph_names = build_simple_encoding_glyph_names(
            base_encoding if base_encoding in BASE_ENCODING_GLYPH_NAMES else "StandardEncoding",
            builtin_encoding,
            differences,
            authoritative_builtin=builtin_encoding_authoritative,
            context=semantic_context,
        )
        if builtin_encoding_authoritative:
            encoding_decode_table = tuple(
                unicode_for_glyph_name(name) or "" for name in simple_encoding_glyph_names
            )
        else:
            key = base_encoding or ("Type3" if is_type3 else "")
            encoding_decode_table = build_decode_table(key, differences, context=semantic_context)
        self.is_cid_font = is_cid_font
        self.cmap = cmap
        self.cid_registry, self.cid_ordering = cid_system_info(font)
        self.base_encoding = base_encoding
        self.differences = differences
        self.encoding_differences = (
            {**builtin_encoding, **differences} if builtin_encoding else differences
        )
        self.simple_encoding_glyph_names = simple_encoding_glyph_names
        self.encoding_decode_table = encoding_decode_table
        glyph_decode = build_glyph_decode_table(base_font_name, differences)
        if glyph_decode is None:
            self.glyph_decode_table = None
            self.glyph_decode_table_authoritative = False
        else:
            self.glyph_decode_table, self.glyph_decode_table_authoritative = glyph_decode

    def glyph_name(self, code: int) -> str:
        if 0 <= code < 256:
            return self.simple_encoding_glyph_names[code]
        return ".notdef"
