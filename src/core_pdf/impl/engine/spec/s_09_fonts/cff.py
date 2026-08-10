# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from contextlib import suppress
from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.engine.spec.s_09_fonts.font_program import (  # noqa: F401
    EMPTY_FEATURE,
    STANDARD_GLYPH_SIDS,
    CFFFont,
    CFFGlyphFeature,
    CFFUnicodeRepairIndex,
    cff_font_for_data,
    cff_unicode_repair_index_for_data,
    glyph_feature_distance,
    is_repairable_to_unicode_label,
)
from core_pdf.impl.engine.spec.s_09_fonts.font_program import (
    internal_type2_glyph_geometry_impl as internal_type2_glyph_geometry,  # noqa: F401
)
from core_pdf.impl.engine.spec.s_09_fonts.font_program_truetype import FONT_PROGRAM_ERRORS
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant
from core_pdf.impl.objects import PdfStream


def single_code_mapping(
    to_unicode: ToUnicodeCMap, cmap: CMapDecoder | None, limit: int | None = None
) -> dict[bytes, tuple[int, str]]:
    mapping: dict[bytes, tuple[int, str]] = {}
    for code_bytes, value in to_unicode.mappings.items():
        if len(code_bytes) not in {1, 2}:
            continue
        cid = int.from_bytes(code_bytes, "big")
        if cmap is not None:
            decoded = cmap.decode_entries(code_bytes)
            if len(decoded) == 1 and decoded[0][0] == code_bytes:
                cid = decoded[0][1]
        if limit is not None and cid >= limit:
            continue
        mapping.setdefault(code_bytes, (cid, value))
    return mapping


def cff_font_for_pdf_font(font: dict[str, Any]) -> CFFFont | None:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    font_subtype = normalize_pdf_name(lookup_dict_key(font_dict, "Subtype"))
    if descendant is not None:
        if font_subtype != "CIDFontType0":
            return None
    elif font_subtype not in {"Type1", "MMType1"}:
        return None
    descriptor = lookup_dict_key(font_dict, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = lookup_dict_key(descriptor, "FontFile3")
    if not isinstance(font_file, PdfStream):
        return None
    subtype = normalize_pdf_name(lookup_dict_key(font_file.dictionary, "Subtype"))
    if descendant is None and subtype != "Type1C":
        return None
    font_data: bytes | None = font_file.data
    if subtype == "OpenType":
        if font_data is None:
            return None
        font_data = internal_extract_cff_table(font_data)
        if font_data is None:
            return None
    try:
        return cff_font_for_data(font_data)
    except ValueError:
        return None


def build_cff_unicode_repair_index(
    font: dict[str, Any], to_unicode: ToUnicodeCMap | None, cmap: CMapDecoder | None
) -> CFFUnicodeRepairIndex | None:
    if to_unicode is None:
        return None
    descendant = get_descendant(font)
    if descendant is None:
        return None
    if normalize_pdf_name(lookup_dict_key(descendant, "Subtype")) != "CIDFontType0":
        return None
    descriptor = lookup_dict_key(descendant, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = lookup_dict_key(descriptor, "FontFile3")
    if not isinstance(font_file, PdfStream) or len(font_file.data) > 750_000:
        return None
    subtype = normalize_pdf_name(lookup_dict_key(font_file.dictionary, "Subtype"))
    font_data: bytes | None = font_file.data
    if subtype == "OpenType":
        if font_data is None:
            return None
        font_data = internal_extract_cff_table(font_data)
        if font_data is None:
            return None
    mapping = single_code_mapping(to_unicode, cmap)
    if not any(is_repairable_to_unicode_label(value) for internal_cid, value in mapping.values()):
        return None
    try:
        return cff_unicode_repair_index_for_data(
            font_data, tuple(sorted((code, cid, value) for code, (cid, value) in mapping.items()))
        )
    except ValueError:
        return None


def internal_extract_cff_table(data: bytes) -> bytes | None:
    font: TTFont | None = None
    try:
        font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
        reader = font.reader
        if reader is None:
            return None
        table = reader.tables.get("CFF ")
        return getattr(table, "data") if table is not None else None
    except FONT_PROGRAM_ERRORS:
        return None
    finally:
        if font is not None:
            with suppress(AttributeError):
                font.close()


__all__ = ("build_cff_unicode_repair_index", "cff_font_for_pdf_font")
