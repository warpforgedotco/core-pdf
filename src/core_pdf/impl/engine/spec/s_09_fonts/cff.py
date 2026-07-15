# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.third_party.cff import (
    REPAIRABLE_TO_UNICODE,
    CFFFont,
    cff_font_for_data,
    cff_unicode_repairs_for_data,
)
from core_pdf.impl.third_party.cid.cmap import CMapDecoder, ToUnicodeCMap


def single_code_mapping(
    to_unicode: ToUnicodeCMap, cmap: CMapDecoder | None, limit: int | None = None
) -> dict[bytes, tuple[int, str]]:
    mapping: dict[bytes, tuple[int, str]] = {}
    for code_bytes, value in to_unicode.mappings.items():
        if len(code_bytes) not in {1, 2}:
            continue
        cid = int.from_bytes(code_bytes, "big")
        if cmap is not None:
            decoded = cmap.decode(code_bytes)
            if len(decoded) == 1 and decoded[0][0] == code_bytes:
                cid = decoded[0][1]
        if limit is not None and cid >= limit:
            continue
        mapping.setdefault(code_bytes, (cid, value))
    return mapping


def cff_font_for_pdf_font(font: dict[str, Any]) -> CFFFont | None:
    descendant = get_descendant(font)
    if descendant is None:
        return None
    if normalize_pdf_name(lookup_dict_key(descendant, "Subtype")) != "CIDFontType0":
        return None
    descriptor = lookup_dict_key(descendant, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = lookup_dict_key(descriptor, "FontFile3")
    if not isinstance(font_file, PdfStream):
        return None
    try:
        return cff_font_for_data(font_file.data)
    except ValueError:
        return None


def build_cff_unicode_repairs(
    font: dict[str, Any], to_unicode: ToUnicodeCMap | None, cmap: CMapDecoder | None
) -> dict[bytes, str]:
    if to_unicode is None:
        return {}
    descendant = get_descendant(font)
    if descendant is None:
        return {}
    if normalize_pdf_name(lookup_dict_key(descendant, "Subtype")) != "CIDFontType0":
        return {}
    descriptor = lookup_dict_key(descendant, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return {}
    font_file = lookup_dict_key(descriptor, "FontFile3")
    if not isinstance(font_file, PdfStream):
        return {}
    if len(font_file.data) > 750_000:
        return {}
    mapping = single_code_mapping(to_unicode, cmap)
    if not any(value in REPAIRABLE_TO_UNICODE for ignored_cid, value in mapping.values()):
        return {}
    try:
        repair_items = cff_unicode_repairs_for_data(
            font_file.data,
            tuple(sorted((code_bytes, cid, value) for code_bytes, (cid, value) in mapping.items())),
        )
    except ValueError:
        return {}
    return dict(repair_items)


__all__ = (
    "build_cff_unicode_repairs",
    "cff_font_for_pdf_font",
    "single_code_mapping",
)
