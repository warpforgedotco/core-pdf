# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.third_party.truetype import TrueTypeFontProgram
from core_pdf.impl.third_party.truetype import tt_font_for_data as _tt_font_for_data


def tt_font_for_pdf_font(font: dict[str, Any]) -> TrueTypeFontProgram | None:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    subtype = normalize_pdf_name(lookup_dict_key(font_dict, "Subtype"))
    if subtype not in {"CIDFontType2", "TrueType"}:
        return None
    descriptor = lookup_dict_key(font_dict, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = lookup_dict_key(descriptor, "FontFile2")
    if not isinstance(font_file, PdfStream):
        return None
    cid_to_gid = None
    if isinstance(descendant, dict):
        cid_to_gid_obj = lookup_dict_key(descendant, "CIDToGIDMap")
        if isinstance(cid_to_gid_obj, PdfStream):
            cid_to_gid = cid_to_gid_obj.data
    try:
        return _tt_font_for_data(font_file.data, cid_to_gid, use_cmap=descendant is None)
    except ValueError:
        return None


__all__ = ("tt_font_for_pdf_font",)
