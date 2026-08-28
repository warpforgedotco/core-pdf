# SPDX-License-Identifier: AGPL-3.0-only
"""Native PDF font-name resolution helpers."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant


def descriptor_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    descriptor = lookup_dict_key(font, "FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_descriptor = lookup_dict_key(descendant, "FontDescriptor")
            descriptor = descendant_descriptor or descriptor
    if not isinstance(descriptor, dict):
        return None
    return normalize_pdf_name(lookup_dict_key(descriptor, "FontName"))


def resolve_base_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    base_font_name = normalize_pdf_name(lookup_dict_key(font, "BaseFont"))
    if base_font_name is not None:
        return base_font_name
    return descriptor_font_name(font, subtype)
