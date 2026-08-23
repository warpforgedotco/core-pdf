# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF link and destination helpers."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_float_strict
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_objects.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.types import PdfDict

LinkResolver = PdfValueResolver


def pdf_name_direct(value: object) -> str | None:
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, str):
        return value.lstrip("/")
    return None


def pdf_box_direct(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return (
            parse_float_strict(value[0]),
            parse_float_strict(value[1]),
            parse_float_strict(value[2]),
            parse_float_strict(value[3]),
        )
    except ValueError:
        return None


def pdf_string_direct(value: object) -> str | None:
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, bytes):
        return decode_pdf_text_string(value)
    if isinstance(value, str):
        return value
    return None


def resolve_annotation_dict(resolver: LinkResolver, value: object) -> PdfDict | None:
    if isinstance(value, PdfReference):
        value = resolver.resolve(value)
    return cast(PdfDict, value) if isinstance(value, dict) else None


def link_target_direct(action: PdfDict, link_type: str | None) -> str | None:
    if link_type == "URI":
        return pdf_string_direct(lookup_dict_key(action, "URI"))
    if link_type == "GoTo":
        return pdf_string_direct(lookup_dict_key(action, "D"))
    return None


def link_target_resolved(
    resolver: LinkResolver, action: PdfDict, link_type: str | None
) -> str | None:
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    if key is None:
        return None
    return resolver.resolve_str(lookup_dict_key(action, key))


__all__ = (
    "link_target_direct",
    "link_target_resolved",
    "pdf_box_direct",
    "pdf_name_direct",
    "resolve_annotation_dict",
)
