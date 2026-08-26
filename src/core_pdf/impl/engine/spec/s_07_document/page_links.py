# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF link and destination helpers."""

from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.spec.s_07_syntax.coercion import parse_float_strict
from core_pdf.impl.engine.spec.s_07_syntax.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.resolver_values import PdfValueResolver
from core_pdf.impl.engine.spec.s_07_syntax.text_string import decode_pdf_text_string
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.types import PdfDict


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


def resolve_annotation_dict(resolver: PdfValueResolver, value: object) -> PdfDict | None:
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
    resolver: PdfValueResolver, action: PdfDict, link_type: str | None
) -> str | None:
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    if key is None:
        return None
    return resolver.resolve_str(lookup_dict_key(action, key))


def resolve_destination_value(resolver: PdfValueResolver, value: object, depth: int = 0) -> object:
    """Resolve an annotation destination into plain serializable values.

    A URI action holds its target behind an indirect reference, so the raw
    action dictionary carries a ``PdfReference`` where the URL belongs.  This
    resolves references to scalars and unwraps PDF strings and names.

    References to composite objects are deliberately left as references: a
    GoTo destination points at a page, and inlining a page dictionary here
    would drag the object graph into the structured document.
    """
    if depth > 8:
        return value
    if isinstance(value, PdfReference):
        resolved = resolver.resolve(value)
        if resolved is None or isinstance(resolved, (dict, list, tuple)):
            return value
        return resolve_destination_value(resolver, resolved, depth + 1)
    if isinstance(value, PdfString):
        return decode_pdf_text_string(value.data)
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): resolve_destination_value(resolver, item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [resolve_destination_value(resolver, item, depth + 1) for item in value]
    return value


__all__ = (
    "link_target_direct",
    "link_target_resolved",
    "pdf_box_direct",
    "pdf_name_direct",
    "resolve_annotation_dict",
    "resolve_destination_value",
)
