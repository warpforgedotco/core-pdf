# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import cast

from core_pdf.impl.document.recovery.text_strings import (
    decode_pdf_text_string,
    parse_text_string,
)
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver


def resolve_annotation_dict(resolver: PdfValueResolver, value: object) -> PdfDict | None:
    if isinstance(value, PdfReference):
        value = resolver.resolve(value)
    return cast(PdfDict, value) if isinstance(value, dict) else None


def link_target_direct(action: PdfDict, link_type: str | None) -> str | None:
    if link_type == "URI":
        return parse_text_string(action.get("URI"))
    if link_type == "GoTo":
        return parse_text_string(action.get("D"))
    return None


def link_target_resolved(
    resolver: PdfValueResolver, action: PdfDict, link_type: str | None
) -> str | None:
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    if key is None:
        return None
    return resolver.resolve_str(action.get(key))


def resolve_destination_value(resolver: PdfValueResolver, value: object, depth: int = 0) -> object:
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
    "resolve_annotation_dict",
    "resolve_destination_value",
)
