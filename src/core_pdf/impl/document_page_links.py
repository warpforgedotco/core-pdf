# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.recovery_text_strings import (
    decode_pdf_text_string,
    parse_text_string,
)
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver


def resolve_annotation_dict(resolver: PdfValueResolver, value: object) -> PdfDict | None:
    if isinstance(value, PdfReference):
        value = resolver.resolve(value)
    return value if isinstance(value, dict) else None


def link_target(resolver: PdfValueResolver, action: PdfDict, link_type: str | None) -> str | None:
    """A URI action's URI or a GoTo action's destination, as text."""
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    if key is None:
        return None
    target = action.get(key)
    text = parse_text_string(target)
    return resolver.resolve_str(target) if text is None else text


def goto_action_destination(resolver: PdfValueResolver, action: object) -> PdfObject:
    """The destination of a GoTo action dictionary, or None."""
    if isinstance(action, dict) and resolver.resolve_name(action.get("S")) == "GoTo":
        return action.get("D")
    return None


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
    "goto_action_destination",
    "link_target",
    "resolve_annotation_dict",
    "resolve_destination_value",
)
