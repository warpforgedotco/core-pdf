# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.recovery_text_strings import parse_text_string
from core_pdf.impl.types import PdfReference
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver


def resolve_annotation_dict(resolver: PdfValueResolver, value: object) -> PdfDict | None:
    if isinstance(value, PdfReference):
        value = resolver.resolve(value)
    return value if isinstance(value, dict) else None


def link_target(resolver: PdfValueResolver, action: PdfDict, link_type: str | None) -> str | None:
    key = "URI" if link_type == "URI" else "D" if link_type == "GoTo" else None
    if key is None:
        return None
    target = action.get(key)
    text = parse_text_string(target)
    return resolver.resolve_str(target) if text is None else text


def goto_action_destination(resolver: PdfValueResolver, action: object) -> PdfObject:
    if isinstance(action, dict) and resolver.resolve_name(action.get("S")) == "GoTo":
        return action.get("D")
    return None


__all__ = (
    "goto_action_destination",
    "link_target",
    "resolve_annotation_dict",
)
