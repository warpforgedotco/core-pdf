# SPDX-License-Identifier: AGPL-3.0-only
"""Native field-value and widget geometry helpers."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.models import RawFormField
from core_pdf.impl.objects import PdfName, PdfReference, PdfString
from core_pdf.impl.types import PdfDict

FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]
FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]
FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_widget_rect(
    document: Any, widget: PdfDict | None
) -> tuple[float, float, float, float] | None:
    if widget is None:
        return None
    return document.resolver.resolve_box(lookup_dict_key(widget, "Rect"))


def field_value_text(document: Any, value: object) -> str:
    parts: list[str] = []
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        current = (
            document.resolver.resolve(current) if isinstance(current, PdfReference) else current
        )
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue
        if isinstance(current, PdfName):
            item_text = current.value
        elif isinstance(current, PdfString):
            item_text = decode_pdf_text_string(current.data).strip()
        elif isinstance(current, bytes):
            item_text = decode_pdf_text_string(current).strip()
        elif isinstance(current, str):
            item_text = current.strip()
        else:
            item_text = str(current).strip()
        if item_text:
            parts.append(item_text)
    return "\n".join(parts)
