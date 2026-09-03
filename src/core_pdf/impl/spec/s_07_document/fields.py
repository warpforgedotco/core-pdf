# SPDX-License-Identifier: AGPL-3.0-only
"""Native field-value and widget geometry helpers."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.spec.s_07_document.records import RawFormField
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string

FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]
FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]
FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_widget_rect(
    document: Any, widget: PdfDict | None
) -> tuple[float, float, float, float] | None:
    if widget is None:
        return None
    return document.resolver.resolve_box(widget.get("Rect"))


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
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            item_text = str(current)
        else:
            # AcroForm values are text strings, names, or arrays of those
            # values.  Signature fields instead store a signature dictionary
            # in /V; coercing that dictionary (or another malformed composite
            # value) to ``str`` leaks PDF object syntax into extracted text.
            continue
        if item_text:
            parts.append(item_text)
    return "\n".join(parts)
