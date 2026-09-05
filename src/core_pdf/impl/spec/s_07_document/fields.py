# SPDX-License-Identifier: AGPL-3.0-only
"""AcroForm field inheritance, traversal, and record construction."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

from core_pdf.impl.spec.s_07_document.records import RawFormField
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf.impl.types import PdfName, PdfReference, PdfString

FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]
FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]
FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_value_text(resolver: PdfValueResolver, value: object) -> str:
    parts: list[str] = []
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        current = resolver.resolve(current) if isinstance(current, PdfReference) else current
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


def internal_field_record(
    resolver: PdfValueResolver,
    node: PdfDict,
    parent_name: str,
    parent_type: str,
    parent_value: object,
    *,
    recover: bool,
    terminal_widget: bool = False,
) -> RawFormField:
    title = resolver.resolve_str(node.get("T"))
    name = f"{parent_name}.{title}" if parent_name and title else title or parent_name
    field_type = resolver.resolve_name_or_text(node.get("FT"), name_like=True) or parent_type
    value = node.get("V")
    if value is None:
        value = cast(PdfObject, parent_value)
    value_text = field_value_text(resolver, value)
    kids = [] if terminal_widget else node.get("Kids")
    if kids is None:
        kids = []
    elif not isinstance(kids, list):
        if not recover:
            raise ValueError("invalid AcroForm Kids array")
        kids = []
    is_widget = terminal_widget or resolver.resolve_name_or_text(node.get("Subtype")) == "Widget"
    return RawFormField(
        name,
        field_type,
        value,
        value_text,
        resolver.resolve_box(node.get("Rect")) if is_widget else None,
        node,
        kids=kids,
        widget=node if is_widget else None,
    )


def collect_field_records(
    resolver: PdfValueResolver,
    node: object,
    *,
    recover: bool,
) -> list[RawFormField]:
    """Collect fields parent-first, retaining terminal widgets as leaf records.

    Only field nodes participate in depth/cycle checks: a widget child is a
    terminal annotation even if a malformed producer supplies it with Kids.
    """
    seen: set[int] = set()
    records: list[RawFormField] = []
    stack: list[FieldTraversalEntry] = [("node", node, "", "", None, 0)]
    while stack:
        entry = stack.pop()
        if entry[0] == "record":
            records.append(entry[1])
            continue
        _, current_node, parent_name, parent_type, parent_value, depth = entry
        if depth > 50:
            if recover:
                continue
            raise ValueError("invalid AcroForm depth")
        current_node = resolver.resolve(current_node)
        if not isinstance(current_node, dict) or id(current_node) in seen:
            if recover:
                continue
            raise ValueError("invalid AcroForm field entry")
        seen.add(id(current_node))
        current_node = cast(PdfDict, current_node)
        record = internal_field_record(
            resolver,
            current_node,
            parent_name,
            parent_type,
            parent_value,
            recover=recover,
        )
        records.append(record)
        for kid in reversed(record.kids):
            resolved_kid = resolver.resolve(kid)
            if not isinstance(resolved_kid, dict):
                if recover:
                    continue
                raise ValueError("invalid AcroForm kid entry")
            resolved_kid = cast(PdfDict, resolved_kid)
            if resolver.resolve_name_or_text(resolved_kid.get("Subtype")) == "Widget":
                stack.append(
                    (
                        "record",
                        internal_field_record(
                            resolver,
                            resolved_kid,
                            record.name,
                            record.type,
                            record.value,
                            recover=recover,
                            terminal_widget=True,
                        ),
                    )
                )
            else:
                stack.append(
                    ("node", resolved_kid, record.name, record.type, record.value, depth + 1)
                )
    return records
