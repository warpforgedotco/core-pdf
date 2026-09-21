# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, cast

from core_pdf.impl._impl.document.records import RawFormField
from core_pdf.impl._impl.document.recovery.text_strings import decode_pdf_text_string
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_document.fields import (
    field_children,
    qualified_field_name,
)
from core_pdf_spec.s_07_syntax.inherited_values import inherited_dictionary_value
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver


class FieldResolver(PdfValueResolver, Protocol):
    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None: ...


FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]
FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]
FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_value_text(resolver: FieldResolver, value: object) -> str:
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
            continue
        if item_text:
            parts.append(item_text)
    return "\n".join(parts)


def internal_field_record(
    resolver: FieldResolver,
    node: PdfDict,
    parent_name: str,
    parent_type: str,
    parent_value: object,
    *,
    recover: bool,
    terminal_widget: bool = False,
) -> RawFormField:
    title = resolver.resolve_str(node.get("T"))
    name = qualified_field_name(parent_name, title)
    field_type = resolver.resolve_name_or_text(node.get("FT"), name_like=True) or parent_type
    value = cast(PdfObject, inherited_dictionary_value(node, "V", parent_value, resolver.resolve))
    value_text = field_value_text(resolver, value)
    try:
        kids = field_children(None if terminal_widget else node.get("Kids"))
    except ValueError:
        if not recover:
            raise
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
    resolver: FieldResolver,
    node: object,
    *,
    recover: bool,
) -> list[RawFormField]:
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
