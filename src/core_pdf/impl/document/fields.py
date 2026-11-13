# SPDX-License-Identifier: AGPL-3.0-only

"""The interactive form: field records, and fields grouped by page.

The PdfDocument methods here are functions taking the document as self;
PdfDocument lists them as its methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from core_pdf.impl.base.raw_records import RawFormField
from core_pdf.impl.base.types import PdfName, PdfString
from core_pdf.impl.base.values import parse_text_string
from core_pdf.impl.document.pdf_page import PdfPage
from core_pdf.impl.recovery.policy import MalformedFn
from core_pdf.impl.spec.s_07_syntax.inherited_values import inherited_dictionary_value
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
lazy from core_pdf.impl.spec.s_07_document.fields import field_children, qualified_field_name

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PageT, PdfDocument


class FieldResolver(PdfValueResolver, Protocol):
    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None: ...

    def as_dict(self, value: object) -> PdfDict | None: ...


FieldTraversalNode: TypeAlias = tuple[Literal["node"], object, str, str, object, int]


FieldTraversalRecord: TypeAlias = tuple[Literal["record"], RawFormField]


FieldTraversalEntry: TypeAlias = FieldTraversalNode | FieldTraversalRecord


def field_value_text(resolver: FieldResolver, value: object) -> str:
    parts: list[str] = []
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        current = resolver.resolve(current)
        match current:
            case None:
                continue
            case list() | tuple():
                stack.extend(reversed(current))
                continue
            case PdfName(value=item_text):
                pass
            case PdfString() | bytes() | str():
                item_text = (parse_text_string(current) or "").strip()
            case int() | float() if type(current) is not bool:
                item_text = str(current)
            case _:
                continue
        if item_text:
            parts.append(item_text)
    return "\n".join(parts)


def field_record(
    resolver: FieldResolver,
    node: PdfDict,
    parent_name: str,
    parent_type: str,
    parent_value: object,
    malformed: MalformedFn,
    *,
    terminal_widget: bool = False,
) -> RawFormField:
    title = resolver.resolve_str(node.get("T"))
    name = qualified_field_name(parent_name, title)
    field_type = resolver.resolve_name_or_text(node.get("FT"), name_like=True) or parent_type
    value = inherited_dictionary_value(node, "V", parent_value, resolver.resolve)
    value_text = field_value_text(resolver, value)
    try:
        kids = field_children(None if terminal_widget else node.get("Kids"))
    except ValueError as error:
        malformed(str(error))
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
    resolver: FieldResolver, node: object, malformed: MalformedFn
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
            malformed("invalid AcroForm depth")
            continue
        current_node = resolver.as_dict(current_node)
        if current_node is None or id(current_node) in seen:
            malformed("invalid AcroForm field entry")
            continue
        seen.add(id(current_node))
        record = field_record(
            resolver, current_node, parent_name, parent_type, parent_value, malformed
        )
        records.append(record)
        for kid in reversed(record.kids):
            resolved_kid = resolver.as_dict(kid)
            if resolved_kid is None:
                malformed("invalid AcroForm kid entry")
                continue
            if resolver.resolve_name_or_text(resolved_kid.get("Subtype")) == "Widget":
                stack.append(
                    (
                        "record",
                        field_record(
                            resolver,
                            resolved_kid,
                            record.name,
                            record.type,
                            record.value,
                            malformed,
                            terminal_widget=True,
                        ),
                    )
                )
            else:
                stack.append(
                    ("node", resolved_kid, record.name, record.type, record.value, depth + 1)
                )
    return records


def fields(self: PdfDocument[Any]) -> list[RawFormField]:
    af = self.acroform
    records: list[RawFormField] = []
    if af is not None:
        field_list = af.get("Fields")
        if field_list is None:
            field_list = []
        elif not isinstance(field_list, list):
            self.malformed("invalid AcroForm Fields array")
            field_list = []
        for field in field_list:
            field_obj = self.resolver.resolve(field)
            records.extend(collect_field_records(self.resolver, field_obj, self.recovery.malformed))
    if not records or self.recovery_enabled:
        records.extend(self.discover_widget_field_records(records))
    return records


def fields_by_page(
    self: PdfDocument[PageT], pages: Sequence[PageT] | None = None
) -> dict[int, list[RawFormField]]:
    if pages is not None:
        return self.group_fields_by_page(tuple(pages))
    return {page_index: list(fields) for page_index, fields in self.cached_fields_by_page().items()}


def cached_fields_by_page(self: PdfDocument[Any]) -> dict[int, list[RawFormField]]:
    return self.caches.get("fields_by_page", self.group_all_fields_by_page)


def group_all_fields_by_page(self: PdfDocument[Any]) -> dict[int, list[RawFormField]]:
    return self.group_fields_by_page(self.pages)


def group_fields_by_page(
    self: PdfDocument[PageT], page_sequence: tuple[PageT, ...]
) -> dict[int, list[RawFormField]]:
    page_indexes_by_dict = {
        id(page.page_dict): page.page_number - 1
        for page in page_sequence
        if isinstance(page, PdfPage)
    }
    grouped: dict[int, list[RawFormField]] = {}
    annot_page_index: dict[int, int] | None = None

    def widget_page_index(widget: object) -> int | None:
        nonlocal annot_page_index
        pg_ref = widget.get("P") if isinstance(widget, dict) else None
        if pg_ref is not None:
            pg_obj = self.resolver.resolve(pg_ref)
            return page_indexes_by_dict.get(id(pg_obj)) if isinstance(pg_obj, dict) else None
        if annot_page_index is None:
            annot_page_index = {
                id(annot): page.page_number - 1
                for page in page_sequence
                if isinstance(page, PdfPage)
                for annot in page.annotation_dicts()
            }
        return annot_page_index.get(id(widget))

    for field in self.fields():
        page_indexes: set[int] = set()
        if field.widget:
            if not isinstance(field.widget, dict):
                raise ValueError("invalid field widget entry")
            page_index = widget_page_index(field.widget)
            if page_index is not None:
                page_indexes.add(page_index)
        elif field.kids:
            if not isinstance(field.kids, list):
                raise ValueError("invalid field kids array")
            for kid_ref in field.kids:
                kid = self.resolver.as_dict(kid_ref)
                if kid is not None and self.resolver.name_at(kid, "Subtype") == "Widget":
                    page_index = widget_page_index(kid)
                    if page_index is not None:
                        page_indexes.add(page_index)
        for page_index in page_indexes:
            grouped.setdefault(page_index, []).append(field)
    return grouped


def discover_widget_field_records(
    self: PdfDocument[Any], existing: list[RawFormField]
) -> list[RawFormField]:
    seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
    records: list[RawFormField] = []
    for page in self.pages:
        for annot in page.annotation_dicts():
            if id(annot) in seen_widgets:
                continue
            subtype = self.resolver.resolve_name_or_text(annot.get("Subtype")) or ""
            if subtype != "Widget":
                continue
            root = self.widget_field_root(annot)
            if id(root) in seen_widgets:
                continue
            seen_widgets.add(id(root))
            seen_widgets.add(id(annot))
            records.extend(collect_field_records(self.resolver, root, self.recovery.malformed))
    return records


def widget_field_root(self: PdfDocument[Any], annot: PdfDict) -> PdfDict:
    node = annot
    seen = {id(node)}
    for _ in range(50):
        parent = self.resolver.dict_at(node, "Parent")
        if parent is None or id(parent) in seen:
            break
        seen.add(id(parent))
        node = parent
    return node
