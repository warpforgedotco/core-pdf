# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
)
from core_pdf.impl.engine.spec.s_07_document.fields import (
    FieldTraversalEntry,
    field_value_text,
    field_widget_rect,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import FieldRecord
from core_pdf.impl.types import PdfDict, PdfObject


class FormsMixin:
    __slots__ = ()

    acroform_cache: PdfDict | None
    fields_cache: list[FieldRecord] | None

    @property
    def acroform(self: Any) -> PdfDict | None:
        with document_cache_lock(self):
            if self.acroform_cache is None:
                acroform_val = self.resolver.resolve(lookup_dict_key(self.catalog(), "AcroForm"))
                recover = document_recovery_enabled(self)
                if acroform_val is None:
                    self.acroform_cache = None
                elif isinstance(acroform_val, dict):
                    self.acroform_cache = cast(PdfDict, acroform_val)
                elif recover:
                    self.acroform_cache = None
                else:
                    raise ValueError("invalid AcroForm dictionary")
            return self.acroform_cache

    def collect_field_records(
        self: Any,
        node: object,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value: object = None,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[FieldRecord]:
        recover = document_recovery_enabled(self)
        if seen is None:
            seen = set()
        records: list[FieldRecord] = []
        stack: list[FieldTraversalEntry] = [
            ("node", node, inherited_name, inherited_type, inherited_value, depth)
        ]

        while stack:
            entry = stack.pop()
            if entry[0] == "record":
                records.append(entry[1])
                continue

            (
                ignored,
                current_node,
                parent_name,
                parent_type,
                parent_value,
                current_depth,
            ) = entry
            if current_depth > 50:
                if recover:
                    continue
                raise ValueError("invalid AcroForm depth")
            current_node = self.resolver.resolve(current_node)
            if not isinstance(current_node, dict):
                if recover:
                    continue
                raise ValueError("invalid AcroForm field entry")
            marker = id(current_node)
            if marker in seen:
                if recover:
                    continue
                raise ValueError("invalid AcroForm field entry")
            seen.add(marker)

            title = self.resolver.resolve_str(lookup_dict_key(current_node, "T"))
            current_name = (
                f"{parent_name}.{title}" if parent_name and title else title or parent_name
            )

            type_value = lookup_dict_key(current_node, "FT")
            field_type = (
                self.resolver.resolve_name(type_value)
                or self.resolver.resolve_name_like_value(type_value)
                or self.resolver.resolve_str(type_value)
                or parent_type
            )

            value = lookup_dict_key(current_node, "V")
            if value is None:
                value = parent_value
            value_text = field_value_text(self, value)

            kids = lookup_dict_key(current_node, "Kids")
            if kids is None:
                kids = []
            elif not isinstance(kids, list):
                if recover:
                    kids = []
                else:
                    raise ValueError("invalid AcroForm Kids array")
            kids = cast(list[PdfObject], kids)
            subtype_value = lookup_dict_key(current_node, "Subtype")
            subtype = (
                self.resolver.resolve_name(subtype_value)
                or self.resolver.resolve_str(subtype_value)
                or ""
            )
            current_node = cast(PdfDict, current_node)
            records.append(
                FieldRecord(
                    current_name,
                    field_type,
                    cast(PdfObject, value),
                    value_text,
                    field_widget_rect(self, current_node if subtype == "Widget" else None),
                    current_node,
                    kids=kids,
                    widget=current_node if subtype == "Widget" else None,
                )
            )

            for kid in reversed(kids):
                resolved_kid = self.resolver.resolve(kid)
                if not isinstance(resolved_kid, dict):
                    if recover:
                        continue
                    raise ValueError("invalid AcroForm kid entry")
                resolved_kid = cast(PdfDict, resolved_kid)
                subtype_value = lookup_dict_key(resolved_kid, "Subtype")
                subtype = (
                    self.resolver.resolve_name(subtype_value)
                    or self.resolver.resolve_str(subtype_value)
                    or ""
                )
                if subtype == "Widget":
                    stack.append(
                        (
                            "record",
                            FieldRecord(
                                current_name,
                                field_type,
                                cast(PdfObject, value),
                                value_text,
                                field_widget_rect(self, resolved_kid),
                                resolved_kid,
                                kids=[],
                                widget=resolved_kid,
                            ),
                        )
                    )
                else:
                    stack.append(
                        (
                            "node",
                            resolved_kid,
                            current_name,
                            field_type,
                            value,
                            current_depth + 1,
                        )
                    )
        return records

    def fields(self: Any) -> list[FieldRecord]:
        with document_cache_lock(self):
            if self.fields_cache is not None:
                return self.fields_cache
            af = self.acroform
            if af is None:
                self.fields_cache = []
                return []
            field_list = lookup_dict_key(af, "Fields")
            if field_list is None:
                field_list = []
            elif not isinstance(field_list, list):
                if document_recovery_enabled(self):
                    field_list = []
                else:
                    raise ValueError("invalid AcroForm Fields array")
            records: list[FieldRecord] = []
            for field in field_list:
                field_obj = self.resolver.resolve(field)
                records.extend(self.collect_field_records(field_obj))
            if document_recovery_enabled(self):
                records.extend(self.discover_widget_field_records(records))
            self.fields_cache = records
            return records

    def discover_widget_field_records(self: Any, existing: list[FieldRecord]) -> list[FieldRecord]:
        seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
        records: list[FieldRecord] = []
        for page_dict in self.iter_page_dicts():
            raw_annots = self.resolver.resolve(lookup_dict_key(page_dict, "Annots"))
            if raw_annots is None:
                continue
            annots = raw_annots if isinstance(raw_annots, list) else [raw_annots]
            for annot_ref in annots:
                annot = self.resolver.resolve(annot_ref)
                if not isinstance(annot, dict):
                    continue
                if id(annot) in seen_widgets:
                    continue
                subtype = (
                    self.resolver.resolve_name(lookup_dict_key(annot, "Subtype"))
                    or self.resolver.resolve_str(lookup_dict_key(annot, "Subtype"))
                    or ""
                )
                if subtype != "Widget":
                    continue
                seen_widgets.add(id(annot))
                records.extend(self.collect_field_records(annot))
        return records
