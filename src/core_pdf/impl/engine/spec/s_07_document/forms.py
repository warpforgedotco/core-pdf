# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_document.document_lock import (
    document_cache_lock,
    document_recovery_enabled,
    get_or_compute,
)
from core_pdf.impl.engine.spec.s_07_document.fields import (
    FieldTraversalEntry,
    field_value_text,
    field_widget_rect,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.models import RawFormField
from core_pdf.impl.types import PdfDict, PdfObject


class FormsMixin:
    __slots__ = ()

    acroform_cache: PdfDict | None
    fields_cache: list[RawFormField] | None

    @property
    def acroform(self: Any) -> PdfDict | None:
        def compute() -> PdfDict | None:
            acroform_val = self.resolver.resolve(lookup_dict_key(self.catalog(), "AcroForm"))
            if acroform_val is None:
                return None
            if isinstance(acroform_val, dict):
                return cast(PdfDict, acroform_val)
            if document_recovery_enabled(self):
                return None
            raise ValueError("invalid AcroForm dictionary")

        return get_or_compute(self, "acroform_cache", compute)

    def collect_field_records(
        self: Any,
        node: object,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value: object = None,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[RawFormField]:
        recover = document_recovery_enabled(self)
        if seen is None:
            seen = set()
        records: list[RawFormField] = []
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
                RawFormField(
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
                            RawFormField(
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

    def fields(self: Any) -> list[RawFormField]:
        with document_cache_lock(self):
            if self.fields_cache is not None:
                return self.fields_cache
            af = self.acroform
            records: list[RawFormField] = []
            if af is not None:
                field_list = lookup_dict_key(af, "Fields")
                if field_list is None:
                    field_list = []
                elif not isinstance(field_list, list):
                    if document_recovery_enabled(self):
                        field_list = []
                    else:
                        raise ValueError("invalid AcroForm Fields array")
                for field in field_list:
                    field_obj = self.resolver.resolve(field)
                    records.extend(self.collect_field_records(field_obj))
            # 12.5.6.19 lets a field with a single widget merge both
            # dictionaries into one, so a widget carrying /FT is itself a field
            # and a missing or empty catalog field tree does not mean the
            # document has none -- producers do ship filled forms that way.
            # Fall back to the pages when the tree tells us nothing, which also
            # keeps well-formed documents clear of a whole-page scan.
            if not records or document_recovery_enabled(self):
                records.extend(self.discover_widget_field_records(records))
            self.fields_cache = records
            return records

    def discover_widget_field_records(
        self: Any, existing: list[RawFormField]
    ) -> list[RawFormField]:
        seen_widgets = {id(record.widget) for record in existing if isinstance(record.widget, dict)}
        records: list[RawFormField] = []
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
                # A widget may be merged with its field or hang off one as a
                # kid. Collect from the root of the chain either way, so the
                # /FT, /T and /V a split field keeps on the parent still reach
                # the record.
                root = self.internal_widget_field_root(annot)
                if id(root) in seen_widgets:
                    continue
                seen_widgets.add(id(root))
                seen_widgets.add(id(annot))
                records.extend(self.collect_field_records(root))
        return records

    def internal_widget_field_root(self: Any, annot: PdfDict) -> PdfDict:
        node = annot
        for _ in range(50):
            parent = self.resolver.resolve(lookup_dict_key(node, "Parent"))
            if not isinstance(parent, dict) or parent is node:
                break
            if lookup_dict_key(parent, "FT") is None and lookup_dict_key(parent, "T") is None:
                break
            node = cast(PdfDict, parent)
        return node
