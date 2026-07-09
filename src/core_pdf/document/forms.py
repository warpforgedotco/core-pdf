from __future__ import annotations

from typing import Any, cast

from core_pdf.document.models import FieldRecord
from core_pdf.document.protocols import DocumentMixinProtocol


class FormsMixin:
    __slots__ = ()

    @property
    def acroform(self: DocumentMixinProtocol) -> dict | None:
        if self.acroform_cache is None:
            acroform_val = self.resolver.resolve(self.catalog().get("AcroForm"))
            if acroform_val is None:
                self.acroform_cache = None
            elif isinstance(acroform_val, dict):
                self.acroform_cache = acroform_val
            else:
                raise ValueError("invalid AcroForm dictionary")
        return self.acroform_cache

    def collect_field_records(
        self: DocumentMixinProtocol,
        node,
        inherited_name: str = "",
        inherited_type: str = "",
        inherited_value=None,
        _depth: int = 0,
    ) -> list[FieldRecord]:
        if _depth > 50:
            raise ValueError("invalid AcroForm depth")
        node = self.resolver.resolve(node)
        if not isinstance(node, dict):
            raise ValueError("invalid AcroForm field entry")

        title = self.resolver.resolve_str(node.get("T"))
        current_name = (
            f"{inherited_name}.{title}" if inherited_name and title else title or inherited_name
        )

        type_value = node.get("FT")
        field_type = (
            self.resolver.resolve_name(type_value)
            or self.resolver.resolve_name_like_value(type_value)
            or self.resolver.resolve_str(type_value)
            or inherited_type
        )

        value = node.get("V")
        if value is None:
            value = inherited_value

        kids = node.get("Kids")
        if kids is None:
            kids = []
        elif not isinstance(kids, list):
            raise ValueError("invalid AcroForm Kids array")
        records = [FieldRecord(current_name, field_type, value, node, kids=kids)]

        for kid in kids:
            kid = self.resolver.resolve(kid)
            if not isinstance(kid, dict):
                raise ValueError("invalid AcroForm kid entry")
            subtype_value = kid.get("Subtype")
            subtype = (
                self.resolver.resolve_name(subtype_value)
                or self.resolver.resolve_str(subtype_value)
                or (str(subtype_value) if subtype_value is not None else "")
            )
            if subtype == "Widget":
                records.append(
                    FieldRecord(current_name, field_type, value, kid, kids=[], widget=kid)
                )
            else:
                records.extend(
                    cast(Any, self).collect_field_records(
                        kid, current_name, field_type, value, _depth + 1
                    )
                )
        return records

    def fields(self: DocumentMixinProtocol) -> list[FieldRecord]:
        if self.fields_cache is not None:
            return self.fields_cache
        af = cast(Any, self).acroform
        if af is None:
            self.fields_cache = []
            return self.fields_cache
        field_list = af.get("Fields")
        if field_list is None:
            field_list = []
        elif not isinstance(field_list, list):
            raise ValueError("invalid AcroForm Fields array")
        records: list[FieldRecord] = []
        for field in field_list:
            field_obj = self.resolver.resolve(field)
            records.extend(cast(Any, self).collect_field_records(field_obj))
        self.fields_cache = records
        return records
