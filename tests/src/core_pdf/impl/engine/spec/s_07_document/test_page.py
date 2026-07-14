# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.models import FieldRecord
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_syntax.primitives import PdfName, PdfObject, parse_name


class FakeResolver:
    def resolve(self, value: PdfObject) -> PdfObject:
        return value

    def resolve_name(self, value: PdfObject) -> str | None:
        return parse_name(value)


class FakeDocument:
    def __init__(self, fields: list[FieldRecord]) -> None:
        self.resolver = FakeResolver()
        self._fields = fields

    def fields(self) -> list[FieldRecord]:
        return self._fields


def test_page_get_fields_matches_direct_widget_annotation_without_page_ref() -> None:
    widget = {
        "Subtype": PdfName.of("Widget"),
        "FT": PdfName.of("Tx"),
        "T": b"name",
        "V": b"Jane Doe",
        "Rect": [40, 700, 300, 720],
    }
    field = FieldRecord("name", "Tx", b"Jane Doe", widget)
    page = PdfPage(FakeDocument([field]), {"Annots": [widget]}, 0)  # type: ignore[arg-type]
    page._inherited_values = {"Annots": [widget]}

    assert page.get_fields() == [field]


def test_page_get_fields_matches_kid_widget_annotation_without_page_ref() -> None:
    widget = {
        "Subtype": PdfName.of("Widget"),
        "Rect": [40, 700, 300, 720],
    }
    parent = {
        "FT": PdfName.of("Tx"),
        "T": b"name",
        "V": b"Jane Doe",
        "Kids": [widget],
    }
    field = FieldRecord("name", "Tx", b"Jane Doe", parent, kids=[widget])
    page = PdfPage(FakeDocument([field]), {"Annots": [widget]}, 0)  # type: ignore[arg-type]
    page._inherited_values = {"Annots": [widget]}

    assert page.get_fields() == [field]
