# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_name
from core_pdf.impl.models import FieldRecord
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.types import PdfArray, PdfDict, PdfObject


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
    field = FieldRecord(
        "name",
        "Tx",
        b"Jane Doe",
        "Jane Doe",
        (40.0, 700.0, 300.0, 720.0),
        cast(PdfDict, widget),
        widget=cast(PdfDict, widget),
    )
    page = PdfPage(cast(Any, FakeDocument([field])), {"Annots": [widget]}, 0)
    page.inherited_values_cache = {"Annots": [widget]}

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
    field = FieldRecord(
        "name",
        "Tx",
        b"Jane Doe",
        "Jane Doe",
        None,
        cast(PdfDict, parent),
        kids=cast(PdfArray, [widget]),
    )
    page = PdfPage(cast(Any, FakeDocument([field])), {"Annots": [widget]}, 0)
    page.inherited_values_cache = {"Annots": [widget]}

    assert page.get_fields() == [field]
