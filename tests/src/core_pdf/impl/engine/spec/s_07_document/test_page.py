# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from core_pdf.impl.engine.document import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_name
from core_pdf.impl.models import RawFormField
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.types import PdfArray, PdfDict, PdfObject

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "g-325a.pdf"


class FakeResolver:
    def resolve(self, value: PdfObject) -> PdfObject:
        return value

    def resolve_name(self, value: PdfObject) -> str | None:
        return parse_name(value)


class FakeDocument:
    def __init__(self, fields: list[RawFormField]) -> None:
        self.resolver = FakeResolver()
        self.internal_fields = fields

    def fields(self) -> list[RawFormField]:
        return self.internal_fields

    def resolve(self, value: PdfObject) -> PdfObject:
        return self.resolver.resolve(value)

    def page_index_for(self, page_obj: object) -> int | None:
        return 0 if isinstance(page_obj, dict) else None


def test_page_get_fields_matches_direct_widget_annotation_without_page_ref() -> None:
    widget = {
        "Subtype": PdfName.of("Widget"),
        "FT": PdfName.of("Tx"),
        "T": b"name",
        "V": b"Jane Doe",
        "Rect": [40, 700, 300, 720],
    }
    field = RawFormField(
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
    field = RawFormField(
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


def test_page_builds_one_canonical_program_for_all_consumers() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = document.pages[0]
        program = page.get_page_program()

        assert page.get_page_program() is program
        assert program.products.runs
        assert program.events.sequence.flags.writeable is False


def test_page_program_is_shared_by_extraction_and_rendering(monkeypatch) -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = document.pages[0]
        calls = 0
        original = page.consume_contents

        def counted(state) -> None:
            nonlocal calls
            calls += 1
            original(state)

        monkeypatch.setattr(page, "consume_contents", counted)
        program = page.get_page_program()
        page.extract()
        page.render()
        assert page.get_page_program() is program
        assert calls == 1
