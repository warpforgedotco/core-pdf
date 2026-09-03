# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
from typing import Any, cast

from core_pdf.impl.document import PdfDocument
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.spec.s_07_document.page import PdfPage
from core_pdf.impl.spec.s_07_document.records import RawFormField
from core_pdf.impl.spec.s_07_syntax.types import PdfArray, PdfDict, PdfObject
from tests.helpers.paths import SCORE_BENCH
from tests.helpers.resolvers import IdentityResolver

SAMPLE_PDF = SCORE_BENCH / "g-325a.pdf"


class FakeDocument:
    def __init__(self, fields: list[RawFormField]) -> None:
        self.internal_cache_lock = threading.RLock()
        self.recovery_enabled = False
        self.internal_page_locks: dict[int, threading.RLock] = {}
        self.resolver = IdentityResolver()
        self.internal_fields = fields
        self.pages: list[Any] = []
        self.fields_by_page_cache: dict[int, list[RawFormField]] | None = None

    def page_lock(self, page_number: int) -> threading.RLock:
        with self.internal_cache_lock:
            return self.internal_page_locks.setdefault(page_number, threading.RLock())

    def fields(self) -> list[RawFormField]:
        return self.internal_fields

    def resolve(self, value: PdfObject) -> PdfObject:
        return cast(PdfObject, self.resolver.resolve(value))

    def page_index_for(self, page_obj: object) -> int | None:
        return 0 if isinstance(page_obj, dict) else None

    def fields_by_page(self) -> dict[int, list[RawFormField]]:
        if self.fields_by_page_cache is None:
            self.fields_by_page_cache = PdfDocument.build_fields_by_page(cast(Any, self))
        return self.fields_by_page_cache


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
    unrelated_widget: PdfDict = {"Subtype": PdfName.of("Widget")}
    unrelated = RawFormField(
        "other",
        "Tx",
        b"Other",
        "Other",
        None,
        unrelated_widget,
        widget=unrelated_widget,
    )
    document = FakeDocument([field, unrelated])
    page = PdfPage(cast(Any, document), {"Annots": [widget]}, 1)
    document.pages = [page]

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
    unrelated_widget: PdfDict = {"Subtype": PdfName.of("Widget")}
    unrelated = RawFormField(
        "other",
        "Tx",
        b"Other",
        "Other",
        None,
        {"Kids": [unrelated_widget]},
        kids=cast(PdfArray, [unrelated_widget]),
    )
    document = FakeDocument([field, unrelated])
    page = PdfPage(cast(Any, document), {"Annots": [widget]}, 1)
    document.pages = [page]

    assert page.get_fields() == [field]


def test_page_program_contains_capture_products_and_immutable_events() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = document.pages[0]
        program = page.get_page_program()

        assert program.runs
        events = program.events
        assert isinstance(events, tuple)
        assert any(event.payload is program.runs[0] for event in events)


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
