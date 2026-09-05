# SPDX-License-Identifier: AGPL-3.0-only
"""Extraction operations release resources before handing results to adapters."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import assert_type

import pytest

from core_pdf import DocumentAdapter, PdfDocument
from core_pdf.api import document as api
from core_pdf.impl._impl.output.model import Document, Page
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.spec.s_07_document.page import PdfPage as SpecPdfPage
from tests.helpers.pdf_bytes import text_pages_pdf


def test_adapters_run_in_order_after_the_extraction_operation_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    document = PdfDocument(text_pages_pdf(("one", "two", "three")))

    def extract(
        source: PdfDocument, context: ExtractionScope, pages: Sequence[SpecPdfPage]
    ) -> Document:
        assert source is document
        assert source.internal_active_operations == 1
        assert [page.page_number for page in pages] == [3, 1]
        context.raise_if_cancelled()
        events.append("extract")
        return Document(pages=tuple(Page(page_number=page.page_number) for page in pages))

    class First:
        def apply(self, result: Document) -> Document:
            assert document.internal_active_operations == 0
            events.append("first")
            document.close()
            assert document.internal_closed
            assert document.raw_data == b""
            return replace(result, metadata={"first": True})

    class Second:
        def apply(self, result: Document) -> Document:
            assert document.internal_active_operations == 0
            assert result.metadata == {"first": True}
            events.append("second")
            return replace(result, metadata={**result.metadata, "second": True})

    def adapters() -> Iterator[DocumentAdapter]:
        assert document.internal_active_operations == 0
        events.append("adapters")
        yield First()
        yield Second()

    monkeypatch.setattr(api, "extract_document", extract)
    try:
        result = document.extract(pages=[3, 1, 3], adapters=adapters())
        assert_type(result, Document)
        assert tuple(page.page_number for page in result.pages) == (3, 1)
        assert result.metadata == {"first": True, "second": True}
        assert events == ["extract", "adapters", "first", "second"]
    finally:
        document.close()


@pytest.mark.parametrize("page_only", [False, True])
@pytest.mark.parametrize("close", [False, True])
def test_extraction_errors_release_the_operation(
    monkeypatch: pytest.MonkeyPatch, page_only: bool, close: bool
) -> None:
    document = PdfDocument(text_pages_pdf(("body",)))
    page = document.pages[0]

    def fail(*args: object) -> None:
        assert document.internal_active_operations == 1
        if close:
            document.close()
            assert not document.internal_closed
            assert document.raw_data
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(api, "extract_page" if page_only else "extract_document", fail)
    try:
        with pytest.raises(RuntimeError, match="extraction failed"):
            if page_only:
                page.extract()
            else:
                document.extract()
        assert document.internal_active_operations == 0
        assert document.closed is close
        assert document.internal_closed is close
        assert bool(document.raw_data) is not close
    finally:
        document.close()


def test_invalid_selection_releases_the_operation_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object) -> None:
        raise AssertionError("invalid selection reached extraction")

    monkeypatch.setattr(api, "extract_document", unexpected)
    with PdfDocument(text_pages_pdf(("body",))) as document:
        with pytest.raises(IndexError, match="out of range"):
            document.extract(pages=[0])
        assert document.internal_active_operations == 0
        assert not document.closed


@pytest.mark.parametrize("fail_in_iterator", [False, True])
def test_adapter_errors_propagate_after_release_and_stop_later_adapters(
    fail_in_iterator: bool,
) -> None:
    events: list[str] = []
    with PdfDocument(text_pages_pdf(("body",))) as document:

        class FailingAdapter:
            def apply(self, result: Document) -> Document:
                assert document.internal_active_operations == 0
                events.append("apply")
                raise RuntimeError("adapter failed")

        def adapters() -> Iterator[DocumentAdapter]:
            assert document.internal_active_operations == 0
            if fail_in_iterator:
                raise RuntimeError("adapter failed")
            yield FailingAdapter()
            events.append("unreachable")

        with pytest.raises(RuntimeError, match="adapter failed"):
            document.extract(adapters=adapters())
        assert document.internal_active_operations == 0
        assert not document.closed
        assert events == ([] if fail_in_iterator else ["apply"])


def test_structured_views_and_adapter_protocol_resolve_from_the_public_api() -> None:
    assert DocumentAdapter is api.DocumentAdapter
    with PdfDocument(text_pages_pdf(("body",))) as document:
        page = document.pages[0]
        extracted_page = assert_type(page.extract(), Page)
        page_view = assert_type(page.structured_view, Page)
        extracted_document = assert_type(document.extract(), Document)
        document_view = assert_type(document.structured_document, Document)
        assert extracted_page.text == page_view.text == "body"
        assert extracted_document.pages[0].text == document_view.pages[0].text == "body"
