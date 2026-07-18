# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Self

from core_pdf.impl.engine.extraction.document_extraction import DocumentExtractionMixin
from core_pdf.impl.engine.extraction.document_structured import DocumentStructuredMixin
from core_pdf.impl.engine.extraction.document_text import DocumentTextMixin
from core_pdf.impl.engine.spec.s_07_document.document import (
    PdfDocument as SpecPdfDocument,
)
from core_pdf.impl.engine.writing import PdfDocumentWritingMixin

if TYPE_CHECKING:
    from core_document import Document, DocumentAdapter


def _create_page(document: Any, page_dict: Any, page_number: int) -> Any:
    from core_pdf.impl.engine.extraction.page import PdfPage

    return PdfPage(document, page_dict, page_number)


class PdfDocument(
    DocumentExtractionMixin,
    DocumentStructuredMixin,
    DocumentTextMixin,
    PdfDocumentWritingMixin,
    SpecPdfDocument,
):
    page_class = staticmethod(_create_page)

    def __enter__(self) -> Self:
        SpecPdfDocument.__enter__(self)
        return self

    @classmethod
    def open(cls, source: Any, password: str = "") -> Self:
        return cls(source, password=password)

    def extract(self, *, adapters: Iterable[DocumentAdapter] = ()) -> Document:
        from core_pdf.impl.engine.extraction.page_text.engine import build_document

        document = build_document(self)
        for adapter in adapters:
            document = adapter.apply(document)
        return document


__all__ = ("PdfDocument",)
