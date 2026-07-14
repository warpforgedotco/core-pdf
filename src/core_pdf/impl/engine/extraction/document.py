# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Self

from core_pdf.impl.engine.extraction.document_extraction import DocumentExtractionMixin
from core_pdf.impl.engine.extraction.document_structured import DocumentStructuredMixin
from core_pdf.impl.engine.extraction.document_text import DocumentTextMixin
from core_pdf.impl.engine.extraction.page import PdfPage
from core_pdf.impl.engine.extraction.page_text.engine import (
    DocumentExtractionResult,
    build_document_extraction_result,
)
from core_pdf.impl.engine.spec.s_07_document.document import (
    PdfDocument as SpecPdfDocument,
)


class PdfDocument(
    DocumentExtractionMixin,
    DocumentStructuredMixin,
    DocumentTextMixin,
    SpecPdfDocument,
):
    page_class = PdfPage

    def __enter__(self) -> Self:
        SpecPdfDocument.__enter__(self)
        return self

    def extract(self) -> DocumentExtractionResult:
        return build_document_extraction_result(self)

    def extract_text(self) -> str:
        return "\f".join(page.extract_text() for page in self.pages) + "\f"


__all__ = ("PdfDocument",)
