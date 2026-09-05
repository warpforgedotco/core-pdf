# SPDX-License-Identifier: AGPL-3.0-only
"""OCR document and page APIs sharing core-pdf's operation lifecycle."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.api.document import PdfDocument as CorePdfDocument
from core_pdf.api.document import PdfPage as CorePdfPage
from core_pdf.impl._impl.output.model import Document, Page
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.spec.s_07_document.page import PdfPage as SpecPdfPage
from core_pdf_ocr.impl.extract.ocr.tesseract import internal_prepare_ocr_signals
from core_pdf_ocr.impl.extract.pipeline import extract_page
from core_pdf_ocr.impl.extract.selection import extract_document

# Recognition owns process signal preparation; importing core-pdf is side-effect free.
internal_prepare_ocr_signals()


class PdfPage(CorePdfPage):
    """A PDF page whose structured extraction can recover recognized text."""

    def internal_extract_page(self, context: ExtractionScope) -> Page:
        return extract_page(self, context)


class PdfDocument(CorePdfDocument):
    """A PDF document with native, hybrid, and recognized extraction routes."""

    page_class = PdfPage

    def internal_extract_document(
        self, context: ExtractionScope, pages: Sequence[SpecPdfPage]
    ) -> Document:
        return extract_document(self, context, pages)
