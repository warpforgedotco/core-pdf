# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

import core_pdf
from core_pdf.impl.exceptions import ExtractionScope
from core_pdf.impl.output_model import Document, Page
from core_pdf_ocr.impl.extract.ocr.tesseract import prepare_ocr_signals
from core_pdf_ocr.impl.extract.pipeline import extract_page
from core_pdf_ocr.impl.extract.selection import extract_document

prepare_ocr_signals()


class PdfPage(core_pdf.PdfPage):
    def run_extract_page(self, context: ExtractionScope) -> Page:
        return extract_page(self, context)


class PdfDocument(core_pdf.PdfDocument):
    page_class = PdfPage

    def run_extract_document(
        self, context: ExtractionScope, pages: Sequence[core_pdf.PdfPage]
    ) -> Document:
        return extract_document(self, context, pages)
