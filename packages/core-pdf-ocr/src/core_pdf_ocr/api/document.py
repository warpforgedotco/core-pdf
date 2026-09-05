# SPDX-License-Identifier: AGPL-3.0-only
"""OCR document and page APIs sharing core-pdf's operation lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from core_pdf.api.document import PdfDocument as CorePdfDocument
from core_pdf.api.document import PdfPage as CorePdfPage
from core_pdf.impl._impl.model.page_selection import PageSelection
from core_pdf.impl._impl.runtime.execution import ExtractionScope
from core_pdf.impl.types import PdfSource
from core_pdf_ocr.impl.extract.ocr.tesseract import internal_prepare_ocr_signals
from core_pdf_ocr.impl.extract.pipeline import extract_page
from core_pdf_ocr.impl.extract.selection import extract_document

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_09_fonts.fallback import RasterFontProviderLike

# Recognition owns process signal preparation; importing core-pdf is side-effect free.
internal_prepare_ocr_signals()


class PdfPage(CorePdfPage):
    """A PDF page whose structured extraction can recover recognized text."""

    def extract(self) -> Any:
        with self.document.acquire_operation() as operation:
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            return extract_page(self, context)


class PdfDocument(CorePdfDocument):
    """A PDF document with native, hybrid, and recognized extraction routes."""

    def __init__(
        self,
        source: PdfSource,
        password: str = "",
        *,
        recovery_scan_all_revisions: bool = True,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        super().__init__(
            source,
            password=password,
            recovery_scan_all_revisions=recovery_scan_all_revisions,
            raster_font_provider=raster_font_provider,
        )
        self.page_class = PdfPage

    def extract(
        self,
        *,
        pages: PageSelection | None = None,
        adapters: Iterable[Any] = (),
    ) -> Any:
        with self.acquire_operation() as operation:
            selected_pages = tuple(page for _index, page in self.iter_selected_pages(pages))
            context = ExtractionScope(cancelled=lambda: operation.cancelled)
            result = extract_document(self, context, selected_pages)
        for adapter in adapters:
            result = adapter.apply(result)
        return result
