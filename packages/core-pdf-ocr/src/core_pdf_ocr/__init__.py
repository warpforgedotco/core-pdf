# SPDX-License-Identifier: AGPL-3.0-only
"""PDF extraction with optical and vector text recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf_ocr.api.document import PdfDocument, PdfPage

internal_EXPORTS = {
    "PdfDocument": ("core_pdf_ocr.api.document", "PdfDocument"),
    "PdfPage": ("core_pdf_ocr.api.document", "PdfPage"),
}

install_lazy_module_exports(globals(), internal_EXPORTS)

__all__ = ("PdfDocument", "PdfPage")
