# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.impl.base.exceptions import (
        PdfContractError,
        PdfDecryptionError,
        PdfDocumentClosedError,
        PdfError,
        PdfParseError,
        PdfRasterTooLargeError,
        PdfSourceError,
        PdfUnsupportedError,
    )
    from core_pdf.impl.base.pages import PageSelection
    from core_pdf.impl.base.text import TextWord
    from core_pdf.impl.base.types import PageScoped
    from core_pdf.impl.capture.program import CaptureOptions
    from core_pdf.impl.document.editing import EditableDocument, NewEncryption, SavedPdf
    from core_pdf.impl.document.pdf_document import DocumentAdapter, PdfDocument
    from core_pdf.impl.document.pdf_page import PdfPage
    from core_pdf.impl.fonts.fallback import (
        PdfRasterFontFace,
        PdfRasterFontProvider,
        PdfRasterFontRequest,
    )
    from core_pdf.impl.output.media import DrawingRecord, ImageMetadata, ImageRecord
    from core_pdf.impl.output.model import (
        ContentNode,
        Document,
        DocumentTableView,
        DocumentTextView,
        TableAssociatedText,
        TableColumnBand,
        TableReference,
        TableRowBand,
        TableView,
        TextView,
    )
EXPORTS = {
    "CaptureOptions": ("core_pdf.impl.capture.program", "CaptureOptions"),
    "DocumentAdapter": ("core_pdf.impl.document.pdf_document", "DocumentAdapter"),
    "Document": ("core_pdf.impl.output.model", "Document"),
    "PageSelection": ("core_pdf.impl.base.pages", "PageSelection"),
    "EditableDocument": ("core_pdf.impl.document.editing", "EditableDocument"),
    "NewEncryption": ("core_pdf.impl.document.editing", "NewEncryption"),
    "SavedPdf": ("core_pdf.impl.document.editing", "SavedPdf"),
    "PdfDocument": ("core_pdf.impl.document.pdf_document", "PdfDocument"),
    "PdfRasterFontFace": ("core_pdf.impl.fonts.fallback", "PdfRasterFontFace"),
    "PdfRasterFontProvider": ("core_pdf.impl.fonts.fallback", "PdfRasterFontProvider"),
    "PdfRasterFontRequest": ("core_pdf.impl.fonts.fallback", "PdfRasterFontRequest"),
    "PdfError": ("core_pdf.impl.base.exceptions", "PdfError"),
    "PdfContractError": ("core_pdf.impl.base.exceptions", "PdfContractError"),
    "ContentNode": ("core_pdf.impl.output.model", "ContentNode"),
    "DocumentTableView": ("core_pdf.impl.output.model", "DocumentTableView"),
    "DocumentTextView": ("core_pdf.impl.output.model", "DocumentTextView"),
    "DrawingRecord": ("core_pdf.impl.output.media", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.output.media", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.output.media", "ImageRecord"),
    "PageScoped": ("core_pdf.impl.base.types", "PageScoped"),
    "TableView": ("core_pdf.impl.output.model", "TableView"),
    "TableReference": ("core_pdf.impl.output.model", "TableReference"),
    "TableAssociatedText": ("core_pdf.impl.output.model", "TableAssociatedText"),
    "TableColumnBand": ("core_pdf.impl.output.model", "TableColumnBand"),
    "TableRowBand": ("core_pdf.impl.output.model", "TableRowBand"),
    "TextView": ("core_pdf.impl.output.model", "TextView"),
    "TextWord": ("core_pdf.impl.base.text", "TextWord"),
    "PdfPage": ("core_pdf.impl.document.pdf_page", "PdfPage"),
    "PdfDecryptionError": ("core_pdf.impl.base.exceptions", "PdfDecryptionError"),
    "PdfDocumentClosedError": ("core_pdf.impl.base.exceptions", "PdfDocumentClosedError"),
    "PdfParseError": ("core_pdf.impl.base.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.base.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.base.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.base.exceptions", "PdfUnsupportedError"),
}


install_lazy_module_exports(globals(), EXPORTS)


__all__ = (
    "CaptureOptions",
    "DocumentAdapter",
    "Document",
    "PageSelection",
    "EditableDocument",
    "NewEncryption",
    "SavedPdf",
    "PdfDocument",
    "PdfError",
    "PdfContractError",
    "ContentNode",
    "DocumentTableView",
    "DocumentTextView",
    "DrawingRecord",
    "ImageMetadata",
    "ImageRecord",
    "PageScoped",
    "TableReference",
    "TableView",
    "TableAssociatedText",
    "TableColumnBand",
    "TableRowBand",
    "TextView",
    "TextWord",
    "PdfPage",
    "PdfDecryptionError",
    "PdfDocumentClosedError",
    "PdfParseError",
    "PdfRasterTooLargeError",
    "PdfSourceError",
    "PdfUnsupportedError",
    "PdfRasterFontFace",
    "PdfRasterFontProvider",
    "PdfRasterFontRequest",
)
