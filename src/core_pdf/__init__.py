# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.impl.document.document import DocumentAdapter, PdfDocument
    from core_pdf.impl.document.page import PdfPage
    from core_pdf.impl.exceptions import (
        PdfContractError,
        PdfDecryptionError,
        PdfDocumentClosedError,
        PdfError,
        PdfParseError,
        PdfRasterTooLargeError,
        PdfSourceError,
        PdfUnsupportedError,
    )
    from core_pdf.impl.fonts.fallback import (
        PdfRasterFontFace,
        PdfRasterFontProvider,
        PdfRasterFontRequest,
    )
    from core_pdf.impl.model.page_selection import PageSelection
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
        TextLineReference,
        TextView,
    )
    from core_pdf.impl.types import (
        DrawingRecord,
        ImageMetadata,
        ImageRecord,
        PageScoped,
        TextWord,
    )
EXPORTS = {
    "DocumentAdapter": ("core_pdf.impl.document.document", "DocumentAdapter"),
    "Document": ("core_pdf.impl.output.model", "Document"),
    "PageSelection": ("core_pdf.impl.model.page_selection", "PageSelection"),
    "PdfDocument": ("core_pdf.impl.document.document", "PdfDocument"),
    "PdfRasterFontFace": (
        "core_pdf.impl.fonts.fallback",
        "PdfRasterFontFace",
    ),
    "PdfRasterFontProvider": (
        "core_pdf.impl.fonts.fallback",
        "PdfRasterFontProvider",
    ),
    "PdfRasterFontRequest": (
        "core_pdf.impl.fonts.fallback",
        "PdfRasterFontRequest",
    ),
    "PdfError": ("core_pdf.impl.exceptions", "PdfError"),
    "PdfContractError": ("core_pdf.impl.exceptions", "PdfContractError"),
    "ContentNode": ("core_pdf.impl.output.model", "ContentNode"),
    "DocumentTableView": ("core_pdf.impl.output.model", "DocumentTableView"),
    "DocumentTextView": ("core_pdf.impl.output.model", "DocumentTextView"),
    "DrawingRecord": ("core_pdf.impl.types", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.types", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.types", "ImageRecord"),
    "PageScoped": ("core_pdf.impl.types", "PageScoped"),
    "TableView": ("core_pdf.impl.output.model", "TableView"),
    "TableReference": ("core_pdf.impl.output.model", "TableReference"),
    "TableAssociatedText": ("core_pdf.impl.output.model", "TableAssociatedText"),
    "TableColumnBand": ("core_pdf.impl.output.model", "TableColumnBand"),
    "TableRowBand": ("core_pdf.impl.output.model", "TableRowBand"),
    "TextView": ("core_pdf.impl.output.model", "TextView"),
    "TextWord": ("core_pdf.impl.types", "TextWord"),
    "TextLineReference": ("core_pdf.impl.output.model", "TextLineReference"),
    "PdfPage": ("core_pdf.impl.document.page", "PdfPage"),
    "PdfDecryptionError": ("core_pdf.impl.exceptions", "PdfDecryptionError"),
    "PdfDocumentClosedError": ("core_pdf.impl.exceptions", "PdfDocumentClosedError"),
    "PdfParseError": ("core_pdf.impl.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.exceptions", "PdfUnsupportedError"),
}


install_lazy_module_exports(globals(), EXPORTS)


__all__ = (
    "DocumentAdapter",
    "Document",
    "PageSelection",
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
    "TextLineReference",
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
