# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.impl.document import PdfDocument, PdfPage
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
    from core_pdf.impl.output import (
        ContentNode,
        DiagnosticTextRun,
        Document,
        DocumentTableView,
        DocumentTextView,
        TableAssociatedText,
        TableColumnBand,
        TableReference,
        TableRowBand,
        TableView,
        TextDiagnostics,
        TextLineReference,
        TextView,
    )
    from core_pdf.impl.pages import PageSelection
    from core_pdf.impl.records import (
        DrawingRecord,
        ImageMetadata,
        ImageRecord,
        PageScoped,
        TextWord,
    )
    from core_pdf.impl.spec.s_09_fonts.fallback import (
        PdfRasterFontFace,
        PdfRasterFontProvider,
        PdfRasterFontRequest,
    )
internal_EXPORTS = {
    "Document": ("core_pdf.impl.output", "Document"),
    "PageSelection": ("core_pdf.impl.pages", "PageSelection"),
    "PdfDocument": ("core_pdf.impl.document", "PdfDocument"),
    "PdfRasterFontFace": (
        "core_pdf.impl.spec.s_09_fonts.fallback",
        "PdfRasterFontFace",
    ),
    "PdfRasterFontProvider": (
        "core_pdf.impl.spec.s_09_fonts.fallback",
        "PdfRasterFontProvider",
    ),
    "PdfRasterFontRequest": (
        "core_pdf.impl.spec.s_09_fonts.fallback",
        "PdfRasterFontRequest",
    ),
    "PdfError": ("core_pdf.impl.exceptions", "PdfError"),
    "PdfContractError": ("core_pdf.impl.exceptions", "PdfContractError"),
    "ContentNode": ("core_pdf.impl.output", "ContentNode"),
    "DiagnosticTextRun": ("core_pdf.impl.output", "DiagnosticTextRun"),
    "DocumentTableView": ("core_pdf.impl.output", "DocumentTableView"),
    "DocumentTextView": ("core_pdf.impl.output", "DocumentTextView"),
    "DrawingRecord": ("core_pdf.impl.records", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.records", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.records", "ImageRecord"),
    "PageScoped": ("core_pdf.impl.records", "PageScoped"),
    "TableView": ("core_pdf.impl.output", "TableView"),
    "TableReference": ("core_pdf.impl.output", "TableReference"),
    "TableAssociatedText": ("core_pdf.impl.output", "TableAssociatedText"),
    "TableColumnBand": ("core_pdf.impl.output", "TableColumnBand"),
    "TableRowBand": ("core_pdf.impl.output", "TableRowBand"),
    "TextView": ("core_pdf.impl.output", "TextView"),
    "TextWord": ("core_pdf.impl.records", "TextWord"),
    "TextDiagnostics": ("core_pdf.impl.output", "TextDiagnostics"),
    "TextLineReference": ("core_pdf.impl.output", "TextLineReference"),
    "PdfPage": ("core_pdf.impl.document", "PdfPage"),
    "PdfDecryptionError": ("core_pdf.impl.exceptions", "PdfDecryptionError"),
    "PdfDocumentClosedError": ("core_pdf.impl.exceptions", "PdfDocumentClosedError"),
    "PdfParseError": ("core_pdf.impl.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.exceptions", "PdfUnsupportedError"),
}


install_lazy_module_exports(globals(), internal_EXPORTS)


__all__ = (
    "Document",
    "PageSelection",
    "PdfDocument",
    "PdfError",
    "PdfContractError",
    "ContentNode",
    "DiagnosticTextRun",
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
    "TextDiagnostics",
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
