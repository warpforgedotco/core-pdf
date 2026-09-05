# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.api.document import PdfDocument, PdfPage
    from core_pdf.impl._impl.model.page_selection import PageSelection
    from core_pdf.impl._impl.output.model import (
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
    "Document": ("core_pdf.impl._impl.output.model", "Document"),
    "PageSelection": ("core_pdf.impl._impl.model.page_selection", "PageSelection"),
    "PdfDocument": ("core_pdf.api.document", "PdfDocument"),
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
    "ContentNode": ("core_pdf.impl._impl.output.model", "ContentNode"),
    "DiagnosticTextRun": ("core_pdf.impl._impl.output.model", "DiagnosticTextRun"),
    "DocumentTableView": ("core_pdf.impl._impl.output.model", "DocumentTableView"),
    "DocumentTextView": ("core_pdf.impl._impl.output.model", "DocumentTextView"),
    "DrawingRecord": ("core_pdf.impl.records", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.records", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.records", "ImageRecord"),
    "PageScoped": ("core_pdf.impl.records", "PageScoped"),
    "TableView": ("core_pdf.impl._impl.output.model", "TableView"),
    "TableReference": ("core_pdf.impl._impl.output.model", "TableReference"),
    "TableAssociatedText": ("core_pdf.impl._impl.output.model", "TableAssociatedText"),
    "TableColumnBand": ("core_pdf.impl._impl.output.model", "TableColumnBand"),
    "TableRowBand": ("core_pdf.impl._impl.output.model", "TableRowBand"),
    "TextView": ("core_pdf.impl._impl.output.model", "TextView"),
    "TextWord": ("core_pdf.impl.records", "TextWord"),
    "TextDiagnostics": ("core_pdf.impl._impl.output.model", "TextDiagnostics"),
    "TextLineReference": ("core_pdf.impl._impl.output.model", "TextLineReference"),
    "PdfPage": ("core_pdf.api.document", "PdfPage"),
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
