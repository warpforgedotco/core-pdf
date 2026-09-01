# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.impl.document import PdfDocument
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
    from core_pdf.impl.models import (
        DrawingRecord,
        ImageMetadata,
        ImageRecord,
        PageScoped,
        TextWord,
    )
    from core_pdf.impl.page import PdfPage
    from core_pdf.impl.pages import PageSelection
    from core_pdf.impl.parse.ocr_tesseract import prewarm_runtime
    from core_pdf.impl.runtime.execution import (
        RuntimeConfig,
        RuntimeMetrics,
        configure_runtime,
        runtime_metrics,
        shutdown_runtime,
    )
    from core_pdf.impl.spec.s_09_fonts.fallback import (
        PdfRasterFontFace,
        PdfRasterFontProvider,
        PdfRasterFontRequest,
    )
    from core_pdf.impl.structured.model import (
        ContentNode,
        DiagnosticTextRun,
        DocumentTableView,
        DocumentTextView,
        TableView,
        TextDiagnostics,
        TextView,
    )
internal_EXPORTS = {
    "Document": ("core_pdf.impl.structured.model", "Document"),
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
    "ContentNode": ("core_pdf.impl.structured.model", "ContentNode"),
    "DiagnosticTextRun": ("core_pdf.impl.structured.model", "DiagnosticTextRun"),
    "DocumentTableView": ("core_pdf.impl.structured.model", "DocumentTableView"),
    "DocumentTextView": ("core_pdf.impl.structured.model", "DocumentTextView"),
    "DrawingRecord": ("core_pdf.impl.models", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.models", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.models", "ImageRecord"),
    "PageScoped": ("core_pdf.impl.models", "PageScoped"),
    "TableView": ("core_pdf.impl.structured.model", "TableView"),
    "TableReference": ("core_pdf.impl.structured.model", "TableReference"),
    "TableAssociatedText": ("core_pdf.impl.structured.model", "TableAssociatedText"),
    "TableColumnBand": ("core_pdf.impl.structured.model", "TableColumnBand"),
    "TableRowBand": ("core_pdf.impl.structured.model", "TableRowBand"),
    "TextView": ("core_pdf.impl.structured.model", "TextView"),
    "TextWord": ("core_pdf.impl.models", "TextWord"),
    "TextDiagnostics": ("core_pdf.impl.structured.model", "TextDiagnostics"),
    "TextLineReference": ("core_pdf.impl.structured.model", "TextLineReference"),
    "PdfPage": ("core_pdf.impl.page", "PdfPage"),
    "PdfDecryptionError": ("core_pdf.impl.exceptions", "PdfDecryptionError"),
    "PdfDocumentClosedError": ("core_pdf.impl.exceptions", "PdfDocumentClosedError"),
    "PdfParseError": ("core_pdf.impl.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.exceptions", "PdfUnsupportedError"),
    "configure_runtime": (
        "core_pdf.impl.runtime.execution",
        "configure_runtime",
    ),
    "RuntimeMetrics": (
        "core_pdf.impl.runtime.execution",
        "RuntimeMetrics",
    ),
    "RuntimeConfig": (
        "core_pdf.impl.runtime.execution",
        "RuntimeConfig",
    ),
    "prewarm_runtime": (
        "core_pdf.impl.parse.ocr_tesseract",
        "prewarm_runtime",
    ),
    "runtime_metrics": (
        "core_pdf.impl.runtime.execution",
        "runtime_metrics",
    ),
    "shutdown_runtime": (
        "core_pdf.impl.runtime.execution",
        "shutdown_runtime",
    ),
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
    "configure_runtime",
    "RuntimeConfig",
    "RuntimeMetrics",
    "prewarm_runtime",
    "runtime_metrics",
    "shutdown_runtime",
)
