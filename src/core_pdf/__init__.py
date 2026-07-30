# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl import install_lazy_module_exports

if TYPE_CHECKING:
    from core_pdf.impl.engine.document import PdfDocument
    from core_pdf.impl.engine.execution import (
        RuntimeConfig,
        RuntimeMetrics,
        configure_runtime,
        runtime_metrics,
        shutdown_runtime,
    )
    from core_pdf.impl.engine.page import PdfPage
    from core_pdf.impl.engine.parse import prewarm_runtime
    from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
    from core_pdf.impl.engine.writing.fonts import (
        PdfFontProvider,
        StandardType1FontProvider,
        TrueTypeFontProvider,
    )
    from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
    from core_pdf.impl.engine.writing.signatures import (
        PdfSignaturePlan,
        PdfSignatureProvider,
    )
    from core_pdf.impl.exceptions import (
        PdfContractError,
        PdfError,
        PdfParseError,
        PdfRasterTooLargeError,
        PdfSourceError,
        PdfUnsupportedError,
    )
    from core_pdf.impl.models import (
        AnnotationContentRecord,
        DrawingRecord,
        ImageMetadata,
        ImageRecord,
        LineRecord,
        PageScoped,
        TableRecord,
        TextRunRecord,
        WordRecord,
    )
    from core_pdf.impl.types import PageSelection

internal_EXPORTS = {
    "Document": ("core_pdf.impl.engine.structured", "Document"),
    "PageSelection": ("core_pdf.impl.types", "PageSelection"),
    "PdfDocument": ("core_pdf.impl.engine.document", "PdfDocument"),
    "serialize_document_to_pdf": (
        "core_pdf.impl.engine.writing.semantic",
        "serialize_document_to_pdf",
    ),
    "PdfFontProvider": ("core_pdf.impl.engine.writing.fonts", "PdfFontProvider"),
    "PdfSignaturePlan": (
        "core_pdf.impl.engine.writing.signatures",
        "PdfSignaturePlan",
    ),
    "PdfSignatureProvider": (
        "core_pdf.impl.engine.writing.signatures",
        "PdfSignatureProvider",
    ),
    "StandardPdfEncryption": (
        "core_pdf.impl.engine.writing.encryption",
        "StandardPdfEncryption",
    ),
    "StandardType1FontProvider": (
        "core_pdf.impl.engine.writing.fonts",
        "StandardType1FontProvider",
    ),
    "TrueTypeFontProvider": (
        "core_pdf.impl.engine.writing.fonts",
        "TrueTypeFontProvider",
    ),
    "PdfError": ("core_pdf.impl.exceptions", "PdfError"),
    "PdfContractError": ("core_pdf.impl.exceptions", "PdfContractError"),
    "AnnotationContentRecord": ("core_pdf.impl.models", "AnnotationContentRecord"),
    "DrawingRecord": ("core_pdf.impl.models", "DrawingRecord"),
    "ImageMetadata": ("core_pdf.impl.models", "ImageMetadata"),
    "ImageRecord": ("core_pdf.impl.models", "ImageRecord"),
    "LineRecord": ("core_pdf.impl.models", "LineRecord"),
    "PageScoped": ("core_pdf.impl.models", "PageScoped"),
    "TableRecord": ("core_pdf.impl.models", "TableRecord"),
    "TextRunRecord": ("core_pdf.impl.models", "TextRunRecord"),
    "WordRecord": ("core_pdf.impl.models", "WordRecord"),
    "PdfPage": ("core_pdf.impl.engine.page", "PdfPage"),
    "PdfParseError": ("core_pdf.impl.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.exceptions", "PdfUnsupportedError"),
    "configure_runtime": (
        "core_pdf.impl.engine.execution",
        "configure_runtime",
    ),
    "RuntimeMetrics": (
        "core_pdf.impl.engine.execution",
        "RuntimeMetrics",
    ),
    "RuntimeConfig": (
        "core_pdf.impl.engine.execution",
        "RuntimeConfig",
    ),
    "SharedMemoryPdfBuffer": (
        "core_pdf.impl.engine.execution",
        "SharedMemoryPdfBuffer",
    ),
    "prewarm_runtime": (
        "core_pdf.impl.engine.parse",
        "prewarm_runtime",
    ),
    "runtime_metrics": (
        "core_pdf.impl.engine.execution",
        "runtime_metrics",
    ),
    "shutdown_runtime": (
        "core_pdf.impl.engine.execution",
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
    "AnnotationContentRecord",
    "DrawingRecord",
    "ImageMetadata",
    "ImageRecord",
    "LineRecord",
    "PageScoped",
    "TableRecord",
    "TextRunRecord",
    "WordRecord",
    "PdfPage",
    "PdfParseError",
    "PdfRasterTooLargeError",
    "PdfSourceError",
    "PdfUnsupportedError",
    "serialize_document_to_pdf",
    "PdfFontProvider",
    "PdfSignaturePlan",
    "PdfSignatureProvider",
    "StandardPdfEncryption",
    "StandardType1FontProvider",
    "TrueTypeFontProvider",
    "configure_runtime",
    "RuntimeConfig",
    "RuntimeMetrics",
    "SharedMemoryPdfBuffer",
    "prewarm_runtime",
    "runtime_metrics",
    "shutdown_runtime",
)
