# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_pdf.impl.engine.extraction.document import PdfDocument
    from core_pdf.impl.engine.extraction.page import PdfPage
    from core_pdf.impl.engine.writing import (
        PdfFontProvider,
        StandardPdfEncryption,
        StandardType1FontProvider,
        TrueTypeFontProvider,
    )
    from core_pdf.impl.engine.writing.semantic import serialize_document_to_pdf
    from core_pdf.impl.exceptions import (
        PdfError,
        PdfParseError,
        PdfRasterTooLargeError,
        PdfSourceError,
        PdfUnsupportedError,
    )
    from core_pdf.impl.types import PageSelection

_EXPORTS = {
    "Document": ("core_document", "Document"),
    "PageSelection": ("core_pdf.impl.types", "PageSelection"),
    "PdfDocument": ("core_pdf.impl.engine.extraction.document", "PdfDocument"),
    "serialize_document_to_pdf": (
        "core_pdf.impl.engine.writing.semantic",
        "serialize_document_to_pdf",
    ),
    "PdfFontProvider": ("core_pdf.impl.engine.writing", "PdfFontProvider"),
    "StandardPdfEncryption": ("core_pdf.impl.engine.writing", "StandardPdfEncryption"),
    "StandardType1FontProvider": (
        "core_pdf.impl.engine.writing",
        "StandardType1FontProvider",
    ),
    "TrueTypeFontProvider": ("core_pdf.impl.engine.writing", "TrueTypeFontProvider"),
    "PdfError": ("core_pdf.impl.exceptions", "PdfError"),
    "PdfPage": ("core_pdf.impl.engine.extraction.page", "PdfPage"),
    "PdfParseError": ("core_pdf.impl.exceptions", "PdfParseError"),
    "PdfRasterTooLargeError": ("core_pdf.impl.exceptions", "PdfRasterTooLargeError"),
    "PdfSourceError": ("core_pdf.impl.exceptions", "PdfSourceError"),
    "PdfUnsupportedError": ("core_pdf.impl.exceptions", "PdfUnsupportedError"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = (
    "Document",
    "PageSelection",
    "PdfDocument",
    "PdfError",
    "PdfPage",
    "PdfParseError",
    "PdfRasterTooLargeError",
    "PdfSourceError",
    "PdfUnsupportedError",
    "serialize_document_to_pdf",
    "PdfFontProvider",
    "StandardPdfEncryption",
    "StandardType1FontProvider",
    "TrueTypeFontProvider",
)
