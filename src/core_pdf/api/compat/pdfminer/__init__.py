"""PDFMiner-compatible APIs backed by core-pdf."""

from __future__ import annotations

from typing import Any, TypeAlias

from ._extract import (
    extract_pages,
    extract_text,
    extract_text_to_fp,
)
from ._layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTComponent,
    LTFigure,
    LTImage,
    LTItem,
    LTPage,
    LTText,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextContainer,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)

PdfInput: TypeAlias = Any

# Preserve the historical public identity for imports, reprs, and pickling.
for internal_export in (
    LAParams,
    LTAnno,
    LTChar,
    LTFigure,
    LTImage,
    LTItem,
    LTPage,
    LTText,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
    LTTextBoxVertical,
    LTTextContainer,
    extract_pages,
    extract_text,
    extract_text_to_fp,
    LTComponent,
):
    internal_export.__module__ = __name__


__all__ = (
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTFigure",
    "LTImage",
    "LTItem",
    "LTPage",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
    "LTTextBoxVertical",
    "LTTextContainer",
    "extract_pages",
    "extract_text",
    "extract_text_to_fp",
)
