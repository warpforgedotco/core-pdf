from __future__ import annotations

# Importing a submodule runs this file first, and layout analysis is not needed to read objects.
__lazy_modules__ = ["core_pdf_compat.pdfminer._extract", "core_pdf_compat.pdfminer._layout"]

from ._extract import (
    extract_pages,
    extract_text,
    extract_text_to_fp,
)
from ._layout import (
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
    LTTextBoxVertical,
    LTTextContainer,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
)
from ._layout import (
    LTComponent as LTComponent,
)

__version__ = "20260107"

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
