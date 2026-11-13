from __future__ import annotations

# Importing a submodule runs this file first, and layout analysis is not needed to read objects.
lazy from ._extract import extract_pages, extract_text, extract_text_to_fp
lazy from ._layout import (
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
lazy from ._layout import LTComponent as LTComponent

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
