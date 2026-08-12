"""High-level compatibility facades over the local core-pdf engine."""

from . import (
    llamaindex,
    pdfminer,
    pdfplumber,
    pikepdf,
    pymupdf,
    pypdf,
    unstructured,
    xray,
)
from .llamaindex import get_nodes_from_documents, load_data
from .pdfminer import (
    LAParams,
    LTAnno,
    LTChar,
    LTImage,
    LTItem,
    LTPage,
    LTText,
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextBoxVertical,
    LTTextLine,
    LTTextLineHorizontal,
    LTTextLineVertical,
    extract_pages,
    extract_text,
    extract_text_to_fp,
)
from .pdfplumber import (
    PDF,
    CroppedPage,
    FilteredPage,
    Page,
    PageImage,
    Table,
    TableFinder,
    TableSettings,
    extract_words,
    open,
    outside_bbox,
    utils,
    within_bbox,
)
from .pdfplumber import (
    extract_text as extract_pdfplumber_text,
)
from .pdfplumber import open as open_pdf
from .pikepdf import Pdf
from .pymupdf import Document as FitzDocument
from .pymupdf import open as open_fitz
from .pypdf import PdfMerger, PdfPageObject, PdfReader, PdfWriter
from .unstructured import partition_pdf
from .xray import inspect as inspect_xray

__all__ = (
    "llamaindex",
    "pdfminer",
    "pdfplumber",
    "pikepdf",
    "pymupdf",
    "pypdf",
    "unstructured",
    "xray",
    "LAParams",
    "LTAnno",
    "LTChar",
    "LTImage",
    "LTItem",
    "LTPage",
    "LTText",
    "LTTextBox",
    "LTTextBoxHorizontal",
    "LTTextBoxVertical",
    "LTTextLine",
    "LTTextLineHorizontal",
    "LTTextLineVertical",
    "extract_pages",
    "extract_text",
    "extract_text_to_fp",
    "inspect_xray",
    "CroppedPage",
    "FilteredPage",
    "PDF",
    "Page",
    "PageImage",
    "Table",
    "TableFinder",
    "TableSettings",
    "extract_pdfplumber_text",
    "extract_words",
    "open",
    "outside_bbox",
    "utils",
    "within_bbox",
    "open_pdf",
    "FitzDocument",
    "open_fitz",
    "Pdf",
    "PdfMerger",
    "PdfPageObject",
    "PdfReader",
    "PdfWriter",
    "get_nodes_from_documents",
    "load_data",
    "partition_pdf",
)
