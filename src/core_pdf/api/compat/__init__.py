"""High-level compatibility facades over the local core-pdf engine.

Facades are loaded independently so importing one compatibility target does not
initialize every other third-party projection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

internal_MODULES = frozenset(
    {"llamaindex", "pdfminer", "pdfplumber", "pikepdf", "pypdf", "unstructured", "xray"}
)
internal_EXPORTS = {
    **{
        name: ("pdfminer", name)
        for name in (
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
        )
    },
    "Pdf": ("pikepdf", "Pdf"),
    "PdfReader": ("pypdf", "PdfReader"),
    "PdfWriter": ("pypdf", "PdfWriter"),
    "get_nodes_from_documents": ("llamaindex", "get_nodes_from_documents"),
    "load_data": ("llamaindex", "load_data"),
    "partition_pdf": ("unstructured", "partition_pdf"),
    "inspect_xray": ("xray", "inspect"),
}


def __getattr__(name: str) -> Any:
    if name in internal_MODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        try:
            module_name, attribute = internal_EXPORTS[name]
        except KeyError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = tuple(sorted((*internal_MODULES, *internal_EXPORTS)))
