# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 7.7 document catalog, pages, name trees, and related structures."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument
    from core_pdf.impl.engine.spec.s_07_document.document_labels import (
        decode_page_label_prefix,
        format_alpha,
        format_page_label,
        format_roman,
        infer_page_tree_node_type,
        normalize_page_label_style,
    )
    from core_pdf.impl.engine.spec.s_07_document.metadata import resolve_metadata
    from core_pdf.impl.engine.spec.s_07_document.name_trees import (
        iter_name_tree_items,
        iter_number_tree_items,
    )
    from core_pdf.impl.engine.spec.s_07_document.page import PdfPage

_EXPORTS = {
    "PdfDocument": ("core_pdf.impl.engine.spec.s_07_document.document", "PdfDocument"),
    "PdfPage": ("core_pdf.impl.engine.spec.s_07_document.page", "PdfPage"),
    "decode_page_label_prefix": (
        "core_pdf.impl.engine.spec.s_07_document.document_labels",
        "decode_page_label_prefix",
    ),
    "format_alpha": ("core_pdf.impl.engine.spec.s_07_document.document_labels", "format_alpha"),
    "format_page_label": (
        "core_pdf.impl.engine.spec.s_07_document.document_labels",
        "format_page_label",
    ),
    "format_roman": ("core_pdf.impl.engine.spec.s_07_document.document_labels", "format_roman"),
    "infer_page_tree_node_type": (
        "core_pdf.impl.engine.spec.s_07_document.document_labels",
        "infer_page_tree_node_type",
    ),
    "iter_name_tree_items": (
        "core_pdf.impl.engine.spec.s_07_document.name_trees",
        "iter_name_tree_items",
    ),
    "iter_number_tree_items": (
        "core_pdf.impl.engine.spec.s_07_document.name_trees",
        "iter_number_tree_items",
    ),
    "normalize_page_label_style": (
        "core_pdf.impl.engine.spec.s_07_document.document_labels",
        "normalize_page_label_style",
    ),
    "resolve_metadata": ("core_pdf.impl.engine.spec.s_07_document.metadata", "resolve_metadata"),
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
    "PdfDocument",
    "PdfPage",
    "decode_page_label_prefix",
    "format_alpha",
    "format_page_label",
    "format_roman",
    "infer_page_tree_node_type",
    "iter_name_tree_items",
    "iter_number_tree_items",
    "normalize_page_label_style",
    "resolve_metadata",
)
