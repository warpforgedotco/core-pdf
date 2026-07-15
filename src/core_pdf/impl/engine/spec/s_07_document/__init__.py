# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 7.7 document catalog, pages, name trees, and related structures."""

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
