# SPDX-License-Identifier: AGPL-3.0-only
"""Damaged page-tree classification and traversal policy."""

from __future__ import annotations

from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver

MAX_PAGE_TREE_DEPTH = 100


def resolve_page_tree_node_type(resolver: PdfValueResolver, node: PdfDict) -> str | None:
    """Resolve a page-tree node type, falling back to structural inference."""
    node_type = resolver.resolve_name(node.get("Type"))
    if node_type is not None:
        return node_type
    inferred = infer_page_tree_node_type(node, include_page_properties=False)
    if inferred is not None:
        return inferred
    # A parent pointer is the defining structural signal for an untyped leaf.
    # Content/media/resource keys alone also occur in Form XObjects and other
    # dictionaries and must not promote those objects into the page tree.
    if node.get("Parent") is not None:
        return "Page"
    return None


def infer_page_tree_node_type(
    node: PdfDict,
    *,
    include_page_properties: bool = True,
) -> str | None:
    if node.get("Kids") is not None:
        return "Pages"
    if node.get("Count") is not None:
        return "Pages"
    if not include_page_properties:
        return None
    for key in ("Contents", "MediaBox", "Resources", "Parent", "Annots"):
        if node.get(key) is not None:
            return "Page"
    return None
