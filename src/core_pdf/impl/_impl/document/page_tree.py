# SPDX-License-Identifier: AGPL-3.0-only
"""Damaged page-tree classification and traversal policy."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_document.page import PAGE_INHERITED_KEYS, PageNode, page_inherited_values
from core_pdf_spec.s_07_syntax.inherited_values import (
    inherited_dictionary_value,
)
from core_pdf_spec.s_07_syntax.types import (
    CachedPdfObject,
    InheritedValueMap,
    PdfDict,
    PdfValueResolver,
)
from core_pdf_spec.types import PdfReference

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


def collect_inherited_values(
    node: PdfDict,
    keys: tuple[str, ...],
    resolve_ref: Callable[[object], object],
) -> InheritedValueMap:
    values: InheritedValueMap = {}
    current: object = node
    # Values pin visited nodes so a freed dictionary's id cannot be reused by
    # a later resolved parent and read as a cycle.
    seen: dict[int, PdfDict] = {}
    references: set[tuple[int, int]] = set()
    while current is not None:
        if not isinstance(current, dict):
            break
        marker = id(current)
        if marker in seen:
            break
        seen[marker] = cast(PdfDict, current)

        current_dict = cast("PdfDict", current)
        for key in keys:
            if key in values:
                continue
            value = inherited_dictionary_value(current_dict, key, None, resolve_ref)
            if value is not None:
                values[key] = cast(CachedPdfObject, value)

        parent = current_dict.get("Parent")
        if isinstance(parent, PdfReference):
            reference = (parent.object_number, parent.generation_number)
            if reference in references:
                break
            references.add(reference)
        current = resolve_ref(parent) if parent is not None else None

    return values


def iter_page_nodes(
    root: object,
    resolve: Callable[[object], object],
    *,
    inherited_keys: tuple[str, ...] = PAGE_INHERITED_KEYS,
    node_type: Callable[[PdfDict], str | None] | None = None,
    on_invalid_child: Callable[[object], bool] | None = None,
    max_depth: int | None = None,
) -> Iterator[PageNode]:
    """Walk Kids in source order and carry each ancestor's inheritable values."""
    stack: list[tuple[object, int, tuple[PdfDict, ...], frozenset[int]]] = [
        (root, 0, (), frozenset())
    ]
    while stack:
        raw, depth, parents, ancestors = stack.pop()
        if max_depth is not None and depth > max_depth:
            raise ValueError("invalid page tree depth")
        current = resolve(raw)
        if not isinstance(current, dict):
            if depth and on_invalid_child is not None and on_invalid_child(current):
                continue
            raise ValueError("invalid page tree node")
        current = cast(PdfDict, current)
        if id(current) in ancestors:
            raise ValueError("page tree cycle detected")
        kind = (
            recover_pdf_name(resolve(current.get("Type")))
            if node_type is None
            else node_type(current)
        )
        if kind == "Page":
            yield PageNode(
                current, page_inherited_values((current, *parents), resolve, inherited_keys)
            )
        elif kind == "Pages":
            kids = resolve(current.get("Kids"))
            if not isinstance(kids, list):
                raise ValueError("invalid page tree Kids array")
            ancestry = ancestors | {id(current)}
            parent_nodes = (current, *parents)
            stack.extend((kid, depth + 1, parent_nodes, ancestry) for kid in reversed(kids))
        else:
            raise ValueError("invalid page tree node")
