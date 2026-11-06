# Ported from pypdf 6.21.4's PdfDocCommon._flatten, Copyright (c) 2006, Mathieu Fenniak,
# under its BSD-style license.
"""pypdf's walk over a page tree, listing its pages with what they inherit.

The tree is walked with an explicit stack, bound by the configured depth and entry limits
rather than the interpreter's recursion limit, and a failed walk leaves no partial list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from core_pdf_compat.pypdf._configuration import get_configuration
from core_pdf_compat.pypdf.errors import LimitReachedError, PdfReadError
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NullObject,
    is_null_or_none,
)

logger = logging.getLogger(__name__)

INHERITABLE_PAGE_ATTRIBUTES = ("/Resources", "/MediaBox", "/CropBox", "/Rotate")

# A page's node, its reference, and what it inherits; returns the page.
MakePage = Callable[[DictionaryObject, IndirectObject | None, dict[str, Any]], Any]
# A /Pages node's value for an inheritable key.
InheritedValue = Callable[[DictionaryObject, str], Any]


class _Item:
    __slots__ = ("depth", "inherit", "node", "reference")

    def __init__(
        self,
        node: DictionaryObject,
        inherit: dict[str, Any],
        reference: IndirectObject | None,
        depth: int,
    ) -> None:
        self.node = node
        self.inherit = inherit
        self.reference = reference
        self.depth = depth


def page_entry_count(node: DictionaryObject, inherit: dict[str, Any], copied: bool) -> int:
    """The entries a page copies: its own, if copied, and what it inherits and lacks."""
    own = len(node) if copied else 0
    return own + sum(key not in node for key in inherit)


def flatten_page_tree(
    pages: Any,
    *,
    strict: bool,
    make_page: MakePage,
    inherited_value: InheritedValue,
    count_own_entries: Callable[[IndirectObject | None], bool],
) -> list[Any]:
    if not isinstance(pages, DictionaryObject):
        raise PdfReadError("Invalid object in /Pages")
    configuration = get_configuration()
    flattened: list[Any] = []
    entry_count = 0
    remaining_page_entries = configuration.page_tree_maximum_page_entries
    # The /Pages nodes on the path to the current node, for multi-hop cycles.
    ancestor_ids: set[int] = set()
    stack: list[tuple[_Item, Iterator[Any]]] = []
    item: _Item | None = _Item(pages, {}, None, 0)
    while item is not None:
        if item.depth > configuration.page_tree_maximum_depth:
            raise LimitReachedError(
                f"Maximum page tree depth reached: {item.depth} > "
                f"{configuration.page_tree_maximum_depth}."
            )
        node = item.node
        if "/Type" in node:
            node_type = node["/Type"]
        elif "/Kids" not in node:
            # Without /Type, a node is a page only if it has a structural page key.
            if strict and not any(key in node for key in ("/Contents", "/MediaBox", "/Parent")):
                raise PdfReadError(f"Non-page object reached through /Kids: {node!r}")
            node_type = "/Page"
        else:
            node_type = "/Pages"
        if node_type == "/Pages":
            ancestor_ids.add(id(node))
            inherit = dict(item.inherit)
            for key in INHERITABLE_PAGE_ATTRIBUTES:
                if key in node:
                    inherit[key] = inherited_value(node, key)
            kids = node.get("/Kids", ArrayObject()).get_object()
            if isinstance(kids, NullObject):
                kids = ArrayObject()
            elif not isinstance(kids, ArrayObject):
                raise PdfReadError(f"Expected /Kids to be an array, got {type(kids).__name__}.")
            item.inherit = inherit
            stack.append((item, iter(kids)))
        elif node_type == "/Page":
            remaining_page_entries -= page_entry_count(
                node, item.inherit, count_own_entries(item.reference)
            )
            if remaining_page_entries < 0:
                raise LimitReachedError("Maximum page tree page entry limit exceeded!")
            flattened.append(make_page(node, item.reference, item.inherit))
        item = None
        while stack and item is None:
            parent, kids_iterator = stack[-1]
            parent_reference = getattr(parent.node, "indirect_reference", object())
            for kid in kids_iterator:
                if getattr(kid, "indirect_reference", object()) == parent_reference:
                    raise PdfReadError("Detected cyclic page references.")
                child = kid.get_object()
                if not isinstance(child, DictionaryObject):
                    if not is_null_or_none(child):
                        logger.warning(
                            "Ignoring page tree entry that is not a dictionary: %s", child
                        )
                    continue
                if not child:
                    # An empty dictionary would otherwise be read as a blank page.
                    continue
                if id(child) in ancestor_ids:
                    raise PdfReadError("Detected cyclic page references.")
                entry_count += 1
                if entry_count > configuration.page_tree_maximum_entries:
                    raise LimitReachedError(
                        "Maximum page tree entry limit reached: "
                        f"{entry_count} > {configuration.page_tree_maximum_entries}."
                    )
                reference = kid if isinstance(kid, IndirectObject) else None
                item = _Item(child, parent.inherit, reference, parent.depth + 1)
                break
            else:
                stack.pop()
                ancestor_ids.discard(id(parent.node))
    return flattened
