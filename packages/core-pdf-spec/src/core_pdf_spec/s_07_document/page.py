# SPDX-License-Identifier: AGPL-3.0-only
"""Page dictionary rules, ISO 32000-2, 7.7.3 and 14.11.2."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import cast

from core_pdf_spec.s_07_syntax.inherited_values import inherited_dictionary_value
from core_pdf_spec.s_07_syntax.types import CachedPdfObject, InheritedValueMap, PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf_spec.types import Rectangle

PAGE_INHERITED_KEYS = ("MediaBox", "CropBox", "Rotate", "Resources")


@dataclass(frozen=True, slots=True)
class PageNode:
    dictionary: PdfDict
    inherited_values: InheritedValueMap


def page_clip(media: Rectangle, crop: Rectangle | None = None) -> Rectangle:
    if crop is None:
        return media
    x0 = max(min(crop[0], crop[2]), min(media[0], media[2]))
    y0 = max(min(crop[1], crop[3]), min(media[1], media[3]))
    x1 = min(max(crop[0], crop[2]), max(media[0], media[2]))
    y1 = min(max(crop[1], crop[3]), max(media[1], media[3]))
    return (x0, y0, max(x0, x1), max(y0, y1))


def page_rotation(value: object) -> int:
    if value is None:
        return 0
    if type(value) is not int or value % 90:
        raise ValueError("invalid page Rotate value")
    return value % 360


def page_inherited_values(
    nodes: Iterable[PdfDict],
    resolve: Callable[[object], object],
    keys: tuple[str, ...] = PAGE_INHERITED_KEYS,
) -> InheritedValueMap:
    """Select leaf-to-root candidates without reading shadowed ancestor entries."""
    values: InheritedValueMap = {}
    for node in nodes:
        for key in keys:
            if key in values:
                continue
            value = inherited_dictionary_value(node, key, None, resolve)
            if value is not None:
                values[key] = cast(CachedPdfObject, value)
    return values


def iter_page_nodes(
    root: object,
    resolve: Callable[[object], object],
    *,
    inherited_keys: tuple[str, ...] = PAGE_INHERITED_KEYS,
) -> Iterator[PageNode]:
    """Walk Kids in source order and carry each ancestor's inheritable values."""
    stack: list[tuple[object, tuple[PdfDict, ...], frozenset[int]]] = [(root, (), frozenset())]
    while stack:
        raw, parents, ancestors = stack.pop()
        current = resolve(raw)
        if not isinstance(current, dict):
            raise ValueError("invalid page tree node")
        current = cast(PdfDict, current)
        if id(current) in ancestors:
            raise ValueError("page tree cycle detected")
        kind = decoded_name(resolve(current.get("Type")))
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
            stack.extend((kid, parent_nodes, ancestry) for kid in reversed(kids))
        else:
            raise ValueError("invalid page tree node")


__all__ = (
    "PAGE_INHERITED_KEYS",
    "PageNode",
    "iter_page_nodes",
    "page_clip",
    "page_inherited_values",
    "page_rotation",
)
