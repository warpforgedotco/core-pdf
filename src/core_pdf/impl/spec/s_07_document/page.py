# SPDX-License-Identifier: AGPL-3.0-only
"""Page dictionary rules, ISO 32000-2, 7.7.3 and 14.11.2."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

from core_pdf.impl.spec.s_07_syntax.types import CachedPdfObject, InheritedValueMap, PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.types import Rectangle

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
    stack: list[tuple[object, int, InheritedValueMap, frozenset[int]]] = [
        (root, 0, {}, frozenset())
    ]
    while stack:
        raw, depth, inherited, ancestors = stack.pop()
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
            normalize_pdf_name(resolve(current.get("Type")))
            if node_type is None
            else node_type(current)
        )
        values = dict(inherited)
        for key in inherited_keys:
            value = current.get(key)
            if value is not None:
                values[key] = cast(CachedPdfObject, value)
        if kind == "Page":
            yield PageNode(current, values)
        elif kind == "Pages":
            kids = resolve(current.get("Kids"))
            if not isinstance(kids, list):
                raise ValueError("invalid page tree Kids array")
            ancestry = ancestors | {id(current)}
            stack.extend((kid, depth + 1, values, ancestry) for kid in reversed(kids))
        else:
            raise ValueError("invalid page tree node")
