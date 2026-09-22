# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from math import isfinite
from typing import Any, ClassVar, NoReturn, Self, cast

from core_pdf_spec.s_07_syntax.inherited_values import inherited_dictionary_value
from core_pdf_spec.s_07_syntax.types import CachedPdfObject, InheritedValueMap, PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_pdf_spec.types import Rectangle

internal_frozen_setattr = object.__setattr__


PAGE_INHERITED_KEYS = ("MediaBox", "CropBox", "Rotate", "Resources")


class PageNode:
    __slots__ = ("dictionary", "inherited_values")

    dictionary: PdfDict
    inherited_values: InheritedValueMap

    __fields__: ClassVar[tuple[str, ...]] = ("dictionary", "inherited_values")
    __match_args__ = ("dictionary", "inherited_values")

    def __init__(self, dictionary: PdfDict, inherited_values: InheritedValueMap) -> None:
        internal_frozen_setattr(self, "dictionary", dictionary)
        internal_frozen_setattr(self, "inherited_values", inherited_values)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"dictionary={self.dictionary!r}, "
            f"inherited_values={self.inherited_values!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.dictionary == other.dictionary and self.inherited_values == other.inherited_values
        )

    def __hash__(self) -> int:
        return hash((self.dictionary, self.inherited_values))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        dictionary = changes.pop("dictionary", self.dictionary)
        inherited_values = changes.pop("inherited_values", self.inherited_values)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(dictionary, inherited_values)


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


def page_user_unit(value: object) -> float:
    if value is None:
        return 1.0
    if type(value) not in (int, float):
        raise ValueError("invalid page UserUnit value")
    try:
        unit = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError("invalid page UserUnit value") from error
    if not isfinite(unit) or unit <= 0.0:
        raise ValueError("invalid page UserUnit value")
    return unit


def page_inherited_values(
    nodes: Iterable[PdfDict],
    resolve: Callable[[object], object],
    keys: tuple[str, ...] = PAGE_INHERITED_KEYS,
) -> InheritedValueMap:
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
    node_type: Callable[[PdfDict], str | None] | None = None,
    on_invalid_child: Callable[[object], bool] | None = None,
    max_depth: int | None = None,
) -> Iterator[PageNode]:
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
            decoded_name(resolve(current.get("Type"))) if node_type is None else node_type(current)
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


__all__ = (
    "PAGE_INHERITED_KEYS",
    "PageNode",
    "iter_page_nodes",
    "page_clip",
    "page_inherited_values",
    "page_rotation",
    "page_user_unit",
)
