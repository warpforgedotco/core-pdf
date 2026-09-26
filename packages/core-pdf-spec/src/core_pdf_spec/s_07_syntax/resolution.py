# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference
from core_records import ReplaceFields, ReprFields

CONTAINER_TYPES = (dict, list, tuple, PdfStream)


def resolve_reference_chain(value: object, resolve: Callable[[object], object]) -> object:
    if type(value) is not PdfReference:
        return value
    seen: set[tuple[int, int]] = set()
    while type(value) is PdfReference:
        marker = (value.object_number, value.generation_number)
        if marker in seen:
            return value
        seen.add(marker)
        value = resolve(value)
    return value


class ResolutionNode(ReprFields, ReplaceFields):
    __slots__ = ("original", "values", "keys", "parents", "changed")

    original: object
    values: list[object]
    keys: tuple[object, ...]
    parents: set[int]
    changed: bool

    __fields__: ClassVar[tuple[str, ...]] = ("original", "values", "keys", "parents", "changed")
    __match_args__ = ("original", "values", "keys", "parents", "changed")

    def __init__(
        self,
        original: object,
        values: list[object] | None = None,
        keys: tuple[object, ...] = (),
        parents: set[int] | None = None,
        changed: bool = False,
    ) -> None:
        self.original = original
        self.values = [] if values is None else values
        self.keys = keys
        self.parents = set() if parents is None else parents
        self.changed = changed

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.original == other.original
            and self.values == other.values
            and self.keys == other.keys
            and self.parents == other.parents
            and self.changed == other.changed
        )

    __hash__ = None  # type: ignore[assignment]


def resolve_object_graph(value: object, resolve: Callable[[object], object]) -> object:
    reference_values: dict[tuple[int, int], object] = {}

    def resolve_once(reference: object) -> object:
        ref = reference
        marker = (ref.object_number, ref.generation_number)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        if marker not in reference_values:
            reference_values[marker] = resolve(ref)
        return reference_values[marker]

    root = resolve_reference_chain(value, resolve_once)
    if type(root) not in CONTAINER_TYPES:
        return root

    nodes = {id(root): ResolutionNode(root)}
    pending = [id(root)]
    while pending:
        marker = pending.pop()
        node = nodes[marker]
        original = node.original
        if type(original) is PdfStream:
            values: list[object] = [original.dictionary]
        elif type(original) is dict:
            mapping = original
            node.keys = tuple(mapping)
            values = list(mapping.values())
        else:
            values = list(original)  # type: ignore[call-overload]  # ty: ignore[invalid-argument-type]
            node.changed = type(original) is tuple

        for item in values:
            resolved = resolve_reference_chain(item, resolve_once)
            node.values.append(resolved)
            if resolved is not item:
                node.changed = True
            if type(resolved) in CONTAINER_TYPES:
                child_marker = id(resolved)
                child = nodes.get(child_marker)
                if child is None:
                    child = nodes[child_marker] = ResolutionNode(resolved)
                    pending.append(child_marker)
                child.parents.add(marker)

    pending = [marker for marker, node in nodes.items() if node.changed]
    while pending:
        for parent in nodes[pending.pop()].parents:
            if not nodes[parent].changed:
                nodes[parent].changed = True
                pending.append(parent)

    results: dict[int, object] = {}
    for marker, node in nodes.items():
        if not node.changed or type(node.original) is PdfStream:
            results[marker] = node.original
        else:
            results[marker] = {} if type(node.original) is dict else []

    for marker, node in nodes.items():
        if node.changed and type(node.original) is PdfStream:
            results[marker] = node.original.replace(
                dictionary=results[id(node.original.dictionary)]  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )

    for marker, node in nodes.items():
        if not node.changed or type(node.original) is PdfStream:
            continue
        resolved_values = [
            results[id(item)] if type(item) in CONTAINER_TYPES else item for item in node.values
        ]
        if type(node.original) is dict:
            results[marker].update(zip(node.keys, resolved_values, strict=True))  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        else:
            results[marker].extend(resolved_values)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return results[id(root)]


__all__ = ("resolve_reference_chain",)
