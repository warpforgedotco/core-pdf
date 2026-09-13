# SPDX-License-Identifier: AGPL-3.0-only
"""Shallow reference chains and identity-preserving PDF object graph resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference

internal_CONTAINER_TYPES = (dict, list, tuple, PdfStream)


def resolve_reference_chain(value: object, resolve: Callable[[object], object]) -> object:
    """Follow scalar references, retaining a repeated reference or terminal object."""
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


@dataclass(slots=True)
class internal_ResolutionNode:
    original: object
    values: list[object] = field(default_factory=list)
    keys: tuple[object, ...] = ()
    parents: set[int] = field(default_factory=set)
    changed: bool = False


def internal_resolve_object_graph(value: object, resolve: Callable[[object], object]) -> object:
    """Copy exactly the containers affected by resolution, including whole cycles.

    Reverse-edge propagation marks every ancestor of a changed slot. In a cycle,
    this marks the complete component before any result is populated, so mutable
    placeholders can retain both sharing and backedges without changing the input.
    """
    reference_values: dict[tuple[int, int], object] = {}

    def resolve_once(reference: object) -> object:
        ref = cast(PdfReference, reference)
        marker = (ref.object_number, ref.generation_number)
        if marker not in reference_values:
            reference_values[marker] = resolve(ref)
        return reference_values[marker]

    root = resolve_reference_chain(value, resolve_once)
    if type(root) not in internal_CONTAINER_TYPES:
        return root

    nodes = {id(root): internal_ResolutionNode(root)}
    pending = [id(root)]
    while pending:
        marker = pending.pop()
        node = nodes[marker]
        original = node.original
        if type(original) is PdfStream:
            values: list[object] = [original.dictionary]
        elif type(original) is dict:
            mapping = cast(dict[object, object], original)
            node.keys = tuple(mapping)
            values = list(mapping.values())
        else:
            values = list(cast(list[object] | tuple[object, ...], original))
            node.changed = type(original) is tuple

        for item in values:
            resolved = resolve_reference_chain(item, resolve_once)
            node.values.append(resolved)
            if resolved is not item:
                node.changed = True
            if type(resolved) in internal_CONTAINER_TYPES:
                child_marker = id(resolved)
                child = nodes.get(child_marker)
                if child is None:
                    child = nodes[child_marker] = internal_ResolutionNode(resolved)
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

    # Stream dictionaries are already allocated, including dictionaries which
    # refer back to the stream. replace preserves source bytes and decoder state.
    for marker, node in nodes.items():
        if node.changed and type(node.original) is PdfStream:
            results[marker] = node.original.replace(
                dictionary=cast(dict[object, object], results[id(node.original.dictionary)])
            )

    for marker, node in nodes.items():
        if not node.changed or type(node.original) is PdfStream:
            continue
        resolved_values = [
            results[id(item)] if type(item) in internal_CONTAINER_TYPES else item
            for item in node.values
        ]
        if type(node.original) is dict:
            cast(dict[object, object], results[marker]).update(
                zip(node.keys, resolved_values, strict=True)
            )
        else:
            cast(list[object], results[marker]).extend(resolved_values)
    return results[id(root)]


__all__ = ("resolve_reference_chain",)
