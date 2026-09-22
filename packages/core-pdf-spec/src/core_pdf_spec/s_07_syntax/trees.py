# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

from core_pdf_spec.types import PdfReference

ResolveFn = Callable[[object], object]
NameDecodeFn = Callable[[object], str | None]
NumberDecodeFn = Callable[[object], int | None]
TreeKeyT = TypeVar("TreeKeyT")


def tree_node(value: object, resolve: ResolveFn, tree_name: str) -> dict:
    return internal_tree_node(resolve(value), tree_name)


def internal_tree_node(current: object, tree_name: str) -> dict:
    if not isinstance(current, dict):
        raise ValueError(f"invalid {tree_name} tree node")
    return current


def tree_array(node: dict, field: str, resolve: ResolveFn, tree_name: str) -> list | None:
    value = resolve(node.get(field))
    if value is not None and not isinstance(value, list):
        raise ValueError(f"invalid {tree_name} tree {field} array")
    return value


def tree_entry[TreeKeyT](
    entries: list, index: int, key: TreeKeyT | None, key_error: str
) -> tuple[TreeKeyT, object]:
    if index + 1 >= len(entries):
        raise ValueError(key_error)
    if key is None:
        raise ValueError(key_error)
    return key, entries[index + 1]


def iter_tree_items[TreeKeyT](
    node: object,
    resolve: ResolveFn,
    decode_key: Callable[[object], TreeKeyT | None],
    *,
    key_field: str,
    tree_name: str,
    key_error: str,
    resolve_values: bool = True,
) -> Iterator[tuple[TreeKeyT, object]]:
    seen: dict[int, dict] = {}
    references: set[tuple[int, int]] = set()
    stack = [node]
    is_root = True
    while stack:
        raw = stack.pop()
        if isinstance(raw, PdfReference):
            reference = (raw.object_number, raw.generation_number)
            if reference in references:
                raise ValueError(f"{tree_name} tree cycle detected")
            references.add(reference)
        resolved = resolve(raw)
        if is_root and resolved is None:
            return
        is_root = False
        current = internal_tree_node(resolved, tree_name)
        marker = id(current)
        if marker in seen:
            raise ValueError(f"{tree_name} tree cycle detected")
        seen[marker] = current
        entries = tree_array(current, key_field, resolve, tree_name)
        if entries is not None:
            if len(entries) % 2:
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            for index in range(0, len(entries), 2):
                key, value = tree_entry(entries, index, decode_key(entries[index]), key_error)
                yield key, resolve(value) if resolve_values else value
        kids = tree_array(current, "Kids", resolve, tree_name)
        if kids is not None:
            stack.extend(reversed(kids))


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    decode_number: NumberDecodeFn | None = None,
    resolve_values: bool = True,
    tree_name: str = "number",
) -> Iterator[tuple[int, object]]:
    decode = decode_number
    if decode is None:

        def decode(value: object) -> int | None:
            value = resolve(value)
            return value if type(value) is int else None

    yield from iter_tree_items(
        node,
        resolve,
        decode,
        key_field="Nums",
        tree_name=tree_name,
        key_error=f"invalid {tree_name} tree key",
        resolve_values=resolve_values,
    )


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    resolve_values: bool = True,
) -> Iterator[tuple[str, object]]:
    yield from iter_tree_items(
        node,
        resolve,
        decode_name,
        key_field="Names",
        tree_name="name",
        key_error="invalid name tree key",
        resolve_values=resolve_values,
    )


__all__ = (
    "NameDecodeFn",
    "NumberDecodeFn",
    "ResolveFn",
    "iter_name_tree_items",
    "iter_number_tree_items",
    "tree_array",
    "tree_entry",
    "tree_node",
)
