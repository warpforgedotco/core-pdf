# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled PDF name and number tree traversal kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key

ResolveFn = Callable[[object], object]
NameDecodeFn = Callable[[object], str | None]
TreeKeyT = TypeVar("TreeKeyT")


def internal_iter_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_key: Callable[[object], TreeKeyT | None],
    *,
    key_field: str,
    tree_name: str,
    key_error: str,
    recover: bool = False,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[TreeKeyT, object]]:
    if seen is None:
        seen = set()
    stack: list[tuple[object, int]] = [(node, depth)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree depth")
        current = resolve(current)
        if current is None:
            continue
        if not isinstance(current, dict):
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree node")
        marker = id(current)
        if marker in seen:
            if recover:
                continue
            raise ValueError(f"{tree_name} tree cycle detected")
        seen.add(marker)

        entries = resolve(lookup_dict_key(current, key_field))
        if entries is not None:
            if not isinstance(entries, list):
                if recover:
                    continue
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            if len(entries) % 2 != 0 and not recover:
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            entries_len = len(entries) - (len(entries) % 2)
            for index in range(0, entries_len, 2):
                key = decode_key(entries[index])
                if key is None:
                    if recover:
                        continue
                    raise ValueError(key_error)
                yield key, resolve(entries[index + 1])

        kids = resolve(lookup_dict_key(current, "Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree Kids array")
        for kid in reversed(kids):
            stack.append((kid, depth + 1))


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    recover: bool = False,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[int, object]]:
    def decode_number(value: object) -> int | None:
        value = resolve(value)
        return value if type(value) is int else None

    yield from internal_iter_tree_items(
        node,
        resolve,
        decode_number,
        key_field="Nums",
        tree_name="number",
        key_error="invalid number tree key",
        recover=recover,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    recover: bool = False,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[str, object]]:
    yield from internal_iter_tree_items(
        node,
        resolve,
        decode_name,
        key_field="Names",
        tree_name="name",
        key_error="invalid name tree key",
        recover=recover,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )
