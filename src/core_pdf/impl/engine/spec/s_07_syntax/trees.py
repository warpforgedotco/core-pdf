# SPDX-License-Identifier: AGPL-3.0-only
"""Shared traversal kernels for PDF name and number trees."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

from core_pdf.impl.engine.spec.s_07_syntax.pdfdict import lookup_dict_key

ResolveFn = Callable[[object], object]
NameDecodeFn = Callable[[object], str | None]
NumberDecodeFn = Callable[[object], int | None]
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
    recover_entries: bool = False,
    resolve_values: bool = True,
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[TreeKeyT, object]]:
    """Iterate a tree without recursion while validating its node shape."""

    if seen is None:
        seen = set()
    stack: list[tuple[object, int]] = [(node, depth)]
    while stack:
        current, current_depth = stack.pop()
        if current_depth > max_depth:
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
            if len(entries) % 2 != 0 and not (recover or recover_entries):
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            entries_len = len(entries) - (len(entries) % 2)
            for index in range(0, entries_len, 2):
                key = decode_key(entries[index])
                if key is None:
                    if recover or recover_entries:
                        continue
                    raise ValueError(key_error)
                value = entries[index + 1]
                yield key, resolve(value) if resolve_values else value

        kids = resolve(lookup_dict_key(current, "Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree Kids array")
        for kid in reversed(kids):
            stack.append((kid, current_depth + 1))


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    decode_number: NumberDecodeFn | None = None,
    recover: bool = False,
    recover_entries: bool = False,
    resolve_values: bool = True,
    tree_name: str = "number",
    max_depth: int = 100,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[int, object]]:
    decode = decode_number
    if decode is None:

        def decode(value: object) -> int | None:
            value = resolve(value)
            return value if type(value) is int else None

    yield from internal_iter_tree_items(
        node,
        resolve,
        decode,
        key_field="Nums",
        tree_name=tree_name,
        key_error=f"invalid {tree_name} tree key",
        recover=recover,
        recover_entries=recover_entries,
        resolve_values=resolve_values,
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
    recover_entries: bool = False,
    resolve_values: bool = True,
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
        recover_entries=recover_entries,
        resolve_values=resolve_values,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )
