# SPDX-License-Identifier: AGPL-3.0-only
"""Tolerant name and number tree readers using the shared syntax traversal."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

from core_pdf_spec.s_07_syntax.trees import NameDecodeFn, NumberDecodeFn, ResolveFn, tree_entry


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
    yield from internal_iter_number_tree_items(
        node,
        resolve,
        decode_number=decode_number,
        on_error=lambda _: recover,
        on_entry_error=lambda _: recover or recover_entries,
        resolve_values=resolve_values,
        tree_name=tree_name,
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
    yield from internal_iter_name_tree_items(
        node,
        resolve,
        decode_name,
        on_error=lambda _: recover,
        on_entry_error=lambda _: recover or recover_entries,
        resolve_values=resolve_values,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )


TreeKeyT = TypeVar("TreeKeyT")


def internal_iter_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_key: Callable[[object], TreeKeyT | None],
    *,
    key_field: str,
    tree_name: str,
    key_error: str,
    on_error: Callable[[str], bool] | None = None,
    on_entry_error: Callable[[str], bool] | None = None,
    resolve_values: bool = True,
    max_depth: int | None = None,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[TreeKeyT, object]]:
    """Iterate a tree without recursion while validating its node shape."""

    if seen is None:
        seen = set()
    stack: list[tuple[object, int]] = [(node, depth)]
    while stack:
        current, current_depth = stack.pop()
        if max_depth is not None and current_depth > max_depth:
            if on_error is not None and on_error(f"invalid {tree_name} tree depth"):
                continue
            raise ValueError(f"invalid {tree_name} tree depth")
        current = resolve(current)
        if current is None:
            continue
        if not isinstance(current, dict):
            if on_error is not None and on_error(f"invalid {tree_name} tree node"):
                continue
            raise ValueError(f"invalid {tree_name} tree node")
        marker = id(current)
        if marker in seen:
            if on_error is not None and on_error(f"{tree_name} tree cycle detected"):
                continue
            raise ValueError(f"{tree_name} tree cycle detected")
        seen.add(marker)

        entries = resolve(current.get(key_field))
        if entries is not None:
            if not isinstance(entries, list):
                if on_error is not None and on_error(f"invalid {tree_name} tree {key_field} array"):
                    continue
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            if len(entries) % 2 != 0 and not (
                on_entry_error is not None
                and on_entry_error(f"invalid {tree_name} tree {key_field} array")
            ):
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            entries_len = len(entries) - (len(entries) % 2)
            for index in range(0, entries_len, 2):
                decoded_key = decode_key(entries[index])
                try:
                    key, value = tree_entry(entries, index, decoded_key, key_error)
                except ValueError:
                    if on_entry_error is not None and on_entry_error(key_error):
                        continue
                    raise
                yield key, resolve(value) if resolve_values else value

        kids = resolve(current.get("Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if on_error is not None and on_error(f"invalid {tree_name} tree Kids array"):
                continue
            raise ValueError(f"invalid {tree_name} tree Kids array")
        for kid in reversed(kids):
            stack.append((kid, current_depth + 1))


def internal_iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    decode_number: NumberDecodeFn | None = None,
    on_error: Callable[[str], bool] | None = None,
    on_entry_error: Callable[[str], bool] | None = None,
    resolve_values: bool = True,
    tree_name: str = "number",
    max_depth: int | None = None,
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
        on_error=on_error,
        on_entry_error=on_entry_error,
        resolve_values=resolve_values,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )


def internal_iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    on_error: Callable[[str], bool] | None = None,
    on_entry_error: Callable[[str], bool] | None = None,
    resolve_values: bool = True,
    max_depth: int | None = None,
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
        on_error=on_error,
        on_entry_error=on_entry_error,
        resolve_values=resolve_values,
        max_depth=max_depth,
        depth=depth,
        seen=seen,
    )
