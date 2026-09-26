# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator

from core_pdf_spec.s_07_syntax.trees import NameDecodeFn, NumberDecodeFn, ResolveFn, tree_entry
from core_pdf_spec.types import PdfReference


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
        recover=recover,
        recover_entries=recover or recover_entries,
        resolve_values=resolve_values,
        max_depth=max_depth,
    )


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    recover: bool = False,
) -> Iterator[tuple[str, object]]:
    yield from iter_tree_items(
        node,
        resolve,
        decode_name,
        key_field="Names",
        tree_name="name",
        recover=recover,
        recover_entries=recover,
        max_depth=100,
    )


def iter_tree_items[TreeKeyT](
    node: object,
    resolve: ResolveFn,
    decode_key: Callable[[object], TreeKeyT | None],
    *,
    key_field: str,
    tree_name: str,
    recover: bool = False,
    recover_entries: bool = False,
    resolve_values: bool = True,
    max_depth: int,
) -> Iterator[tuple[TreeKeyT, object]]:
    key_error = f"invalid {tree_name} tree key"
    # Keyed by id(); holding each node keeps its id from being reused.
    visited: dict[int, dict] = {}
    references: set[tuple[int, int]] = set()
    stack: list[tuple[object, int]] = [(node, 0)]
    while stack:
        current, current_depth = stack.pop()
        if current_depth > max_depth:
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree depth")
        if isinstance(current, PdfReference):
            reference = (current.object_number, current.generation_number)
            if reference in references:
                if recover:
                    continue
                raise ValueError(f"{tree_name} tree cycle detected")
            references.add(reference)
        current = resolve(current)
        if current is None:
            continue
        if not isinstance(current, dict):
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree node")
        marker = id(current)
        if marker in visited:
            if recover:
                continue
            raise ValueError(f"{tree_name} tree cycle detected")
        visited[marker] = current

        entries = resolve(current.get(key_field))
        if entries is not None:
            if not isinstance(entries, list):
                if recover:
                    continue
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            if len(entries) % 2 != 0 and not recover_entries:
                raise ValueError(f"invalid {tree_name} tree {key_field} array")
            entries_len = len(entries) - (len(entries) % 2)
            for index in range(0, entries_len, 2):
                decoded_key = decode_key(entries[index])
                try:
                    key, value = tree_entry(entries, index, decoded_key, key_error)
                except ValueError:
                    if recover_entries:
                        continue
                    raise
                yield key, resolve(value) if resolve_values else value

        kids = resolve(current.get("Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            if recover:
                continue
            raise ValueError(f"invalid {tree_name} tree Kids array")
        stack.extend((kid, current_depth + 1) for kid in reversed(kids))
