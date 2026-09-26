# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.types import PdfReference

ResolveFn = Callable[[object], object]
NameDecodeFn = Callable[[object], str | None]
NumberDecodeFn = Callable[[object], int | None]
TreeKeyT = TypeVar("TreeKeyT")


class MalformedTreeNode(str):
    """The message for a tree node that is not a dictionary, and the node.

    A str, so every on_malformed callback takes it as the message it raises
    or skips; one that wants to treat some nodes differently -- a reader
    that reads a null kid as an empty subtree -- looks at `node`.
    """

    node: object

    def __new__(cls, message: str, node: object) -> MalformedTreeNode:
        report = super().__new__(cls, message)
        report.node = node
        return report


MalformedFn = Callable[[str], None]


def raise_malformed(message: str) -> None:
    """The strict response to a malformed tree: ValueError(message)."""
    raise ValueError(str(message))


def iter_tree_items[TreeKeyT](
    node: object,
    resolve: ResolveFn,
    decode_key: Callable[[object], TreeKeyT | None],
    *,
    key_field: str,
    tree_name: str,
    key_error: str,
    resolve_values: bool = True,
    max_depth: int | None = None,
    on_malformed: MalformedFn = raise_malformed,
    on_malformed_entry: MalformedFn | None = None,
) -> Iterator[tuple[TreeKeyT, object]]:
    """The key and value of each entry of the tree at node, in key order.

    A malformed node -- a cycle, a non-dictionary, a Nums, Names or Kids
    entry that is not an array, or one past max_depth -- is reported to
    on_malformed, and a malformed entry -- an odd-length array or an
    undecodable key -- to on_malformed_entry, which defaults to it. The
    default raises ValueError; a reader's callback that returns instead has
    the node skipped, the entry skipped, or the odd array's last key
    dropped. A null root is an empty tree; any other null node is malformed,
    reported as a MalformedTreeNode whose node is None.
    """
    entry_malformed = on_malformed if on_malformed_entry is None else on_malformed_entry
    seen: dict[int, PdfDict] = {}
    references: set[tuple[int, int]] = set()
    stack: list[tuple[object, int]] = [(node, 0)]
    is_root = True
    while stack:
        raw, depth = stack.pop()
        if max_depth is not None and depth > max_depth:
            on_malformed(f"invalid {tree_name} tree depth")
            continue
        if isinstance(raw, PdfReference):
            reference = (raw.object_number, raw.generation_number)
            if reference in references:
                on_malformed(f"{tree_name} tree cycle detected")
                continue
            references.add(reference)
        resolved = resolve(raw)
        if resolved is None and is_root:
            is_root = False
            continue
        is_root = False
        if not isinstance(resolved, dict):
            on_malformed(MalformedTreeNode(f"invalid {tree_name} tree node", resolved))
            continue
        current = resolved
        marker = id(current)
        if marker in seen:
            on_malformed(f"{tree_name} tree cycle detected")
            continue
        seen[marker] = current
        entries = resolve(current.get(key_field))
        if entries is not None:
            if not isinstance(entries, list):
                on_malformed(f"invalid {tree_name} tree {key_field} array")
                continue
            if len(entries) % 2:
                entry_malformed(f"invalid {tree_name} tree {key_field} array")
            for index in range(0, len(entries) - len(entries) % 2, 2):
                try:
                    key = decode_key(entries[index])
                except ValueError as error:
                    entry_malformed(str(error))
                    continue
                if key is None:
                    entry_malformed(key_error)
                    continue
                value = entries[index + 1]
                yield key, resolve(value) if resolve_values else value
        kids = resolve(current.get("Kids"))
        if kids is None:
            continue
        if not isinstance(kids, list):
            on_malformed(f"invalid {tree_name} tree Kids array")
            continue
        stack.extend((kid, depth + 1) for kid in reversed(kids))


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    decode_number: NumberDecodeFn | None = None,
    resolve_values: bool = True,
    tree_name: str = "number",
    max_depth: int | None = None,
    on_malformed: MalformedFn = raise_malformed,
    on_malformed_entry: MalformedFn | None = None,
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
        max_depth=max_depth,
        on_malformed=on_malformed,
        on_malformed_entry=on_malformed_entry,
    )


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    resolve_values: bool = True,
    max_depth: int | None = None,
    on_malformed: MalformedFn = raise_malformed,
    on_malformed_entry: MalformedFn | None = None,
) -> Iterator[tuple[str, object]]:
    yield from iter_tree_items(
        node,
        resolve,
        decode_name,
        key_field="Names",
        tree_name="name",
        key_error="invalid name tree key",
        resolve_values=resolve_values,
        max_depth=max_depth,
        on_malformed=on_malformed,
        on_malformed_entry=on_malformed_entry,
    )


__all__ = (
    "MalformedFn",
    "MalformedTreeNode",
    "NameDecodeFn",
    "NumberDecodeFn",
    "ResolveFn",
    "iter_name_tree_items",
    "iter_number_tree_items",
    "iter_tree_items",
    "raise_malformed",
)
