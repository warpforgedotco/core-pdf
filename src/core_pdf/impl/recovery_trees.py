# SPDX-License-Identifier: AGPL-3.0-only
"""Spec's number and name tree walkers with the reader's limits: a null node
is an empty subtree, and trees end 100 levels down."""

from __future__ import annotations

from collections.abc import Iterator

from core_pdf_spec.s_07_syntax import trees
from core_pdf_spec.s_07_syntax.trees import (
    MalformedFn,
    MalformedTreeNode,
    NameDecodeFn,
    NumberDecodeFn,
    ResolveFn,
    raise_malformed,
)

MAX_TREE_DEPTH = 100


def skipping_null_nodes(on_malformed: MalformedFn) -> MalformedFn:
    """on_malformed, except that a null node is skipped as an empty subtree."""

    def report(message: str) -> None:
        if isinstance(message, MalformedTreeNode) and message.node is None:
            return
        on_malformed(message)

    return report


def iter_number_tree_items(
    node: object,
    resolve: ResolveFn,
    *,
    decode_number: NumberDecodeFn | None = None,
    on_malformed: MalformedFn = raise_malformed,
    on_malformed_entry: MalformedFn | None = None,
    resolve_values: bool = True,
    tree_name: str = "number",
    max_depth: int = MAX_TREE_DEPTH,
) -> Iterator[tuple[int, object]]:
    yield from trees.iter_number_tree_items(
        node,
        resolve,
        decode_number=decode_number,
        resolve_values=resolve_values,
        tree_name=tree_name,
        max_depth=max_depth,
        on_malformed=skipping_null_nodes(on_malformed),
        on_malformed_entry=on_malformed_entry,
    )


def iter_name_tree_items(
    node: object,
    resolve: ResolveFn,
    decode_name: NameDecodeFn,
    *,
    on_malformed: MalformedFn = raise_malformed,
) -> Iterator[tuple[str, object]]:
    yield from trees.iter_name_tree_items(
        node,
        resolve,
        decode_name,
        max_depth=MAX_TREE_DEPTH,
        on_malformed=skipping_null_nodes(on_malformed),
    )
