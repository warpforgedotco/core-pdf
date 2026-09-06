# SPDX-License-Identifier: AGPL-3.0-only
"""Tolerant name and number tree readers using the shared syntax traversal."""

from __future__ import annotations

from collections.abc import Iterator

from core_pdf.impl.spec.s_07_syntax import trees as syntax_trees
from core_pdf.impl.spec.s_07_syntax.trees import NameDecodeFn, NumberDecodeFn, ResolveFn


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
    yield from syntax_trees.iter_number_tree_items(
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
    yield from syntax_trees.iter_name_tree_items(
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
