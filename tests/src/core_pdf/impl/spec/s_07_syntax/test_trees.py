# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_07_syntax.trees import iter_number_tree_items


def identity(value: object) -> object:
    return value


def test_number_tree_walks_nested_kids_in_document_order() -> None:
    tree = {
        "Kids": [
            {"Nums": [0, "zero", 2, "two"]},
            {"Nums": [5, "five"]},
        ]
    }

    assert list(iter_number_tree_items(tree, identity)) == [
        (0, "zero"),
        (2, "two"),
        (5, "five"),
    ]


def test_number_tree_cycle_is_rejected_or_skipped_during_recovery() -> None:
    tree: dict[str, object] = {"Nums": [0, "zero"]}
    tree["Kids"] = [tree]

    with pytest.raises(ValueError, match="number tree cycle detected"):
        list(iter_number_tree_items(tree, identity))

    assert list(iter_number_tree_items(tree, identity, recover=True)) == [(0, "zero")]


def test_number_tree_entry_recovery_does_not_hide_invalid_node_shape() -> None:
    with pytest.raises(ValueError, match="invalid parent tree Kids array"):
        list(
            iter_number_tree_items(
                {"Nums": [0, "zero", "trailing"], "Kids": "invalid"},
                identity,
                recover_entries=True,
                tree_name="parent",
            )
        )
