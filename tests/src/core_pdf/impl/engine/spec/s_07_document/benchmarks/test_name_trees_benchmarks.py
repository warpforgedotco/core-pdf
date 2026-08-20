# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_document.name_trees import (
    iter_name_tree_items,
    iter_number_tree_items,
)

pytestmark = pytest.mark.benchmark_high_impact

LEAF_FANOUT = 8
INTERMEDIATE_FANOUT = 8


def identity_resolve(value: object) -> object:
    return value


def decode_name(value: object) -> str | None:
    return value if isinstance(value, str) else None


def build_number_tree(intermediate_count: int, leaf_count: int) -> dict[str, object]:
    kids = []
    for group in range(intermediate_count):
        nums: list[object] = []
        base = group * leaf_count
        for offset in range(leaf_count):
            nums.append(base + offset)
            nums.append({"PageOffset": base + offset})
        kids.append({"Nums": nums})
    return {"Kids": kids}


def build_name_tree(intermediate_count: int, leaf_count: int) -> dict[str, object]:
    kids = []
    for group in range(intermediate_count):
        names: list[object] = []
        for offset in range(leaf_count):
            names.append(f"Dest-{group}-{offset}")
            names.append({"D": [f"{group}-{offset} 0 R", "Fit"]})
        kids.append({"Names": names})
    return {"Kids": kids}


NUMBER_TREE = build_number_tree(INTERMEDIATE_FANOUT, LEAF_FANOUT * 8)
NAME_TREE = build_name_tree(INTERMEDIATE_FANOUT, LEAF_FANOUT * 8)


def test_iter_number_tree_items_benchmark(benchmark) -> None:
    result = benchmark(lambda: list(iter_number_tree_items(NUMBER_TREE, identity_resolve)))
    assert len(result) == INTERMEDIATE_FANOUT * LEAF_FANOUT * 8


def test_iter_name_tree_items_benchmark(benchmark) -> None:
    result = benchmark(lambda: list(iter_name_tree_items(NAME_TREE, identity_resolve, decode_name)))
    assert len(result) == INTERMEDIATE_FANOUT * LEAF_FANOUT * 8
