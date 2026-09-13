# SPDX-License-Identifier: AGPL-3.0-only
"""Traversal identity remains correct with callbacks which do not cache dictionaries."""

import pytest

from core_pdf_spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf_spec.s_07_syntax.trees import iter_number_tree_items
from core_pdf_spec.types import PdfReference


def test_tree_retains_visited_nodes_with_non_caching_resolution() -> None:
    root_calls = 0

    def resolve(value: object) -> object:
        nonlocal root_calls
        if not isinstance(value, PdfReference):
            return value
        if value.object_number == 1:
            root_calls += 1
            return {"Kids": [PdfReference(number) for number in range(2, 42)]}
        return {"Nums": [value.object_number, "value"]}

    assert list(iter_number_tree_items(PdfReference(1), resolve)) == [
        (number, "value") for number in range(2, 42)
    ]
    assert root_calls == 1


def test_tree_detects_repeated_references_with_fresh_dictionaries() -> None:
    calls = 0

    def resolve(value: object) -> object:
        nonlocal calls
        if isinstance(value, PdfReference):
            calls += 1
            if calls > 2:
                raise AssertionError("reference cycle was not detected")
            return {"Nums": [0, "value"], "Kids": [PdfReference(1)]}
        return value

    items = iter_number_tree_items(PdfReference(1), resolve)
    assert next(items) == (0, "value")
    with pytest.raises(ValueError, match="cycle"):
        next(items)


def test_null_root_is_resolved_once_and_null_children_remain_invalid() -> None:
    calls: list[object] = []

    def resolve(value: object) -> object:
        calls.append(value)
        return None if isinstance(value, PdfReference) else value

    reference = PdfReference(1)
    assert list(iter_number_tree_items(reference, resolve)) == []
    assert calls == [reference]
    with pytest.raises(ValueError, match="tree node"):
        list(iter_number_tree_items({"Kids": [reference]}, resolve))


def test_inheritance_retains_fresh_parent_dictionaries() -> None:
    def resolve(value: object) -> object:
        if not isinstance(value, PdfReference):
            return value
        number = value.object_number
        return {"Parent": PdfReference(number + 1)} if number < 40 else {"Rotate": 90}

    assert collect_inherited_values({"Parent": PdfReference(1)}, ("Rotate",), resolve) == {
        "Rotate": 90
    }


def test_inheritance_rejects_repeated_parent_references_with_fresh_dictionaries() -> None:
    calls = 0

    def resolve(value: object) -> object:
        nonlocal calls
        if isinstance(value, PdfReference):
            calls += 1
            if calls > 2:
                raise AssertionError("reference cycle was not detected")
            return {"Parent": PdfReference(1)}
        return value

    with pytest.raises(ValueError, match="cycle"):
        collect_inherited_values({"Parent": PdfReference(1)}, ("Rotate",), resolve)
