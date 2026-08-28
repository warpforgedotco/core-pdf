# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

from core_pdf.impl.spec.s_14_structure.tree import (
    StructureElement,
    StructureTree,
    find_all,
)


class Resolver:
    @staticmethod
    def resolve(value: object) -> object:
        return value

    @staticmethod
    def resolve_int(value: object) -> int | None:
        if type(value) is int:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None


class Document:
    resolver = Resolver()
    xref_was_recovered = False
    page_tree_was_recovered = False


def test_parent_tree_uses_shared_number_tree_walker() -> None:
    first_parent = ["first"]
    second_parent = ["second"]
    document = cast(Any, Document())
    structure = StructureTree(
        document,
        {
            "ParentTree": {
                "Kids": [
                    {"Nums": ["1", first_parent]},
                    {"Nums": [2, second_parent]},
                ]
            }
        },
    )

    assert structure.parent_tree == {1: first_parent, 2: second_parent}


def test_find_all_leaves_the_caller_owned_list_intact() -> None:
    first = StructureElement(cast(Any, Document()), cast(Any, {}))
    second = StructureElement(cast(Any, Document()), cast(Any, {}))
    elements = [first, second]

    assert list(find_all(elements)) == [first, second]
    assert elements == [first, second]


def test_structure_element_hash_stays_stable_when_properties_are_cached() -> None:
    props: dict[str, object] = {}
    element = StructureElement(cast(Any, Document()), cast(Any, props))
    initial_hash = hash(element)

    props["cached"] = "value"

    assert hash(element) == initial_hash
