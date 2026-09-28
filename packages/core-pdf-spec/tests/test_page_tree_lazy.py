# SPDX-License-Identifier: AGPL-3.0-only

from core_pdf_spec.s_07_document.page import LazyInheritedValues, iter_page_nodes


def test_lazy_inherited_values_match_eager_ones_and_resolve_once() -> None:
    resources = {"Font": {}}
    parent = {"Type": "Pages", "MediaBox": [0, 0, 10, 10], "Resources": resources}
    page = {"Type": "Page", "Rotate": 90}
    parent["Kids"] = [page]

    def resolve(value: object) -> object:
        return value

    def node_type(node: dict) -> str | None:
        return node.get("Type")

    eager = [
        node.inherited_values for node in iter_page_nodes(parent, resolve, node_type=node_type)
    ]
    lazy_nodes = list(iter_page_nodes(parent, resolve, node_type=node_type, lazy_inherited=True))
    lazy = lazy_nodes[0].inherited_values
    assert isinstance(lazy, LazyInheritedValues)
    assert lazy.resolved_values is None
    assert dict(lazy) == eager[0]
    assert sorted(lazy.keys()) == sorted(eager[0].keys())
    assert list(lazy.values()) == list(eager[0].values())
    first = lazy.resolved_values
    assert first is not None
    assert dict(lazy) == eager[0]
    assert lazy.resolved_values is first
