import pytest

from core_pdf.impl.recovery_trees import iter_name_tree_items, iter_number_tree_items


def identity(value: object) -> object:
    return value


def name(value: object) -> str | None:
    return value if isinstance(value, str) else None


def test_reader_trees_read_a_null_kid_as_an_empty_subtree_even_strictly() -> None:
    numbers = {"Kids": [None, {"Nums": [1, "one"]}]}
    names = {"Kids": [{"Names": ["a", 1]}, None]}
    assert list(iter_number_tree_items(numbers, identity)) == [(1, "one")]
    assert list(iter_name_tree_items(names, identity, name)) == [("a", 1)]


def test_reader_trees_still_report_other_malformed_nodes() -> None:
    tree = {"Kids": [17, {"Nums": [1, "one"]}]}
    with pytest.raises(ValueError, match="invalid number tree node"):
        list(iter_number_tree_items(tree, identity))
    reports: list[str] = []
    assert list(iter_number_tree_items(tree, identity, on_malformed=reports.append)) == [(1, "one")]
    assert reports == ["invalid number tree node"]
