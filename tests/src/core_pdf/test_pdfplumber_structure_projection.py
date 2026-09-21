from io import BytesIO
from types import SimpleNamespace

import pytest

from core_pdf.api.compat import pdfplumber as compat


class Element:
    def __init__(self, role="P", page_index=None, children: tuple[object, ...] = ()):
        self.role = role
        self.type = "StructElem"
        self.title = "section"
        self.actual_text = "replacement"
        self.alternate_description = "description"
        self.page_index = page_index
        self.children = children

    def __iter__(self):
        return iter(self.children)


@pytest.mark.parametrize("owner", [0, 1, None])
def test_structure_projection_filters_root_page_ownership_and_preserves_fields(
    text_pdf_bytes, monkeypatch, owner
):
    child = Element("Span", 0)
    root = Element("P", owner, (child, 7, SimpleNamespace(type="MCR")))
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(type(pdf._document), "structure", property(lambda self: (root, 42)))
        result = pdf.pages[0].structure_tree
        if owner == 1:
            assert result == []
        else:
            assert len(result) == 1
            node = result[0]
            assert node == {
                "type": "StructElem",
                "role": "P",
                "title": "section",
                "page_number": 1 if owner == 0 else None,
                "actual_text": "replacement",
                "alt": "description",
                "children": [
                    {
                        "type": "StructElem",
                        "role": "Span",
                        "title": "section",
                        "page_number": 1,
                        "actual_text": "replacement",
                        "alt": "description",
                        "children": [],
                    }
                ],
            }
            assert pdf.structure_tree == result
        assert root.children[0] is child


@pytest.mark.parametrize("error_type", [KeyError, TypeError, ValueError])
@pytest.mark.parametrize("boundary", ["root", "children", "page-index"])
def test_structure_projection_recovers_only_the_malformed_boundary(
    text_pdf_bytes, monkeypatch, error_type, boundary
):
    class BrokenChildren(Element):
        def __iter__(self):
            raise error_type("broken children")

    class BrokenPage:
        role = "P"

        @property
        def page_index(self):
            raise error_type("broken page index")

        def __iter__(self):
            return iter(())

    tree = (
        BrokenChildren()
        if boundary == "root"
        else (BrokenChildren(),)
        if boundary == "children"
        else (BrokenPage(),)
    )
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(type(pdf._document), "structure", property(lambda self: tree))
        result = pdf.pages[0].structure_tree
        if boundary == "root":
            assert result == []
        else:
            assert len(result) == 1
            assert result[0]["children"] == []
            assert result[0]["page_number"] is None
            assert result[0]["role"] == "P"


@pytest.mark.parametrize("tree", [None, (), (None, 3, "ignored")])
def test_missing_or_nonstructural_tree_has_no_projected_nodes(text_pdf_bytes, monkeypatch, tree):
    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(type(pdf._document), "structure", property(lambda self: tree))
        assert pdf.structure_tree == []


def test_unexpected_structure_failure_is_not_silently_converted_to_an_empty_tree(
    text_pdf_bytes, monkeypatch
):
    class Broken:
        def __iter__(self):
            raise RuntimeError("unexpected failure")

    with compat.open(BytesIO(text_pdf_bytes)) as pdf:
        monkeypatch.setattr(type(pdf._document), "structure", property(lambda self: Broken()))
        with pytest.raises(RuntimeError, match="unexpected failure"):
            _ = pdf.pages[0].structure_tree
