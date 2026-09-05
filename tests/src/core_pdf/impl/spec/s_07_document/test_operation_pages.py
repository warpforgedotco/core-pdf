# SPDX-License-Identifier: AGPL-3.0-only
"""Page snapshots are shared inside an operation and discarded between operations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.spec.s_07_document.document import internal_PageLookup, internal_PageNode
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_14_structure.tree import StructureElement
from core_pdf.impl.types import PdfReference
from tests.helpers.pdf_bytes import assemble_pdf


class CountingDocument(PdfDocument):
    page_walks = 0

    def internal_iter_page_nodes(self) -> Iterator[internal_PageNode]:
        self.page_walks += 1
        yield from super().internal_iter_page_nodes()


def navigation_pdf(*, recover: bool = False) -> bytes:
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /Names << /Dests 6 0 R >> "
            b"/Outlines 7 0 R /StructTreeRoot 11 0 R >>",
            b"<< /Type /Pages /Count 3 /MediaBox [10 20 210 320] /Rotate 90 /Kids "
            + (b"[]" if recover else b"[3 0 R 4 0 R 5 0 R]")
            + b" >>",
            b"<< /Type /Page /Parent 2 0 R /StructParents 0 >>",
            b"<< /Type /Page /Parent 2 0 R /StructParents 1 >>",
            b"<< /Type /Page /Parent 2 0 R /StructParents 2 >>",
            b"<< /Names [(alias) (first) (broken) [null /Fit] "
            b"(cycle-a) (cycle-b) (cycle-b) (cycle-a) "
            b"(first) [3 0 R /Fit] (third) [5 0 R /Fit]] >>",
            b"<< /Type /Outlines /First 8 0 R >>",
            b"<< /Title (First) /Dest (alias) /First 10 0 R /Next 9 0 R >>",
            b"<< /Title (Third) /A << /S /GoTo /D (third) >> >>",
            b"<< /Title (Second) /Dest [4 0 R /Fit] >>",
            b"<< /Type /StructTreeRoot /K [12 0 R 13 0 R] "
            b"/ParentTree << /Nums [0 [12 0 R] 1 [14 0 R] 2 [13 0 R]] >> >>",
            b"<< /S /P /P 11 0 R /Pg 3 0 R /K [0 "
            b"<< /Type /MCR /Pg 4 0 R /MCID 1 >> "
            b"<< /Type /OBJR /Pg 5 0 R /Obj << /Type /Annot >> >> 14 0 R] >>",
            b"<< /S /Sect /P 11 0 R /Pg 5 0 R /K 2 >>",
            b"<< /S /Span /P 12 0 R /Pg 4 0 R /K 3 >>",
        ]
    )


@pytest.mark.parametrize("recover", [False, True])
def test_navigation_reuses_pages_with_nested_outlines_and_damaged_names(recover: bool) -> None:
    with CountingDocument(navigation_pdf(recover=recover)) as document:
        names = document.named_destinations()
        assert {name: dest.page_index for name, dest in names.items()} == {
            "alias": 0,
            "first": 0,
            "broken": None,
            "cycle-a": None,
            "cycle-b": None,
            "third": 2,
        }
        assert document.page_walks == 1

        outlines = document.iter_outlines()
        assert [(item.title, item.level, item.page_index) for item in outlines] == [
            ("First", 0, 0),
            ("Second", 1, 1),
            ("Third", 0, 2),
        ]
        assert document.page_walks == 2
        assert document.page_tree_was_recovered is recover


def test_new_navigation_operations_observe_changed_page_order() -> None:
    with CountingDocument(navigation_pdf()) as document:
        assert document.named_destinations()["first"].page_index == 0
        root = document.resolve(PdfReference(2))
        assert isinstance(root, dict)
        root = cast(PdfDict, root)
        root["Kids"] = [PdfReference(5), PdfReference(4), PdfReference(3)]

        assert document.named_destinations()["first"].page_index == 2
        assert [item.page_index for item in document.iter_outlines()] == [2, 1, 0]
        assert document.page_index_for(document.resolve(PdfReference(3))) == 2
        assert document.page_walks == 4


@pytest.mark.parametrize("recover", [False, True])
def test_structure_shares_pages_across_children_parent_entries_and_slices(recover: bool) -> None:
    with CountingDocument(navigation_pdf(recover=recover)) as document:
        tree = document.structure
        assert tree is not None
        assert document.page_walks == 0
        elements = list(tree.find_all())
        assert [element.type for element in elements] == ["P", "Span", "Sect"]
        assert [element.page_index for element in elements] == [0, 1, 2]
        assert [child.page_index for child in elements[0]] == [0, 1, 2, 1]
        first_page = elements[0].page
        assert first_page is not None
        assert first_page.page_dict is document.resolve(PdfReference(3))
        assert first_page.media_box == (10.0, 20.0, 210.0, 320.0)
        assert first_page.rotation == 90
        page_structure = tree.page_structure(first_page)
        parent = page_structure[:][0]
        assert isinstance(parent, StructureElement)
        assert parent.page is first_page
        assert [child.page_index for child in parent] == [0, 1, 2, 1]
        ancestor = elements[1].parent
        assert isinstance(ancestor, StructureElement)
        assert ancestor.page is first_page
        assert document.page_walks == 1

        rebuilt = document.structure
        assert rebuilt is not None
        assert rebuilt is not tree
        assert [element.page_index for element in rebuilt.find_all()] == [0, 1, 2]
        assert document.page_walks == 2


def test_page_index_snapshot_preserves_fallback_matching_and_first_identity() -> None:
    with CountingDocument(navigation_pdf()) as document:
        root = document.resolve(PdfReference(2))
        first = document.resolve(PdfReference(3))
        third = document.resolve(PdfReference(5))
        assert isinstance(root, dict)
        assert isinstance(first, dict)
        assert isinstance(third, dict)
        root = cast(PdfDict, root)
        first = cast(PdfDict, first)
        third = cast(PdfDict, third)
        root["Kids"] = [PdfReference(3), PdfReference(4), PdfReference(5), PdfReference(3)]
        third["Contents"] = PdfReference(99)
        first.pop("StructParents")
        lookup = internal_PageLookup(document)

        assert lookup.page_index_for(first) == 0
        assert lookup.page_index_for({"StructParents": 1}) == 1
        assert lookup.page_index_for(dict(first)) == 0
        assert lookup.page_index_for({"Contents": PdfReference(99), "Extra": 1}) == 2
        assert lookup.page_index_for({"Unrelated": True}) is None
        assert lookup.page_index_for(None) is None
        assert document.page_walks == 1


def test_structure_root_parents_share_the_current_operation_snapshot() -> None:
    with CountingDocument(navigation_pdf()) as document:
        tree = document.structure
        assert tree is not None
        elements = list(tree.find_all())
        first = elements[0]
        assert first.page_index == 0
        before = first.parent
        assert before is not None

        root = document.resolve(PdfReference(2))
        assert isinstance(root, dict)
        root = cast(PdfDict, root)
        root["Kids"] = [PdfReference(5), PdfReference(4), PdfReference(3)]
        after = elements[2].parent
        assert after is not None

        assert [element.page_index for element in before.find_all()] == [0, 1, 2]
        assert [element.page_index for element in after.find_all()] == [0, 1, 2]
        assert document.page_walks == 1

        # A directly constructed element and a newly requested tree each begin
        # their own operation and therefore observe the changed page sequence.
        standalone = StructureElement(document, first.props)
        standalone_root = standalone.parent
        assert standalone_root is not None
        assert [element.page_index for element in standalone_root.find_all()] == [2, 1, 0]
        rebuilt = document.structure
        assert rebuilt is not None
        assert [element.page_index for element in rebuilt.find_all()] == [2, 1, 0]
        assert document.page_walks == 3
