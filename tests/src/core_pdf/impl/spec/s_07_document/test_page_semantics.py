# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl._impl.document.page_labels import format_page_label as reader_page_label
from core_pdf.impl._impl.document.page_tree import (
    collect_inherited_values as reader_inherited_values,
)
from core_pdf.impl.spec.s_07_document.document_labels import format_page_label
from core_pdf.impl.spec.s_07_document.page import PAGE_INHERITED_KEYS, iter_page_nodes, page_clip
from core_pdf.impl.spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfName


def test_page_inheritance_contains_only_the_pdf_inheritable_entries() -> None:
    leaf: PdfDict = {"Type": PdfName.of("Page")}
    tree: PdfDict = {
        "Type": PdfName.of("Pages"),
        "Kids": [leaf],
        "MediaBox": [0, 0, 100, 100],
        "Annots": [1],
    }
    (node,) = iter_page_nodes(tree, lambda value: value)
    assert node.dictionary is leaf
    assert "MediaBox" in node.inherited_values
    assert "Annots" not in node.inherited_values
    assert PAGE_INHERITED_KEYS == ("MediaBox", "CropBox", "Rotate", "Resources")


def test_disjoint_crop_box_is_an_empty_spec_intersection() -> None:
    clip = page_clip((0, 0, 100, 100), (200, 200, 300, 300))
    assert clip[0] == clip[2]
    assert clip[1] == clip[3]


def test_cyclic_inheritance_is_rejected_or_explicitly_recovered() -> None:
    node: PdfDict = {"MediaBox": [0, 0, 100, 100]}
    node["Parent"] = node
    with pytest.raises(ValueError, match="cycle"):
        collect_inherited_values(node, ("MediaBox",), lambda value: value)
    assert reader_inherited_values(node, ("MediaBox",), lambda value: value) == {
        "MediaBox": [0, 0, 100, 100]
    }


def test_invalid_label_start_is_reader_policy() -> None:
    label: PdfDict = {"S": PdfName.of("D"), "St": 0}
    with pytest.raises(ValueError, match="start"):
        format_page_label(label, 0, lambda value: value)
    assert reader_page_label(label, 0, lambda value: value) == "1"


def test_appearance_substate_inference_is_reader_policy() -> None:
    from core_pdf.impl._impl.capture.page import select_appearance_stream
    from core_pdf.impl.spec.s_07_document.annotation_appearance import normal_appearance_stream
    from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
    from core_pdf.impl.spec.s_07_syntax.stream import PdfStream

    stream = PdfStream(decoded_data=b"appearance")
    appearances = {"N": {"On": stream}}
    resolver = ObjectResolver(b"", {}, {})
    try:
        with pytest.raises(ValueError, match="state is required"):
            normal_appearance_stream(resolver, appearances, None)
        assert select_appearance_stream(resolver, appearances, None) is stream
        assert normal_appearance_stream(resolver, appearances, "Off") is None
    finally:
        resolver.close()
