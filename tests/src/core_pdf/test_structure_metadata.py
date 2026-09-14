"""Logical-structure metadata projected through the real document resolver."""

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf.impl._impl.document.structure import (
    PageStructure,
    StructureContentItem,
    StructureContentObject,
    StructureElement,
    StructureTree,
    literal_name,
    structure_key_name,
)
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_syntax.types import PdfDict


@pytest.fixture
def document(text_pdf_bytes):
    with PdfDocument(text_pdf_bytes) as document:
        yield document


@pytest.mark.parametrize("page_index", [None, 0, 2])
def test_content_records_preserve_valid_page_and_payload_identity(page_index):
    stream: PdfDict = {"payload": 1}
    item = StructureContentItem(page_index, 7, stream)
    obj = StructureContentObject(page_index, stream)
    assert (item.page_index, item.mcid, item.stream) == (page_index, 7, stream)
    assert item.stream is stream
    assert obj.page_index == page_index
    assert obj.props is stream


@pytest.mark.parametrize("page_index", [-1, True, 1.5, "0"])
@pytest.mark.parametrize("object_item", [False, True])
def test_content_records_reject_invalid_page_indexes(page_index, object_item):
    with pytest.raises(ValueError, match="page index"):
        if object_item:
            StructureContentObject(page_index, {})
        else:
            StructureContentItem(page_index, 0)


@pytest.mark.parametrize("props", [[]])
def test_object_content_requires_dictionary(props):
    with pytest.raises(ValueError, match="props"):
        StructureContentObject(None, props)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (PdfReference(3, 0), None), (PdfName(b"Figure"), "Figure")],
)
def test_literal_names_do_not_resolve_references(value, expected):
    assert literal_name(value) == expected
    assert structure_key_name(42) == "42"


def test_element_metadata_is_decoded_once_and_cached(document):
    props: PdfDict = {
        "S": PdfName(b"P"),
        "T": PdfString(b"Title"),
        "Lang": PdfString(b"en"),
        "Alt": PdfString(b"Description"),
        "ActualText": PdfString(b"Replacement"),
    }
    element = StructureElement(document, props)
    for attribute, key, expected in [
        ("type", "S", "P"),
        ("title", "T", "Title"),
        ("language", "Lang", "en"),
        ("alternate_description", "Alt", "Description"),
        ("actual_text", "ActualText", "Replacement"),
    ]:
        assert getattr(element, attribute) == expected
        props.pop(key)
        assert getattr(element, attribute) == expected
    assert element.page_index is None
    assert element.page is None


def test_element_resolves_page_and_rejects_foreign_page_dictionary(document):
    element = StructureElement(document, {"Pg": PdfReference(3, 0)})
    assert element.page_index == 0
    page = element.page
    assert page is not None
    assert page.page_dict is document.pages[0].page_dict
    assert page.document is document
    invalid = StructureElement(document, {"Pg": {"Type": PdfName(b"Page")}})
    with pytest.raises(ValueError, match="page reference"):
        _ = invalid.page_index


def test_attributes_normalize_keys_and_cache_result(document):
    element = StructureElement(document, {"A": {PdfName(b"O"): PdfName(b"Layout"), 42: 3}})
    assert element.attributes == {"O": PdfName(b"Layout"), "42": 3}
    assert element.attributes is element.attributes


@pytest.mark.parametrize("value", [None, 7, []])
def test_missing_or_empty_attributes_return_none(document, value):
    element = StructureElement(document, {"A": value})
    assert element.attributes is None
    assert element.attributes is None


def test_attribute_revisions_choose_latest_and_keep_first_on_tie(document):
    element = StructureElement(
        document, {"A": [{"value": 1}, 2, {"value": 2}, 1, {"value": 3}, 3, {"value": 4}, 3]}
    )
    assert element.attributes == {"value": 3}


@pytest.mark.parametrize("value", [[{}], [7, 0], [{}, {}]])
def test_malformed_attribute_arrays_fail_clearly(document, value):
    element = StructureElement(document, {"A": value})
    with pytest.raises(ValueError):
        _ = element.attributes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (PdfName(b"One"), "One"),
        ([PdfName(b"One")], "One"),
        ([PdfName(b"One"), 0, PdfName(b"Two"), 1], "Two"),
    ],
)
def test_class_names_handle_single_names_and_revision_arrays(document, value, expected):
    element = StructureElement(document, {"C": value})
    assert element.class_name == expected
    assert element.class_name == expected


@pytest.mark.parametrize("value", [PdfReference(999, 0), [None]])
def test_invalid_class_entries_are_rejected_or_resolve_to_absent(document, value):
    element = StructureElement(document, {"C": value})
    if isinstance(value, list):
        with pytest.raises(ValueError, match="class name"):
            _ = element.class_name
    else:
        assert element.class_name is None


def test_parent_projection_is_cached_and_missing_parent_is_none(document):
    parent = {"S": PdfName(b"Sect")}
    element = StructureElement(document, {"P": parent})
    assert isinstance(element.parent, StructureElement)
    assert element.parent.props is parent
    assert element.parent is element.parent
    absent = StructureElement(document, {})
    assert absent.parent is None
    assert absent.parent is None
    invalid = StructureElement(document, {"P": 7})
    with pytest.raises(ValueError, match="parent entry"):
        _ = invalid.parent


def test_role_map_is_normalized_and_cached(document):
    tree = StructureTree(document, {"RoleMap": {"Custom": PdfName(b"P")}})
    assert tree.type == tree.role == "StructTreeRoot"
    assert tree.role_map == {"Custom": "P"}
    assert tree.role_map is tree.role_map
    empty = StructureTree(document, {})
    assert empty.role_map == {}
    assert empty.role_map is empty.role_map
    with pytest.raises(ValueError, match="role map"):
        _ = StructureTree(document, {"RoleMap": 7}).role_map


def test_page_structure_preserves_shared_parent_identity_and_sequence_semantics(document):
    parent = {"S": PdfName(b"P")}
    existing = StructureElement(document, {"S": PdfName(b"Span")})
    view = PageStructure(document.pages[0], [parent, None, parent, existing])
    assert len(view) == 4
    assert view[0] is view[2]
    assert view[1] is None
    assert view[-1] is existing
    assert len(view[1:3]) == 2
    assert view[1:3][0] is None
    assert list(view.find_all()) == [view[0], existing]
    assert view.find("P") is view[0]
    assert view.find("Figure") is None
    assert len(PageStructure(document.pages[0], None)) == 0
    with pytest.raises(IndexError):
        _ = view[4]
    with pytest.raises(ValueError, match="parents"):
        PageStructure(document.pages[0], 5)
    with pytest.raises(ValueError, match="parent entry"):
        _ = PageStructure(document.pages[0], [5])[0]
