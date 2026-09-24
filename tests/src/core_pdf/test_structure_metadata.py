import pytest

from core_pdf.impl.document import structure
from core_pdf.impl.document.document import PageLookup, PdfDocument
from core_pdf.impl.document.structure import (
    PageStructure,
    StructureContentItem,
    StructureContentObject,
    StructureElement,
    StructureTree,
    structure_key_name,
)
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import key_for
from core_pdf_spec.standards import PdfVersion, SemanticContext


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
    assert recover_pdf_name(value) == expected
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
    element = StructureElement(document, {"A": {PdfName(b"O"): PdfName(b"Layout"), b"Width": 3}})
    assert element.attributes == {"O": PdfName(b"Layout"), "Width": 3}
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
    parent: PdfDict = {"S": PdfName(b"Sect")}
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


def test_parent_tree_preserves_values_and_caches_number_tree_results(document):
    parents = [{"S": PdfName(b"P")}, None]
    tree = StructureTree(document, {"ParentTree": {"Kids": [{"Nums": [4, parents]}]}})
    assert tree.parent_tree == {4: parents}
    assert tree.parent_tree[4] is parents
    assert tree.parent_tree is tree.parent_tree
    page = document.pages[0]
    page.page_dict["StructParents"] = 4
    view = tree.page_structure(page)
    assert len(view) == 2
    element = view[0]
    assert element is not None
    assert element.role == "P"
    assert view[1] is None


def test_absent_parent_tree_is_cached_and_invalid_tree_rejected(document):
    tree = StructureTree(document, {})
    assert tree.parent_tree == {}
    assert tree.parent_tree is tree.parent_tree
    with pytest.raises(ValueError, match="parent tree dictionary"):
        _ = StructureTree(document, {"ParentTree": 5}).parent_tree


@pytest.mark.parametrize(
    ("key", "parents", "message"),
    [(None, [], "StructParents"), (7, [], "parent tree entry"), (4, {}, "structure parents")],
)
def test_page_parent_tree_slice_rejects_invalid_keys_and_values(document, key, parents, message):
    tree = StructureTree(document, {"ParentTree": {"Nums": [4, parents]}})
    page = document.pages[0]
    page.page_dict["StructParents"] = key
    with pytest.raises(ValueError, match=message):
        tree.page_structure(page)


def test_structure_kids_keep_nested_order_page_and_stream_reference(document):
    page = document.pages[0]
    stream = PdfReference(5, 0)
    kids = [
        None,
        2,
        [3, {"Type": PdfName(b"MCR"), "MCID": 4, "Pg": PdfReference(3, 0), "Stm": stream}],
        {"S": PdfName(b"P")},
    ]
    result = list(structure.make_kids(kids, page, document))
    for item, mcid in zip(result[:3], (2, 3, 4), strict=True):
        assert isinstance(item, StructureContentItem)
        assert item.mcid == mcid
    assert all(item.page_index == 0 for item in result[:3])
    marked = result[2]
    assert isinstance(marked, StructureContentItem)
    assert marked.stream is stream
    assert isinstance(result[3], StructureElement)
    assert result[3].role == "P"
    assert next(structure.make_kids(8, None, document)).page_index is None


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize(
    "kid",
    [
        True,
        -1,
        1.5,
        {"Type": PdfName(b"MCR")},
        {"Type": PdfName(b"OBJR")},
        {"Type": PdfName(b"OBJR"), "Obj": 5},
    ],
)
def test_invalid_children_raise_in_strict_mode_and_skip_in_recovery(document, recovery, kid):
    document.xref_was_recovered = recovery
    if recovery:
        result = list(structure.make_kids([kid, 7], None, document))
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, StructureContentItem)
        assert item.mcid == 7
    else:
        with pytest.raises(ValueError):
            list(structure.make_kids(kid, None, document))


@pytest.mark.parametrize("recovery", [False, True])
def test_structure_depth_limit_applies_to_nested_children(document, recovery):
    document.xref_was_recovered = recovery
    if recovery:
        assert list(structure.make_kids(0, None, document, structure.MAX_STRUCTURE_DEPTH + 1)) == []
    else:
        with pytest.raises(ValueError, match="depth"):
            list(structure.make_kids(0, None, document, structure.MAX_STRUCTURE_DEPTH + 1))


@pytest.mark.parametrize("indirect", [False, True])
def test_object_reference_children_resolve_annotation_dictionary(document, indirect):
    annotation: PdfDict = {"Type": PdfName(b"Annot"), "Subtype": PdfName(b"Text")}
    reference = PdfReference(99, 0)
    document.resolver.objects[key_for(99, 0)] = annotation
    kid = {"Type": PdfName(b"OBJR"), "Obj": reference if indirect else annotation}
    result = list(structure.make_kids(kid, document.pages[0], document))
    assert len(result) == 1
    assert isinstance(result[0], StructureContentObject)
    assert result[0].props is annotation
    assert result[0].page_index == 0


def test_indirect_children_and_page_parents_share_resolved_dictionary(document):
    parent: PdfDict = {"S": PdfName(b"P")}
    reference = PdfReference(99, 0)
    document.resolver.objects[key_for(99, 0)] = parent
    child = next(structure.make_kids(reference, None, document))
    assert isinstance(child, StructureElement)
    assert child.props is parent
    view = PageStructure(document.pages[0], [reference, reference, parent])
    assert view[0] is view[1] is view[2]
    document.resolver.objects[key_for(100, 0)] = 5
    with pytest.raises(ValueError, match="parent entry"):
        _ = PageStructure(document.pages[0], [PdfReference(100, 0)])[0]


def test_tree_search_is_depth_first_and_ignores_content_items(document):
    tree = StructureTree(
        document,
        {"K": [{"S": PdfName(b"Sect"), "K": [0, {"S": PdfName(b"P")}]}, {"S": PdfName(b"Figure")}]},
    )
    found = list(tree.find_all())
    assert [item.role for item in found] == ["Sect", "P", "Figure"]
    assert tree.find("P") is found[1]
    assert tree.find(lambda item: item.role == "Figure") is found[2]
    assert tree.find("missing") is None
    assert found[0].find("P") is found[1]
    assert found[0].find("missing") is None
    assert list(tree) == list(tree)


def test_standard_role_exposes_namespace_and_cached_resolution(document):
    element = StructureElement(document, {"S": PdfName(b"P")})
    assert element.role == "P"
    assert element.role_namespace == "http://iso.org/pdf/ssn"
    assert element.role_error is None
    assert element.role_resolution is element.role_resolution


@pytest.mark.parametrize("version", [None, PdfVersion(9, 9)])
def test_unknown_document_versions_still_resolve_identified_structure_roles(document, version):
    document.resolver.semantic_context = SemanticContext(version)
    element = StructureElement(document, {"S": PdfName(b"P")})
    assert element.role == "P"
    assert element.role_error is None


def test_malformed_namespace_retains_type_and_cached_diagnostic(document):
    element = StructureElement(document, {"S": PdfName(b"Custom"), "NS": 7})
    assert element.role == "Custom"
    assert element.role_namespace is None
    assert element.role_error
    diagnostic = element.role_error
    element.props.pop("NS")
    assert element.role_resolution is None
    assert element.role_error == diagnostic


def test_stream_attribute_is_rejected_by_dictionary_projection(document):
    element = StructureElement(document, {"A": [PdfStream({}, b""), 0]})
    with pytest.raises(ValueError, match="attribute entry"):
        _ = element.attributes


@pytest.mark.parametrize("value", [42, [None]])
def test_invalid_class_names_keep_failing_on_repeated_access(document, value):
    element = StructureElement(document, {"C": value})
    for _ in range(2):
        with pytest.raises(ValueError, match="class name"):
            _ = element.class_name


@pytest.mark.parametrize("with_lookup", [False, True])
def test_root_parent_uses_document_tree_and_retains_lookup(document, with_lookup):
    root: PdfDict = {"Type": PdfName(b"StructTreeRoot")}
    document.catalog()["StructTreeRoot"] = root
    lookup = PageLookup(document) if with_lookup else None
    element = StructureElement(document, {"P": root}, page_lookup=lookup)
    parent = element.parent
    assert isinstance(parent, StructureTree)
    assert parent.props is root
    if lookup is not None:
        assert parent.page_lookup is lookup
    assert element.parent is parent


def test_page_search_can_find_ancestor_instead_of_immediate_parent(document):
    ancestor: PdfDict = {"S": PdfName(b"Sect")}
    child: PdfDict = {"S": PdfName(b"P"), "P": ancestor}
    view = PageStructure(document.pages[0], [child, None])
    result = view.find("Sect")
    assert result is not None
    assert result.props is ancestor
    assert view.find("Figure") is None


def test_child_page_lookup_rejects_foreign_reference_and_allows_absent_page(document):
    assert structure.get_kid_page_index(document, None, {}) is None
    lookup = PageLookup(document)
    assert structure.get_kid_page_index(document, None, {"Pg": PdfReference(3, 0)}, lookup) == 0
    with pytest.raises(ValueError, match="page reference"):
        structure.get_kid_page_index(document, None, {"Pg": 7})


def test_element_page_reuses_explicit_lookup_page_wrapper(document):
    lookup = PageLookup(document)
    element = StructureElement(document, {"Pg": PdfReference(3, 0)}, page_lookup=lookup)
    assert element.page is lookup.pages[0]
    assert element.page_index == 0
