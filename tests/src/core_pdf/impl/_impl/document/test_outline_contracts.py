"""Outline recovery retains valid siblings without hiding strict failures."""

import pytest

from core_pdf.impl._impl.document.document import PdfDocument
from core_pdf.impl.types import PdfName


@pytest.mark.parametrize("action", [False, True])
def test_outline_children_and_siblings_preserve_order(text_pdf_bytes, action):
    with PdfDocument(text_pdf_bytes) as document:
        destination = [document.build_page_dicts()[0], PdfName(b"Fit")]
        entry = {"Title": b"first", "Count": -1}
        if action:
            entry["A"] = {"S": PdfName(b"GoTo"), "D": destination}
        else:
            entry["Dest"] = destination
        entry["First"] = {"Title": b"child"}
        entry["Next"] = {"Title": b"sibling", "A": {"S": PdfName(b"URI"), "D": destination}}
        document.catalog()["Outlines"] = {"First": entry}
        actual = document.iter_outlines()
        assert [(x.title, x.level, x.page_index, x.count) for x in actual] == [
            ("first", 0, 0, -1),
            ("child", 1, None, 0),
            ("sibling", 0, None, 0),
        ]
        assert actual[0].dest is destination


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    ("invalid", "message", "retained"),
    [
        ({"First": 1}, "invalid outline child", ["first", "last"]),
        ({"Dest": []}, "invalid destination array", ["last"]),
        ({"Count": b"invalid"}, "invalid outline count", ["first", "last"]),
        ({"Next": 1}, "invalid outline item", ["first"]),
    ],
)
def test_outline_recovery_retains_valid_siblings(
    text_pdf_bytes, recover, invalid, message, retained
):
    with PdfDocument(text_pdf_bytes) as document:
        document.xref_was_recovered = recover
        first = {"Title": b"first", "Next": {"Title": b"last"}, **invalid}
        document.catalog()["Outlines"] = {"First": first}
        if recover:
            assert [x.title for x in document.iter_outlines()] == retained
        else:
            with pytest.raises(ValueError, match=message):
                document.iter_outlines()


@pytest.mark.parametrize("recover", [False, True])
def test_outline_sibling_cycle_is_bounded(text_pdf_bytes, recover):
    with PdfDocument(text_pdf_bytes) as document:
        document.xref_was_recovered = recover
        first = {"Title": b"once"}
        first["Next"] = first
        if recover:
            assert [x.title for x in document.walk_outlines(first, 0)] == ["once"]
        else:
            with pytest.raises(ValueError, match="cycle detected"):
                document.walk_outlines(first, 0)


@pytest.mark.parametrize("value", [None, 1, b"invalid", []])
@pytest.mark.parametrize("recover", [False, True])
def test_non_dictionary_outline_entries(text_pdf_bytes, value, recover):
    with PdfDocument(text_pdf_bytes) as document:
        document.xref_was_recovered = recover
        if recover:
            assert document.walk_outlines(value, 0) == []
        else:
            with pytest.raises(ValueError, match="invalid outline item"):
                document.walk_outlines(value, 0)


@pytest.mark.parametrize("value", [None, {}, {"First": None}])
def test_empty_outline_catalog_entries(text_pdf_bytes, value):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["Outlines"] = value
        assert document.iter_outlines() == []


def test_outline_catalog_type_and_depth_are_validated(text_pdf_bytes):
    with PdfDocument(text_pdf_bytes) as document:
        document.catalog()["Outlines"] = 1
        with pytest.raises(ValueError, match="invalid Outlines dictionary"):
            document.iter_outlines()
        with pytest.raises(ValueError, match="invalid outline depth"):
            document.walk_outlines({}, 201)
