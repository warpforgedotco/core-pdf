"""Literal structure attributes remain separate from reader projection choices."""

import pytest

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_14_structure.dictionaries import (
    attribute_entries,
    marked_content_id,
    parse_role_map,
)


def identity(value: object) -> object:
    return value


def integer(value: object) -> int | None:
    return value if type(value) is int else None


def test_attributes_keep_every_owner_revision_and_optional_revision() -> None:
    first = {"O": "Layout", "TextAlign": "Start"}
    second = {"O": "List", "ListNumbering": "Decimal"}
    stream = PdfStream({}, b"", decoded_data=b"custom attributes")
    entries = list(
        attribute_entries([first, 3, second, stream, 5], resolve=identity, resolve_revision=integer)
    )
    assert [(entry.value, entry.revision, entry.explicit_revision) for entry in entries] == [
        (first, 3, True),
        (second, 0, False),
        (stream, 5, True),
    ]
    assert entries[0].value is first
    assert entries[2].value is stream


def test_role_map_keeps_document_supplied_mapping() -> None:
    assert parse_role_map([("CustomHeading", "H1")], lambda value: str(value)) == {
        "CustomHeading": "H1"
    }
    with pytest.raises(ValueError, match="role map entry"):
        parse_role_map([("CustomHeading", object())], lambda value: None)


def test_marked_content_identifier_is_nonnegative_integer() -> None:
    assert marked_content_id(0) == 0
    for value in (-1, True, "1"):
        with pytest.raises(ValueError, match="mcid"):
            marked_content_id(value)
