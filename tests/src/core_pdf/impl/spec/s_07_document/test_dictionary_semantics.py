"""Raw PDF dictionary semantics are independent of reader output/recovery."""

from typing import cast

import pytest

from core_pdf.impl.spec.s_07_document.fields import (
    field_children,
    inherited_field_value,
    qualified_field_name,
)
from core_pdf.impl.spec.s_07_document.metadata import info_dictionary, metadata_stream
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.types import PdfString
from tests.helpers.resolvers import IdentityResolver


def test_field_names_and_inheritance_preserve_empty_literal_values() -> None:
    parent = PdfString(b"parent")
    empty = PdfString(b"")
    assert qualified_field_name("parent", "child") == "parent.child"
    assert qualified_field_name("parent", None) == "parent"
    assert inherited_field_value({}, "V", parent) is parent
    assert inherited_field_value({"V": None}, "V", parent) is parent
    assert inherited_field_value({"V": empty}, "V", parent) is empty
    with pytest.raises(ValueError, match="Kids"):
        field_children(False)


def test_metadata_resolution_preserves_raw_dictionary_and_stream() -> None:
    resolver = cast(PdfValueResolver, IdentityResolver())
    info: PdfDict = {"Title": PdfString(b"  literal title  ")}
    stream = PdfStream({}, b"", decoded_data=b"<xmp/>")
    trailer: PdfDict = {"Info": info, "Root": {"Metadata": stream}}
    assert info_dictionary(resolver, trailer) is info
    assert metadata_stream(resolver, trailer) is stream
    assert info["Title"] == PdfString(b"  literal title  ")
    with pytest.raises(ValueError, match="Metadata stream"):
        metadata_stream(resolver, {"Root": {"Metadata": 7}})
