# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for native AcroForm field helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.spec.s_07_document.fields import collect_field_records, field_value_text
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from tests.helpers.paths import FIXTURES, require_fixture
from tests.helpers.resolvers import IdentityResolver


class Document:
    resolver = IdentityResolver()


def test_field_value_text_decodes_supported_scalar_and_array_values() -> None:
    value = [PdfString(b"first"), PdfName.of("second"), b"third", " fourth "]

    assert field_value_text(Document.resolver, value) == "first\nsecond\nthird\nfourth"


def test_field_value_text_ignores_signature_dictionary() -> None:
    value = {PdfName.of("Type"): PdfName.of("Sig"), PdfName.of("Contents"): b"signature"}

    assert field_value_text(Document.resolver, value) == ""


def test_field_collection_preserves_depth_first_order_and_widget_inheritance() -> None:
    widget: PdfDict = {
        "Subtype": PdfName.of("Widget"),
        "T": PdfString(b"leaf"),
        "V": PdfString(b""),
        "Kids": False,
    }
    branch: PdfDict = {"T": PdfString(b"branch"), "Kids": [widget]}
    sibling: PdfDict = {"Subtype": PdfName.of("Widget"), "Rect": [1, 2, 3, 4]}
    root: PdfDict = {
        "T": PdfString(b"root"),
        "FT": PdfName.of("Tx"),
        "V": PdfString(b"parent"),
        "Kids": [branch, sibling],
    }

    records = collect_field_records(Document.resolver, root, recover=False)

    assert [record.name for record in records] == [
        "root",
        "root.branch",
        "root.branch.leaf",
        "root",
    ]
    assert [record.type for record in records] == ["Tx"] * 4
    assert [record.value_text for record in records] == ["parent", "parent", "", "parent"]
    assert records[2].widget is widget
    assert records[2].kids == []
    assert records[3].rect == (1, 2, 3, 4)


def test_field_collection_cycle_is_strict_or_skipped_in_recovery() -> None:
    root: PdfDict = {"T": PdfString(b"root")}
    root["Kids"] = [root, {"T": PdfString(b"sibling")}]

    with pytest.raises(ValueError, match="invalid AcroForm field entry"):
        collect_field_records(Document.resolver, root, recover=False)
    assert [
        record.name for record in collect_field_records(Document.resolver, root, recover=True)
    ] == ["root", "root.sibling"]


@pytest.mark.parametrize("recover", [False, True])
def test_field_collection_depth_limit(recover: bool) -> None:
    root: PdfDict = {}
    node = root
    for _ in range(51):
        child: PdfDict = {}
        node["Kids"] = [child]
        node = child
    if recover:
        assert len(collect_field_records(Document.resolver, root, recover=True)) == 51
    else:
        with pytest.raises(ValueError, match="invalid AcroForm depth"):
            collect_field_records(Document.resolver, root, recover=False)


@pytest.mark.parametrize(
    ("node", "error"),
    [(False, "field entry"), ({"Kids": False}, "Kids array"), ({"Kids": [False]}, "kid entry")],
)
def test_field_collection_malformed_entries(node: object, error: str) -> None:
    with pytest.raises(ValueError, match=f"invalid AcroForm {error}"):
        collect_field_records(Document.resolver, node, recover=False)
    assert len(collect_field_records(Document.resolver, node, recover=True)) == (
        0 if node is False else 1
    )


def test_widget_value_overrides_empty_parent_value() -> None:
    pdf_path = require_fixture(
        FIXTURES / "pikepdf" / "tests" / "resources" / "form_dd0293.pdf",
        "pikepdf fixture submodule is not initialized",
    )

    with PdfDocument.open(pdf_path) as document:
        fields = document.pages[0].get_fields()

    assert any(field.value_text == "Controlled by: CUI Category: LDC: POC:" for field in fields)


def test_widget_field_root_crosses_non_field_parent_nodes() -> None:
    root: PdfDict = {PdfName.of("FT"): PdfName.of("Tx")}
    middle: PdfDict = {PdfName.of("Parent"): root}
    widget: PdfDict = {PdfName.of("Parent"): middle}

    assert SpecPdfDocument.internal_widget_field_root(cast(Any, Document()), widget) is root
