# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for native AcroForm field helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.spec.s_07_document.document import PdfDocument as SpecPdfDocument
from core_pdf.impl.engine.spec.s_07_document.fields import field_value_text
from core_pdf.impl.primitives import PdfName, PdfString
from core_pdf.impl.types import PdfDict


class IdentityResolver:
    def resolve(self, value: object) -> object:
        return value


class Document:
    resolver = IdentityResolver()


def test_field_value_text_decodes_supported_scalar_and_array_values() -> None:
    value = [PdfString(b"first"), PdfName.of("second"), b"third", " fourth "]

    assert field_value_text(Document(), value) == "first\nsecond\nthird\nfourth"


def test_field_value_text_ignores_signature_dictionary() -> None:
    value = {PdfName.of("Type"): PdfName.of("Sig"), PdfName.of("Contents"): b"signature"}

    assert field_value_text(Document(), value) == ""


def test_widget_value_overrides_empty_parent_value() -> None:
    pdf_path = (
        Path(__file__).resolve().parents[7]
        / "tests/fixtures/pikepdf/tests/resources/form_dd0293.pdf"
    )
    if not pdf_path.is_file():
        pytest.skip("pikepdf fixture submodule is not initialized")

    with PdfDocument.open(pdf_path) as document:
        fields = document.pages[0].get_fields()

    assert any(field.value_text == "Controlled by: CUI Category: LDC: POC:" for field in fields)


def test_widget_field_root_crosses_non_field_parent_nodes() -> None:
    root: PdfDict = {PdfName.of("FT"): PdfName.of("Tx")}
    middle: PdfDict = {PdfName.of("Parent"): root}
    widget: PdfDict = {PdfName.of("Parent"): middle}

    assert SpecPdfDocument.internal_widget_field_root(cast(Any, Document()), widget) is root
