# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for native AcroForm field helpers."""

from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_document.fields import field_value_text
from core_pdf.impl.objects import PdfName, PdfString


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
