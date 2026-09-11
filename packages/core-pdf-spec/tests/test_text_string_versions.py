# SPDX-License-Identifier: AGPL-3.0-only
"""The bytes that select Unicode encodings acquired meaning in specific versions."""

import pytest

from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfString


@pytest.mark.parametrize("version", [PdfVersion(1, 0), PdfVersion(1, 1)])
def test_utf16_bom_is_pdfdocencoding_before_pdf_1_2(version: PdfVersion) -> None:
    # Adobe PDF Reference 1.2, 4.4 (printed p.45) distinguishes PDF 1.1's
    # PDFDocEncoding from the new FE FF-prefixed Unicode text representation.
    # https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.2.pdf
    context = SemanticContext(version)
    assert decode_pdf_text_string(b"\xfe\xff\x00H", context=context) == "þÿ\x00H"
    # A byte count invalid for Unicode is still a valid legacy text string.
    assert decode_pdf_text_string(b"\xfe\xffH", context=context) == "þÿH"


@pytest.mark.parametrize("version", [*[PdfVersion(1, n) for n in range(2, 8)], PdfVersion(2, 0)])
def test_utf16_bom_selects_unicode_since_pdf_1_2(version: PdfVersion) -> None:
    context = SemanticContext(version)
    assert decode_pdf_text_string(memoryview(b"\xfe\xff\x00H"), context=context) == "H"
    with pytest.raises(ValueError, match="invalid UTF-16BE data"):
        decode_pdf_text_string(b"\xfe\xffH", context=context)


@pytest.mark.parametrize("version", [*[PdfVersion(1, n) for n in range(8)], PdfVersion(2, 0)])
def test_plain_text_encoding_is_shared_across_versions(version: PdfVersion) -> None:
    assert decode_pdf_text_string(b"A\x80B", context=SemanticContext(version)) == "A•B"


def test_utf16_without_context_preserves_existing_unicode_api() -> None:
    assert decode_pdf_text_string(b"\xfe\xff\x00H") == "H"
    with pytest.raises(ValueError, match="invalid UTF-16BE data"):
        decode_pdf_text_string(b"\xfe\xffH")


def test_resolver_applies_the_selected_utf16_semantics() -> None:
    resolver = ObjectResolver(b"", {}, semantic_context=SemanticContext(PdfVersion(1, 1)))
    try:
        value = PdfString(b"\xfe\xff\x00H")
        assert resolver.resolve_str(value) == "þÿ\x00H"
        resolver.semantic_context = SemanticContext(PdfVersion(1, 2))
        assert resolver.resolve_str(value) == "H"
    finally:
        resolver.close()
