# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfString


@pytest.mark.parametrize("version", [PdfVersion(1, 0), PdfVersion(1, 1)])
def test_utf16_bom_is_pdfdocencoding_before_pdf_1_2(version: PdfVersion) -> None:
    context = SemanticContext(version)
    assert decode_pdf_text_string(b"\xfe\xff\x00H", context=context) == "þÿ\x00H"
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


@pytest.mark.parametrize("version", [PdfVersion(1, n) for n in range(3)])
def test_pdfdocencoding_euro_slot_is_undefined_before_pdf_1_3(version: PdfVersion) -> None:
    with pytest.raises(ValueError, match="undefined PDFDocEncoding byte 0xA0"):
        decode_pdf_text_string(memoryview(b"price \xa0"), context=SemanticContext(version))
    assert decode_pdf_text_string(b"\x99\x9e", context=SemanticContext(version)) == "Žž"


@pytest.mark.parametrize("version", [*[PdfVersion(1, n) for n in range(3, 8)], PdfVersion(2, 0)])
def test_pdfdocencoding_euro_since_pdf_1_3(version: PdfVersion) -> None:
    assert decode_pdf_text_string(b"price \xa0", context=SemanticContext(version)) == "price €"


def test_pdfdocencoding_euro_without_context_preserves_existing_api() -> None:
    assert decode_pdf_text_string(b"price \xa0") == "price €"


def test_pdfdocencoding_undefined_slot_does_not_restrict_unicode_bytes() -> None:
    assert (
        decode_pdf_text_string(
            b"\xfe\xff\x00\xa0\xa0\x00", context=SemanticContext(PdfVersion(1, 2))
        )
        == "\u00a0\ua000"
    )


def test_resolver_uses_the_pdfdocencoding_euro_boundary() -> None:
    resolver = ObjectResolver(b"", {}, semantic_context=SemanticContext(PdfVersion(1, 2)))
    try:
        with pytest.raises(ValueError, match="undefined PDFDocEncoding"):
            resolver.resolve_str(PdfString(b"\xa0"))
        resolver.semantic_context = SemanticContext(PdfVersion(1, 3))
        assert resolver.resolve_str(PdfString(b"\xa0")) == "€"
    finally:
        resolver.close()
