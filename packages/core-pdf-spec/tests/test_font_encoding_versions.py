# SPDX-License-Identifier: AGPL-3.0-only
"""Historical encoding assignments, separately from explicit font glyph selection."""

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_09_fonts.helpers import (
    build_simple_encoding_glyph_names,
    get_base_encoding_glyph_names,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext

VERSIONS = [*[PdfVersion(1, n) for n in range(8)], PdfVersion(2, 0)]


@pytest.mark.parametrize("version", VERSIONS)
def test_winansi_assignments_change_in_pdf_1_3(version: PdfVersion) -> None:
    # Adobe PDF 1.2, Annex C.1 (printed p.343): unused WinAnsi high codes select
    # bullet. Adobe PDF 1.3, Annex D.1 notes 1–3 (printed p.554) reassigns these
    # three slots, while the PDF MacRoman currency slot remains unchanged.
    # https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.3.pdf
    names = get_base_encoding_glyph_names("WinAnsiEncoding", context=SemanticContext(version))
    assert tuple(names[code] for code in (0x80, 0x8E, 0x9E)) == (
        ("bullet", "bullet", "bullet")
        if version < PdfVersion(1, 3)
        else ("Euro", "Zcaron", "zcaron")
    )
    assert names[0x95] == names[0x81] == "bullet"
    assert names[0xA0] == "space"


@pytest.mark.parametrize("version", VERSIONS)
def test_macroman_keeps_currency_despite_mac_os_euro_change(version: PdfVersion) -> None:
    names = get_base_encoding_glyph_names("MacRomanEncoding", context=SemanticContext(version))
    assert names[0xDB] == "currency"


@pytest.mark.parametrize("version", [PdfVersion(1, n) for n in range(3)])
def test_historical_winansi_still_allows_explicit_glyph_overrides(version: PdfVersion) -> None:
    names = build_simple_encoding_glyph_names(
        "WinAnsiEncoding",
        {0x80: "Euro"},
        {0x8E: "Zcaron", 0x9E: "zcaron"},
        authoritative_builtin=False,
        context=SemanticContext(version),
    )
    assert tuple(names[code] for code in (0x80, 0x8E, 0x9E)) == ("Euro", "Zcaron", "zcaron")


def test_authoritative_program_encoding_does_not_inherit_historical_bullets() -> None:
    names = build_simple_encoding_glyph_names(
        "WinAnsiEncoding",
        {0x80: "Euro"},
        {0x80: "custom"},
        authoritative_builtin=True,
        context=SemanticContext(PdfVersion(1, 2)),
    )
    assert names[0x80] == "custom"
    assert names[0x8E] == names[0x9E] == ".notdef"


def test_omitted_font_encoding_context_retains_modern_assignments() -> None:
    names = build_simple_encoding_glyph_names(
        "WinAnsiEncoding", {}, {}, authoritative_builtin=False
    )
    assert tuple(names[code] for code in (0x80, 0x8E, 0x9E)) == ("Euro", "Zcaron", "zcaron")


@pytest.mark.parametrize("version", [None, PdfVersion(9, 0)])
@pytest.mark.parametrize("authoritative", [False, True])
def test_font_encoding_rejects_unrecognized_explicit_context(
    version: PdfVersion | None, authoritative: bool
) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        build_simple_encoding_glyph_names(
            "WinAnsiEncoding",
            {},
            {},
            authoritative_builtin=authoritative,
            context=SemanticContext(version),
        )
