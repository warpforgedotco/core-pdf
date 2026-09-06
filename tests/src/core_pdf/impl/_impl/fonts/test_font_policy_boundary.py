"""PDF rules remain available independently of the application's recovery choices."""

from __future__ import annotations

import pytest

from core_pdf.impl._impl.fonts.cmap_decoder import CMapDecoder as ApplicationCMap
from core_pdf.impl._impl.fonts.cmap_tounicode import ToUnicodeCMap as RecoveringToUnicode
from core_pdf.impl._impl.fonts.cmap_widths import parse_cid_widths as recover_cid_widths
from core_pdf.impl._impl.fonts.font_program import CFFFont as RecoveringCFF
from core_pdf.impl._impl.fonts.glyphs import glyph_name_to_unicode as recover_glyph_name
from core_pdf.impl._impl.fonts.widths import parse_font_widths as recover_widths
from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.cmap_widths import parse_cid_widths
from core_pdf.impl.spec.s_09_fonts.font_program import CFFFont
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.spec.s_09_fonts.widths import parse_font_widths


def cmap(body: bytes, codespace: bytes = b"<00> <ff>") -> bytes:
    return b"1 begincodespacerange " + codespace + b" endcodespacerange " + body


def test_odd_utf16_destination_is_application_recovery() -> None:
    data = cmap(b"1 beginbfchar <01> <41> endbfchar")
    with pytest.raises(UnicodeDecodeError):
        ToUnicodeCMap(data)
    assert RecoveringToUnicode(data).decode(b"\x01") == "A"


def test_invalid_codespace_salvage_is_application_recovery() -> None:
    data = cmap(b"1 beginbfchar <0083> <0041> endbfchar", b"<0083> <020c>")
    with pytest.raises(ValueError):
        ToUnicodeCMap(data)
    assert RecoveringToUnicode(data).mappings[b"\x00\x83"] == "A"


def test_one_byte_identity_is_an_application_resource_alias() -> None:
    program = b"/OneByteIdentityH usecmap"
    assert CMapDecoder(program).decode_entries(b"A") == [(b"A", 0)]
    assert ApplicationCMap(program).decode_entries(b"A") == [(b"A", 65)]


def test_zapf_mapping_requires_the_zapf_font_context() -> None:
    assert glyph_name_to_unicode("a1") == ""
    assert glyph_name_to_unicode("a1", zapf_dingbats=True) == "\u2701"
    assert recover_glyph_name("a1") == "\u2701"


def test_missing_simple_font_width_is_not_an_invented_em() -> None:
    assert parse_font_widths({}, "Type1").default_width == 0
    assert recover_widths({}, "Type1").default_width == 1000
    explicit = {"FontDescriptor": {"MissingWidth": 0}}
    assert parse_font_widths(explicit, "Type1").default_width == 0
    assert recover_widths(explicit, "Type1").default_width == 0


def test_clipping_malformed_cid_ranges_is_application_recovery() -> None:
    with pytest.raises(ValueError):
        parse_cid_widths([-1, [10, 20]])
    assert dict(recover_cid_widths([-1, [10, 20]])) == {0: 20}


def test_missing_and_out_of_bounds_cff_data_are_application_recovery() -> None:
    with pytest.raises(ValueError):
        CFFFont(None)
    font = RecoveringCFF(None)
    assert font.charstrings == []
    with pytest.raises(ValueError):
        CFFFont.internal_read_charset(font, 50, 2)
    assert font.internal_read_charset(50, 2) == {0: 0, 1: 1}


@pytest.mark.parametrize(
    ("encoded", "literal", "recovered"),
    [
        (b"(first\n\rsecond)", b"first\n\nsecond", b"first\nsecond"),
        (b"(a\r\nb\n\rc\rd\ne)", b"a\nb\n\nc\nd\ne", b"a\nb\nc\nd\ne"),
        (b"(a\\\r\nb\\\n\rc\\\rd\\\ne)", b"ab\ncde", b"abcde"),
    ],
)
def test_cmap_literal_eol_recovery_is_application_owned(
    encoded: bytes, literal: bytes, recovered: bytes
) -> None:
    from core_pdf.impl._impl.fonts.cmap_tokenizer import decode_cmap_token
    from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import decode_pdf_literal_string

    assert decode_pdf_literal_string(encoded) == literal
    assert decode_cmap_token(encoded) == recovered


@pytest.mark.parametrize(
    ("body", "recovered"),
    [
        (b"2 beginbfchar <41> <0041> <42> endbfchar", {b"A": "A"}),
        (b"1 beginbfchar (Z) <0041> endbfchar", {b"Z": "A"}),
        (b"1 beginbfrange <41> <43> [<0041>] endbfrange", {b"A": "A"}),
        (b"1 beginbfrange <41> <41> [<0041> <0042>] endbfrange", {b"A": "A"}),
        (
            b"1 beginbfrange <41> <43> [<0041> <zz> <0043>] endbfrange",
            {b"A": "A", b"C": "C"},
        ),
        (b"2 beginbfrange <41> <41> <0041> <42> endbfrange", {b"A": "A"}),
        (b"1 beginbfrange <41> <42> <41> endbfrange", {b"A": "A", b"B": "B"}),
        (
            b"1 beginbfrange <41> <42> <d7ff> endbfrange",
            {b"A": "\ud7ff", b"B": "\ufffd"},
        ),
    ],
)
def test_shared_mapping_records_preserve_reader_recovery(
    body: bytes, recovered: dict[bytes, str]
) -> None:
    program = cmap(body)
    with pytest.raises(ValueError):
        ToUnicodeCMap(program)
    assert RecoveringToUnicode(program).mappings == recovered


def test_shared_mapping_records_leave_bom_policy_at_the_decoder() -> None:
    program = cmap(b"1 beginbfrange <41> <42> <feff0041> endbfrange")
    assert ToUnicodeCMap(program).mappings == {b"A": "\ufeffA", b"B": "\ufeffB"}
    assert RecoveringToUnicode(program).mappings == {b"A": "A", b"B": "B"}


def test_partial_reader_arrays_do_not_expand_the_declared_source_range() -> None:
    program = cmap(
        b"1 beginbfrange <00000000> <ffffffff> [<0041>] endbfrange",
        b"<00000000> <ffffffff>",
    )
    with pytest.raises(ValueError):
        ToUnicodeCMap(program)
    assert RecoveringToUnicode(program).mappings == {b"\x00\x00\x00\x00": "A"}
