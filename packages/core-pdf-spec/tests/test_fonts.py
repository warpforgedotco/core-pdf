"""Standalone font semantics reject damage and retain prescribed PDF defaults."""

from __future__ import annotations

import importlib
import importlib.util
import re

import pytest

from core_pdf_spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf_spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf_spec.s_09_fonts.cmap_tokenizer import CMapProgram, decode_cmap_hex_token
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf_spec.s_09_fonts.dictionaries import prepare_font_program_inputs
from core_pdf_spec.s_09_fonts.font_program import (
    CFFFont,
    execute_type2_charstring,
)
from core_pdf_spec.s_09_fonts.font_program_type1 import (
    binary_entries,
    decode_charstring,
    decode_eexec_payload,
)
from core_pdf_spec.s_09_fonts.helpers import (
    base_encoding_glyph_names,
    build_simple_encoding_glyph_names,
)
from core_pdf_spec.s_09_fonts.widths import get_descendant, parse_font_widths

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"


@pytest.mark.parametrize(
    "mapping",
    [
        b"2 begincidchar <01> 7 <02> bad endcidchar",
        b"2 begincidchar <01> 7 <02> 65536 endcidchar",
        b"2 begincidchar <01> 7 <02> endcidchar",
        b"1 begincidrange <01> <03> 65535 endcidrange",
        b"1 begincidrange <03> <01> 7 endcidrange",
        b"1 begincidchar {<01>} 7 endcidchar",
        b"1 begincidchar <0100> 7 endcidchar",
        b"1 begincidchar <01> 7",
    ],
)
def test_cid_cmap_rejects_malformed_mappings(mapping: bytes) -> None:
    with pytest.raises(ValueError):
        CMapDecoder(CODESPACE + mapping)


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_cmap_rejects_unresolved_and_cyclic_parent(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    data = CODESPACE + b"/Missing usecmap"
    with pytest.raises(ValueError, match="unresolved"):
        reader(data)
    with pytest.raises(ValueError, match="cyclic"):
        reader(data, usecmap_resolver=lambda name: data)


@pytest.mark.parametrize(
    "suffix",
    [b"<unterminated", b"(unterminated", b"[<00>", b"{dup", b"begincmap"],
)
def test_cmap_rejects_incomplete_tokens_and_scope(suffix: bytes) -> None:
    with pytest.raises(ValueError, match="unterminated"):
        CMapProgram.parse(CODESPACE + suffix)


def test_cmap_preserves_spec_defined_identity_and_invalid_code_consumption() -> None:
    cmap = CMapDecoder(b"/Identity-H usecmap")
    assert cmap.decode_entries(b"\x00A\x00") == [(b"\x00A", 65), (b"\x00", 0)]
    assert decode_cmap_hex_token(b"<4>") == b"@"


def test_tounicode_uses_utf16be_and_local_mapping_overrides_parent() -> None:
    parent = CODESPACE + b"1 beginbfchar <01> <0041> endbfchar"
    child = b"/Parent usecmap 1 beginbfchar <01> <D83DDE00> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.lookup(b"\x01") == "\U0001f600"
    with pytest.raises(UnicodeDecodeError):
        ToUnicodeCMap(CODESPACE + b"1 beginbfchar <01> <41> endbfchar")


def test_adobe_resources_load_with_parent_inheritance() -> None:
    cmap = resolve_cmap_decoder("UniJIS-UTF16-V")
    assert cmap is not None
    assert cmap.wmode == 1
    assert cmap.decode_entries(b"\x00A")[0][1] > 0


def test_static_annex_d_glyph_names_preserve_exact_character_slots() -> None:
    standard = base_encoding_glyph_names("StandardEncoding")
    mac = base_encoding_glyph_names("MacRomanEncoding")
    win = base_encoding_glyph_names("WinAnsiEncoding")
    assert len(standard) == len(mac) == len(win) == 256
    assert standard[39] == "quoteright"
    assert (mac[0], mac[0xCA], mac[0xDB], mac[0xF0]) == (
        ".notdef",
        "space",
        "currency",
        ".notdef",
    )
    assert (win[0x7F], win[0xAD], win[0xB2]) == ("bullet", "hyphen", "twosuperior")


def test_cff_parses_without_font_backend_and_keeps_standard_encoding() -> None:
    # Header, one name, one Top DICT (CharStrings offset 21), empty String
    # and Global Subr indexes, then one .notdef charstring containing endchar.
    font = CFFFont(
        b"\x01\x00\x04\x04"
        b"\x00\x01\x01\x01\x02F"
        b"\x00\x01\x01\x01\x03\xa0\x11"
        b"\x00\x00\x00\x00"
        b"\x00\x01\x01\x01\x02\x0e"
    )
    assert font.charstrings == [b"\x0e"]
    assert font.glyph_id_for_name("missing") == 0
    assert font.builtin_encoding()[65] == "A"
    assert font.font_matrix(0) == (0.001, 0.0, 0.0, 0.001, 0.0, 0.0)
    with pytest.raises(ValueError, match="charset"):
        font.read_charset(len(font.data), 2)
    with pytest.raises(ValueError, match="encoding"):
        font.read_encoding_codes(len(font.data))


def test_type2_operators_emit_exact_displacements() -> None:
    moves: list[tuple[float, float]] = []
    lines: list[tuple[float, float]] = []
    assert (
        execute_type2_charstring(
            bytes([149, 159, 21, 169, 139, 5, 14]),
            local_subrs=(),
            global_subrs=(),
            move=lambda x, y: moves.append((x, y)),
            line=lambda x, y: lines.append((x, y)),
            curve=lambda *args: None,
            flush_contour=lambda: None,
            has_current_point=lambda: bool(moves),
            seac=lambda *args: None,
            random_value=lambda: 0.5,
        )
        is False
    )
    assert moves == [(10.0, 20.0)]
    assert lines == [(30.0, 0.0)]


def test_type1_byte_primitives_reject_truncation_and_preserve_unencrypted_charstrings() -> None:
    assert decode_charstring(b"\x8b\x0e", -1) == b"\x8b\x0e"
    with pytest.raises(ValueError, match="prefix"):
        decode_charstring(b"\x00", 4)
    with pytest.raises(ValueError, match="odd"):
        decode_eexec_payload(b"0000000000", 1)
    with pytest.raises(ValueError, match="truncated"):
        list(binary_entries(b"/A 5 RD abc", re.compile(rb"/(\w+) (\d+) RD ")))


def test_malformed_font_descriptor_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": 42})
    with pytest.raises(ValueError, match="stream"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": {"FontFile": 42}})


def test_missing_width_is_spec_defined_zero() -> None:
    assert parse_font_widths({}, "Type1").default_width == 0.0
    assert parse_font_widths(
        {"FontDescriptor": {"MissingWidth": 0}}, "Type1"
    ).default_width_explicit


def test_fonttools_backend_is_not_exposed_by_spec() -> None:
    assert importlib.util.find_spec("core_pdf_spec.s_09_fonts.font_program_opentype") is None
    module = importlib.import_module("core_pdf_spec.s_09_fonts.font_program_type1")
    assert not hasattr(module, "Type1FontProgram")


@pytest.mark.parametrize("offset", [21.5, -1.0, float("nan")])
def test_cff_rejects_non_integral_or_invalid_offsets(offset: float) -> None:
    font = CFFFont.__new__(CFFFont)
    font.top_dict = {17: [offset]}
    with pytest.raises(ValueError, match="offset"):
        font.dict_offset(17)


def test_cff_rejects_invalid_header_fields() -> None:
    with pytest.raises(ValueError, match="header"):
        CFFFont(b"\x01\x00\x02\x00")


@pytest.mark.parametrize("entry", [{256: "A"}, {65: ""}])
def test_simple_encoding_rejects_malformed_explicit_entries(entry: dict[int, str]) -> None:
    with pytest.raises(ValueError, match="entry"):
        build_simple_encoding_glyph_names(None, entry, {}, authoritative_builtin=False)


@pytest.mark.parametrize("descendants", [[], [{}, {}], [42]])
def test_descendant_font_array_must_contain_one_dictionary(descendants: list[object]) -> None:
    with pytest.raises(ValueError, match="DescendantFonts"):
        get_descendant({"DescendantFonts": descendants})


def test_cid_bounds_are_supported_exports() -> None:
    from core_pdf_spec.s_09_fonts import cmap_widths

    assert (cmap_widths.MIN_CID, cmap_widths.MAX_CID) == (0, 65535)
    assert {"MIN_CID", "MAX_CID"} <= set(cmap_widths.__all__)


def test_cff_dict_rejects_unconsumed_operands() -> None:
    font = CFFFont.__new__(CFFFont)
    assert font.read_dict_entries(b"\x8b\x11\x8c") == ({17: [0.0]}, [1.0])
    with pytest.raises(ValueError, match="unterminated"):
        font.parse_dict(b"\x8b\x11\x8c")


def test_explicit_font_widths_require_a_character_range() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_font_widths({"Widths": [500]}, "Type1")


def test_cid_widths_reject_nonfinite_numbers_in_compact_array() -> None:
    from core_pdf_spec.s_09_fonts.cmap_widths import parse_cid_widths

    with pytest.raises(ValueError, match="CID width"):
        parse_cid_widths([0, [float("nan")]])
