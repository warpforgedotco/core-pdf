from __future__ import annotations

import importlib
import importlib.util

import pytest

from core_adobe_fonts.cmap.decoder import CMapDecoder
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf_spec.s_09_fonts.dictionaries import get_descendant, prepare_font_program_inputs
from core_pdf_spec.s_09_fonts.helpers import (
    BASE_ENCODING_GLYPH_NAMES,
    build_simple_encoding_glyph_names,
)
from core_pdf_spec.s_09_fonts.widths import parse_font_widths

CODESPACE = b"1 begincodespacerange <00> <ff> endcodespacerange\n"


@pytest.mark.parametrize("reader", [CMapDecoder, ToUnicodeCMap])
def test_cmap_rejects_unresolved_and_cyclic_parent(
    reader: type[CMapDecoder | ToUnicodeCMap],
) -> None:
    data = b"/Missing usecmap"
    with pytest.raises(ValueError, match="unresolved"):
        reader(data)
    with pytest.raises(ValueError, match="cyclic"):
        reader(data, usecmap_resolver=lambda name: data)


class ParentlessToUnicodeCMap(ToUnicodeCMap):
    __slots__ = ()

    max_inheritance_depth = 1

    def reject_parent(self, reason: str) -> ToUnicodeCMap | None:  # noqa: ARG002
        return None

    def validate_mappings(self) -> None:
        pass


def test_tounicode_parent_hooks_keep_strict_defaults() -> None:
    local = CODESPACE + b"1 beginbfchar <01> <0041> endbfchar"
    data = b"/Loop usecmap 1 beginbfchar <01> <0041> endbfchar"
    with pytest.raises(ValueError, match="unresolved"):
        ToUnicodeCMap(data)
    assert ParentlessToUnicodeCMap(data).lookup(b"\x01") == "A"
    calls: list[str] = []

    def resolve(name: str) -> bytes:
        calls.append(name)
        return data

    assert ParentlessToUnicodeCMap(data, usecmap_resolver=resolve).lookup(b"\x01") == "A"
    assert calls == ["Loop"]
    assert ToUnicodeCMap(local, inheritance_depth=99).lookup(b"\x01") == "A"
    with pytest.raises(ValueError, match="nesting too deep"):
        ParentlessToUnicodeCMap(local, inheritance_depth=2)


def test_tounicode_uses_utf16be_and_local_mapping_overrides_parent() -> None:
    parent = CODESPACE + b"1 beginbfchar <01> <0041> endbfchar"
    child = b"/Parent usecmap 1 beginbfchar <01> <D83DDE00> endbfchar"
    cmap = ToUnicodeCMap(child, usecmap_resolver=lambda name: parent)
    assert cmap.lookup(b"\x01") == "\U0001f600"
    with pytest.raises(UnicodeDecodeError):
        ToUnicodeCMap(CODESPACE + b"1 beginbfchar <01> <41> endbfchar")


def test_static_annex_d_glyph_names_preserve_exact_character_slots() -> None:
    standard = BASE_ENCODING_GLYPH_NAMES["StandardEncoding"]
    mac = BASE_ENCODING_GLYPH_NAMES["MacRomanEncoding"]
    win = BASE_ENCODING_GLYPH_NAMES["WinAnsiEncoding"]
    assert len(standard) == len(mac) == len(win) == 256
    assert standard[39] == "quoteright"
    assert (mac[0], mac[0xCA], mac[0xDB], mac[0xF0]) == (
        ".notdef",
        "space",
        "currency",
        ".notdef",
    )
    assert (win[0x7F], win[0xAD], win[0xB2]) == ("bullet", "hyphen", "twosuperior")


def test_malformed_font_descriptor_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": 42})
    with pytest.raises(ValueError, match="stream"):
        prepare_font_program_inputs({"Subtype": "Type1", "FontDescriptor": {"FontFile": 42}})


def test_font_program_input_readers_replace_the_strict_coercions() -> None:
    font = {
        "Subtype": "/Type0",
        "FontDescriptor": 42,
        "DescendantFonts": [{"Subtype": "CIDFontType2", "FontDescriptor": {"FontFile2": 7}}],
    }
    with pytest.raises(ValueError, match="descriptor"):
        prepare_font_program_inputs(font)
    inputs = prepare_font_program_inputs(
        font,
        read_name=lambda value: value.lstrip("/") if isinstance(value, str) else None,
        read_descriptor=lambda value: value if isinstance(value, dict) else None,
        read_font_file=lambda descriptor, key: None,
    )
    assert (inputs.subtype, inputs.original_subtype) == ("CIDFontType2", "Type0")
    assert inputs.descendant is font["DescendantFonts"][0]
    assert (inputs.font_file, inputs.font_file2, inputs.font_file3) == (None, None, None)


def test_missing_width_is_spec_defined_zero() -> None:
    assert parse_font_widths({}, "Type1").default_width == 0.0
    relaxed = parse_font_widths({}, "Type1", default_width=1000.0)
    assert (relaxed.default_width, relaxed.default_width_explicit) == (1000.0, False)
    described = {"FontDescriptor": {"MissingWidth": 250}}
    assert parse_font_widths(described, "Type1", default_width=1000.0).default_width == 250.0
    assert (
        parse_font_widths({"DescendantFonts": [{}]}, "Type0", default_width=5.0).default_width
        == 1000.0
    )
    assert parse_font_widths(
        {"FontDescriptor": {"MissingWidth": 0}}, "Type1"
    ).default_width_explicit


def test_fonttools_backend_is_not_exposed_by_spec() -> None:
    assert importlib.util.find_spec("core_pdf_spec.s_09_fonts.font_program_opentype") is None
    assert importlib.util.find_spec("core_pdf_spec.s_09_fonts.font_program") is None
    module = importlib.import_module("core_adobe_fonts.type1.program")
    assert not hasattr(module, "Type1FontProgram")


@pytest.mark.parametrize("entry", [{256: "A"}, {65: ""}])
def test_simple_encoding_rejects_malformed_explicit_entries(entry: dict[int, str]) -> None:
    with pytest.raises(ValueError, match="entry"):
        build_simple_encoding_glyph_names(None, entry, {}, authoritative_builtin=False)


@pytest.mark.parametrize("descendants", [[], [{}, {}], [42]])
def test_descendant_font_array_must_contain_one_dictionary(descendants: list[object]) -> None:
    with pytest.raises(ValueError, match="DescendantFonts"):
        get_descendant({"DescendantFonts": descendants})


def test_descendant_font_single_dictionary_is_the_shared_valid_case() -> None:
    descendant = {"Subtype": "CIDFontType2"}
    assert get_descendant({"DescendantFonts": [descendant]}) is descendant


def test_cid_bounds_are_supported_exports() -> None:
    from core_pdf_spec.s_09_fonts import widths

    assert (widths.MIN_CID, widths.MAX_CID) == (0, 65535)
    assert {"MIN_CID", "MAX_CID"} <= set(widths.__all__)


def test_explicit_font_widths_require_a_character_range() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_font_widths({"Widths": [500]}, "Type1")


def test_cid_widths_reject_nonfinite_numbers_in_compact_array() -> None:
    from core_pdf_spec.s_09_fonts.widths import parse_cid_widths

    with pytest.raises(ValueError, match="CID width"):
        parse_cid_widths([0, [float("nan")]])
