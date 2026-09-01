# SPDX-License-Identifier: AGPL-3.0-only
"""Font conformance rules from ISO 32000-1 clauses 9.6 and 9.7.

Each test names the clause it pins.
"""

from __future__ import annotations

import io
from typing import Any, cast

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import TrueTypeFontProgram

TWO_BYTE_CMAP = b"""
2 begincodespacerange <0000> <D7FF> <E000> <FFFF> endcodespacerange
1 begincidrange <0000> <FFFF> 0 endcidrange
"""


def test_invalid_code_consumes_the_chosen_codespace_length() -> None:
    """9.7.6.3: "The length of the codes in the chosen codespace range determines
    the total number of bytes to consume from the string".

    0xD800 falls in the gap between the two ranges. Consuming a single byte for
    it desynchronized every following code in the string.
    """
    decoder = CMapDecoder(TWO_BYTE_CMAP)

    entries = decoder.decode_entries(b"\xd8\x00\x00\x41\x00\x42")

    assert [(bytes(code), cid) for code, cid in entries] == [
        (b"\xd8\x00", 0),
        (b"\x00\x41", 0x41),
        (b"\x00\x42", 0x42),
    ]


def internal_simple_font(missing_width: object = None) -> dict[str, object]:
    descriptor: dict[str, object] = {"Flags": 4}
    if missing_width is not None:
        descriptor["MissingWidth"] = missing_width
    return {
        "Subtype": "TrueType",
        "FirstChar": 65,
        "LastChar": 66,
        "Widths": [500, 600],
        "FontDescriptor": descriptor,
    }


def test_explicit_missing_width_of_zero_is_honoured() -> None:
    """Table 122 defaults MissingWidth to 0, so a stated 0 must be used verbatim.

    It was replaced with a full em (and a half em for the space code), which
    shifts every following glyph on the line.
    """
    decoder = FontDecoder(internal_simple_font(0))

    assert decoder.default_width == 0.0
    assert decoder.glyph_width(0x30) == 0.0
    assert decoder.glyph_width(32) == 0.0


def test_absent_missing_width_keeps_the_lenient_default() -> None:
    """Deliberate leniency: a font that states nothing should not pile glyphs up."""
    decoder = FontDecoder(internal_simple_font())

    assert decoder.glyph_width(0x30) == 1000.0


def test_listed_widths_still_win_over_the_default() -> None:
    decoder = FontDecoder(internal_simple_font(0))

    assert decoder.glyph_width(65) == 500.0
    assert decoder.glyph_width(66) == 600.0


def test_cid_font_default_width_comes_from_dw_not_missing_width() -> None:
    """Table 117 makes DW the CIDFont default width.

    Table 122 scopes MissingWidth to "character codes whose widths are not
    specified in a font dictionary's Widths array"; a CIDFont has no Widths
    array, so a descriptor MissingWidth must not override DW.
    """
    font = {
        "Subtype": "Type0",
        "BaseFont": "X",
        "Encoding": "Identity-H",
        "DescendantFonts": [
            {
                "Subtype": "CIDFontType2",
                "DW": 500,
                "W": [120, [400, 325, 500]],
                "FontDescriptor": {"MissingWidth": 250},
            }
        ],
    }

    decoder = FontDecoder(font)

    assert decoder.default_width == 500.0
    assert decoder.glyph_width(999) == 500.0
    assert decoder.glyph_width(120) == 400.0


def test_cid_font_default_width_defaults_to_1000_without_dw() -> None:
    """Table 117: "Default value: 1000"."""
    font = {
        "Subtype": "Type0",
        "BaseFont": "X",
        "Encoding": "Identity-H",
        "DescendantFonts": [{"Subtype": "CIDFontType2", "W": []}],
    }

    assert FontDecoder(font).default_width == 1000.0


def internal_symbol_font() -> bytes:
    """A TrueType program carrying only a (3, 0) subtable, as symbol fonts do."""
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder([".notdef", "sq"])
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((0, 700))
    pen.lineTo((700, 700))
    pen.lineTo((700, 0))
    pen.closePath()
    builder.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "sq": pen.glyph()})
    builder.setupHorizontalMetrics({".notdef": (600, 0), "sq": (700, 0)})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupCharacterMap({0xF041: "sq"})
    builder.setupNameTable({"familyName": "T", "styleName": "R", "psName": "T-R"})
    builder.setupOS2()
    builder.setupPost()

    # newTable returns the untyped DefaultTable base, so the cmap-specific
    # attributes are set through a cast.
    table = cast(Any, newTable("cmap"))
    table.tableVersion = 0
    subtable = CmapSubtable.newSubtable(4)
    subtable.platformID, subtable.platEncID, subtable.language = 3, 0, 0
    subtable.cmap = {0xF041: "sq"}
    table.tables = [subtable]
    builder.font["cmap"] = table

    buffer = io.BytesIO()
    builder.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("code", [0x41, 0xF041])
def test_symbolic_truetype_resolves_through_the_3_0_subtable(code: int) -> None:
    """9.6.6.4: with a (3, 0) subtable "each byte from the string shall be
    prepended with the high byte of the range".

    The subtable's 0xF0xx keys are not Unicode scalars, so treating them as
    such made every code miss and resolve to GID 0 -- every glyph .notdef.
    """
    program = TrueTypeFontProgram(internal_symbol_font(), use_cmap=True)

    assert program.glyph_id_for_code(code) == 1
