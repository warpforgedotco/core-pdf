# SPDX-License-Identifier: AGPL-3.0-only
"""Font conformance rules from ISO 32000-1 clauses 9.6 and 9.7.

Each test names the clause it pins.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
    internal_best_unicode_gid_cmap,
)

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


class internal_SymbolFont:
    """The same stand-in, as a class, so ``font["cmap"]`` subscripts properly."""

    def __init__(self, cmap: dict[int, str]) -> None:
        self.cmap = cmap
        self.reverse = {name: gid for gid, name in enumerate([".notdef", *sorted(cmap.values())])}

    def __getitem__(self, key: str) -> Any:
        assert key == "cmap"
        return SimpleNamespace(
            getBestCmap=lambda: None,
            getcmap=lambda platform, encoding: (
                SimpleNamespace(cmap=self.cmap) if (platform, encoding) == (3, 0) else None
            ),
        )

    def getReverseGlyphMap(self) -> dict[str, int]:
        return self.reverse


@pytest.mark.parametrize("code", [0x41, 0xF041])
def test_symbolic_truetype_registers_the_single_byte_code(code: int) -> None:
    """9.6.6.4: with a (3, 0) subtable "each byte from the string shall be
    prepended with the high byte of the range", the ranges being 0x0000-0x00FF,
    0xF000-0xF0FF, 0xF100-0xF1FF and 0xF200-0xF2FF.

    Those 0xF0xx keys are not Unicode scalars, but they pass a scalar test as
    private-use codepoints and were stored as if they were, which suppressed
    the fallback that registers the single-byte aliases. Every code then missed
    and resolved to GID 0 -- every glyph .notdef.
    """
    mapping = internal_best_unicode_gid_cmap(cast(Any, internal_SymbolFont({0xF041: "sq"})))

    assert mapping[code] == 1


def test_unicode_cmap_gains_no_single_byte_aliases() -> None:
    """The alias only applies to the (3, 0) symbol fallback, not to a real
    Unicode subtable, where a 0xF0xx key genuinely is a private-use codepoint."""

    class internal_UnicodeFont(internal_SymbolFont):
        def __getitem__(self, key: str) -> Any:
            return SimpleNamespace(getBestCmap=lambda: self.cmap, getcmap=lambda p, e: None)

    mapping = internal_best_unicode_gid_cmap(cast(Any, internal_UnicodeFont({0xF041: "sq"})))

    assert mapping == {0xF041: 1}
