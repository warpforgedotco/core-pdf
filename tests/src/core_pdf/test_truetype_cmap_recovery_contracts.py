import struct
from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf._vendor.fontTools.ttLib.tables._c_m_a_p import CmapSubtable, table__c_m_a_p
from core_pdf.impl._impl.fonts import font_program_truetype as tt


def internal_font(tables):
    font = TTFont()
    font.setGlyphOrder([".notdef", "A", "B"])
    cmap = table__c_m_a_p()
    cmap.tableVersion = 0
    cmap.tables = []
    for platform, encoding, mapping in tables:
        table = CmapSubtable.newSubtable(4)
        table.platformID, table.platEncID, table.language = platform, encoding, 0
        table.cmap = mapping
        cmap.tables.append(table)
    font["cmap"] = cmap
    return font


@pytest.mark.parametrize("offset", [0, 0xF000, 0xF100, 0xF200])
def test_symbol_unicode_fallback_registers_byte_code_aliases(offset):
    with internal_font([(3, 0, {offset + 65: "A"})]) as font:
        assert tt.internal_best_unicode_gid_cmap(font) == {offset + 65: 1, 65: 1}


def test_unicode_cmap_precedes_symbol_fallback():
    with internal_font([(3, 0, {0xF041: "B"}), (3, 1, {65: "A"})]) as font:
        assert tt.internal_best_unicode_gid_cmap(font) == {65: 1}


@pytest.mark.parametrize("tables", [[], [(1, 0, {65: "A"})]])
def test_nonunicode_tables_do_not_become_unicode_mappings(tables):
    with internal_font(tables) as font:
        assert tt.internal_best_unicode_gid_cmap(font) == {}


def test_unicode_cmap_recovers_generated_glyph_names_but_rejects_unusable_entries():
    with internal_font(
        [
            (
                3,
                1,
                {
                    65: "glyph00007",
                    66: "glyphbad",
                    67: "missing",
                    68: ".notdef",
                    0xD800: "A",
                    0x110000: "B",
                },
            )
        ]
    ) as font:
        assert tt.internal_best_unicode_gid_cmap(font) == {65: 7}


@pytest.mark.parametrize("symbol", [None, {}, {65: ".notdef"}, {0xF041: "B"}])
def test_raw_cmap_prefers_usable_symbol_table_then_macintosh(symbol):
    tables = [(1, 0, {65: "A"})]
    if symbol is not None:
        tables.insert(0, (3, 0, symbol))
    with internal_font(tables) as font:
        expected = {0xF041: 2, 65: 2} if symbol == {0xF041: "B"} else {65: 1}
        assert tt.internal_code_gid_cmap(font) == expected


def test_raw_cmap_recovers_generated_names_and_preserves_explicit_byte_mapping():
    with internal_font(
        [
            (
                3,
                0,
                {
                    65: "A",
                    0xF041: "B",
                    66: "glyph00007",
                    67: "glyphbad",
                    68: "missing",
                    69: ".notdef",
                },
            )
        ]
    ) as font:
        assert tt.internal_code_gid_cmap(font) == {65: 1, 0xF041: 2, 66: 7}


@pytest.mark.parametrize("function", [tt.internal_code_gid_cmap, tt.internal_best_unicode_gid_cmap])
def test_missing_cmap_is_recoverable(function):
    with TTFont() as font:
        assert function(font) == {}


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        ("BA", "A"),
        ("!A", "A"),
        (" !", "!"),
        ("\ue000 ", " "),
        ("\x01\ue000", "\ue000"),
        ("\u200b\ue000", "\u200b"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_cmap_inversion_is_order_independent_and_prefers_readable_text(texts, expected, reverse):
    values = texts[::-1] if reverse else texts
    mapping = {ord(char): 7 for char in values}
    mapping.update({0xD800: 8, 0x110000: 9, 90: 0})
    assert tt.internal_invert_unicode_cmap(mapping) == {7: expected}


@pytest.mark.parametrize("contours", [-1, 1, 3])
def test_glyph_header_recovers_bounds_without_decompiling_contours(contours):
    data = struct.pack(">hhhhh", contours, -10, -20, 30, 40)
    assert tt.internal_glyph_header_bbox([0, 10], data, 0) == (-10, -20, 30, 40)


@pytest.mark.parametrize(
    ("locations", "gid"), [([0, 10], -1), ([0, 10], 1), ([0, 0], 0), ([0, 9], 0), ([0, 11], 0)]
)
def test_unusable_glyph_locations_have_no_bounds(locations, gid):
    assert tt.internal_glyph_header_bbox(locations, bytes(10), gid) is None


def test_contourless_glyph_has_no_bounds():
    assert tt.internal_glyph_header_bbox([0, 10], bytes(10), 0) is None


@pytest.mark.parametrize("count", [0, -1, 3])
def test_corrupt_post_order_recovers_only_positive_glyph_counts(count):
    class BrokenFont:
        def getGlyphOrder(self):
            raise ValueError("broken post")

        def __getitem__(self, key):
            return SimpleNamespace(numGlyphs=count)

        def setGlyphOrder(self, order):
            self.order = order

    font: Any = BrokenFont()
    if count > 0:
        tt.internal_ensure_glyph_order(font)
        assert font.order == [".notdef", "glyph00001", "glyph00002"]
    else:
        with pytest.raises(ValueError, match="invalid TrueType glyph order"):
            tt.internal_ensure_glyph_order(font)
