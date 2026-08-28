# SPDX-License-Identifier: AGPL-3.0-only
"""Pin the Annex D.2 base encodings to the sources they were derived from.

The tables are close enough to cp1252 and Mac OS Roman that decoding with the
code pages looks right until it silently is not, so every divergence from them
is asserted here by hand rather than left to a round-trip.
"""

from __future__ import annotations

import pytest

from core_pdf._vendor.fontTools.encodings.MacRoman import MacRoman
from core_pdf._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.spec.s_09_fonts.helpers import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    MAC_ROMAN_ENCODING_TABLE,
    STANDARD_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_TABLE,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_TABLE,
)

# Annex D.2 assigns these codes differently from the code page they otherwise
# track. Each entry is (code, code-page text, Annex D.2 text).
WIN_ANSI_DIVERGENCES = (
    (0xA0, " ", " "),
    (0xAD, "­", "-"),
    (0x7F, "", "•"),
    (0x81, "", "•"),
    (0x8D, "", "•"),
    (0x8F, "", "•"),
    (0x90, "", "•"),
    (0x9D, "", "•"),
)

MAC_ROMAN_DIVERGENCES = (
    (0xCA, " ", " "),
    (0xDB, "€", "¤"),
    (0xF0, "", ""),
    (0xBD, "Ω", "Ω"),
    (0x7F, "\x7f", ""),
)


def test_standard_encoding_matches_vendored_font_tools_table() -> None:
    for code, name in enumerate(StandardEncoding):
        expected = "" if name in (None, ".notdef", "") else glyph_name_to_unicode(name)
        assert STANDARD_ENCODING[code] == expected, f"code {code:#04x}"


def test_mac_roman_encoding_matches_vendored_table_except_annex_d_corrections() -> None:
    corrections = {code: pdf for code, _codepage, pdf in MAC_ROMAN_DIVERGENCES}
    for code, name in enumerate(MacRoman):
        # fontTools reuses 0-037 of its MacRoman list for the Mac glyph
        # ordering rather than for character codes; Annex D.2 leaves the
        # control range undefined.
        if code < 32 or code in corrections:
            continue
        expected = "" if name in (None, ".notdef", "") else glyph_name_to_unicode(name)
        assert MAC_ROMAN_ENCODING[code] == expected, f"code {code:#04x}"


def test_win_ansi_encoding_matches_cp1252_except_annex_d_corrections() -> None:
    corrections = {code for code, _codepage, _pdf in WIN_ANSI_DIVERGENCES}
    for code in range(32, 256):
        if code in corrections:
            continue
        expected = bytes([code]).decode("cp1252")
        assert WIN_ANSI_ENCODING[code] == expected, f"code {code:#04x}"


@pytest.mark.parametrize(("code", "codepage", "pdf"), WIN_ANSI_DIVERGENCES)
def test_win_ansi_encoding_prefers_annex_d_over_cp1252(code: int, codepage: str, pdf: str) -> None:
    assert WIN_ANSI_ENCODING[code] == pdf
    assert WIN_ANSI_ENCODING_TABLE[code] == pdf
    assert pdf != codepage


@pytest.mark.parametrize(("code", "codepage", "pdf"), MAC_ROMAN_DIVERGENCES)
def test_mac_roman_encoding_prefers_annex_d_over_the_code_page(
    code: int, codepage: str, pdf: str
) -> None:
    assert MAC_ROMAN_ENCODING[code] == pdf
    assert MAC_ROMAN_ENCODING_TABLE[code] == pdf
    assert pdf != codepage


def test_standard_encoding_defines_the_upper_range_annex_d_lists() -> None:
    # The previous table stopped at 0267 and misplaced everything it did hold,
    # so spot-check both ends of the range it never reached.
    assert STANDARD_ENCODING[0xA1] == "¡"  # exclamdown
    assert STANDARD_ENCODING[0xA4] == "⁄"  # fraction
    assert STANDARD_ENCODING[0xA9] == "'"  # quotesingle
    assert STANDARD_ENCODING[0xD0] == "—"  # emdash
    assert STANDARD_ENCODING[0xF5] == "ı"  # dotlessi
    assert STANDARD_ENCODING[0xFB] == "ß"  # germandbls
    assert STANDARD_ENCODING[0xA0] == ""  # undefined


def test_decode_tables_expand_ligatures_and_leave_undefined_codes_empty() -> None:
    assert STANDARD_ENCODING[0xAE] == "ﬁ"
    assert STANDARD_ENCODING_TABLE[0xAE] == "fi"
    assert MAC_ROMAN_ENCODING_TABLE[0xDE] == "fi"
    assert STANDARD_ENCODING_TABLE[0xA0] == ""


def test_decode_tables_keep_raw_values_for_the_control_range() -> None:
    for table in (STANDARD_ENCODING_TABLE, WIN_ANSI_ENCODING_TABLE, MAC_ROMAN_ENCODING_TABLE):
        assert table[0x00] == "\x00"
        assert table[0x0C] == "\x0c"


def test_glyph_name_tables_do_not_invent_names_for_undefined_controls() -> None:
    for table in (
        STANDARD_ENCODING_GLYPH_NAMES,
        WIN_ANSI_ENCODING_GLYPH_NAMES,
        MAC_ROMAN_ENCODING_GLYPH_NAMES,
    ):
        assert table[0] == ".notdef"
        assert table[0x0C] == ".notdef"


def test_glyph_name_tables_preserve_annex_d_names() -> None:
    assert STANDARD_ENCODING_GLYPH_NAMES[0x27] == "quoteright"
    assert WIN_ANSI_ENCODING_GLYPH_NAMES[0x80] == "Euro"
    assert WIN_ANSI_ENCODING_GLYPH_NAMES[0xB2] == "twosuperior"
    assert MAC_ROMAN_ENCODING_GLYPH_NAMES[0xBD] == "Omega"
    assert MAC_ROMAN_ENCODING_GLYPH_NAMES[0xDB] == "currency"
