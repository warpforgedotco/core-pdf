# SPDX-License-Identifier: AGPL-3.0-only
"""The Unicode text of each code in the three base encodings (ISO 32000-2, Annex D).

Each table maps a code's glyph name through the Adobe Glyph List; an unused
code (.notdef) is the empty string.
"""

from __future__ import annotations

from collections.abc import Mapping

from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_adobe_fonts.encodings import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
)


def encoding_text(
    glyph_names: tuple[str, ...], overrides: Mapping[int, str] | None = None
) -> tuple[str, ...]:
    table = [GLYPH_DATA.get(name, "") for name in glyph_names]
    for code, text in (overrides or {}).items():
        table[code] = text
    return tuple(table)


STANDARD_ENCODING: tuple[str, ...] = encoding_text(STANDARD_ENCODING_GLYPH_NAMES)
WIN_ANSI_ENCODING: tuple[str, ...] = encoding_text(WIN_ANSI_ENCODING_GLYPH_NAMES)
# The Glyph List maps Omega to U+2126 OHM SIGN; MacRoman's 0xBD is
# U+03A9 GREEK CAPITAL LETTER OMEGA, as this table has always given it.
MAC_ROMAN_ENCODING: tuple[str, ...] = encoding_text(
    MAC_ROMAN_ENCODING_GLYPH_NAMES, {0xBD: "\u03a9"}
)


__all__ = ["STANDARD_ENCODING", "WIN_ANSI_ENCODING", "MAC_ROMAN_ENCODING"]
