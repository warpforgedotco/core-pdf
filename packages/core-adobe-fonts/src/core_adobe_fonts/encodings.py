# SPDX-License-Identifier: AGPL-3.0-only
"""Standard, MacRoman, and WinAnsi glyph-name tables (256 entries each).

The name data derives from fontTools and the Adobe Glyph List (see
``_vendor/font_data``); the MacRoman and WinAnsi code assignments follow
ISO 32000-1 Annex D, which is where PDF consumers apply them.
"""

from __future__ import annotations

from core_adobe_fonts._vendor.font_data.encoding_names import (
    MAC_ROMAN_ENCODING_GLYPH_NAMES,
    STANDARD_ENCODING_GLYPH_NAMES,
    WIN_ANSI_ENCODING_GLYPH_NAMES,
)

__all__ = (
    "STANDARD_ENCODING_GLYPH_NAMES",
    "MAC_ROMAN_ENCODING_GLYPH_NAMES",
    "WIN_ANSI_ENCODING_GLYPH_NAMES",
)
