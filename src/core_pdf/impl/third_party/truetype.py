"""Compatibility imports for the standalone font-program package."""

from core_font_programs.impl.truetype import (
    Point,
    TrueTypeFontProgram,
    _invert_unicode_cmap,
    rasterize_contours,
    tt_font_for_data,
)

__all__ = (
    "Point",
    "TrueTypeFontProgram",
    "rasterize_contours",
    "tt_font_for_data",
)
