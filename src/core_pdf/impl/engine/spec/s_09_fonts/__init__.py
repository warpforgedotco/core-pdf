# SPDX-License-Identifier: AGPL-3.0-only
"""PDF 9 text and font data structures."""

from core_pdf.impl.engine.spec.s_09_fonts.cff import cff_font_for_pdf_font
from core_pdf.impl.engine.spec.s_09_fonts.font_program import (
    CFFFont,
    CFFGlyphFeature,
    CFFUnicodeRepairIndex,
    cff_font_for_data,
    cff_unicode_repair_index_for_data,
    glyph_feature_distance,
    is_repairable_to_unicode_label,
)
from core_pdf.impl.engine.spec.s_09_fonts.truetype import (
    Point,
    TrueTypeFontProgram,
    rasterize_contours,
    tt_font_for_data,
)

__all__ = (
    "CFFFont",
    "CFFGlyphFeature",
    "CFFUnicodeRepairIndex",
    "Point",
    "TrueTypeFontProgram",
    "cff_font_for_data",
    "cff_font_for_pdf_font",
    "cff_unicode_repair_index_for_data",
    "glyph_feature_distance",
    "is_repairable_to_unicode_label",
    "rasterize_contours",
    "tt_font_for_data",
)
