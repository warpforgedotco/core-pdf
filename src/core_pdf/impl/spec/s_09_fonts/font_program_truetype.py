"""TrueType table decoding and exact glyph bounds."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf._vendor.fontTools.ttLib import TTFont


def parse_truetype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True)
    if not {"maxp", "glyf", "loca", "head"} <= set(font.keys()):
        raise ValueError("invalid TrueType glyph tables")
    return font


def symbol_character_code(codepoint: int) -> int:
    """PDF 9.6.6.4 selects symbolic TrueType glyphs with a single-byte code."""
    return codepoint & 0xFF if 0xF000 <= codepoint <= 0xF2FF else codepoint


def internal_fonttools_bbox(
    font: Any,
    glyph_id: int,
    scale: float,
) -> tuple[float, float, float, float] | None:
    glyph_name = font.getGlyphName(glyph_id)
    glyph_set = font.getGlyphSet()
    bounds_pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(TransformPen(bounds_pen, (scale, 0.0, 0.0, scale, 0.0, 0.0)))
    if bounds_pen.bounds is None:
        return None
    x_min, y_min, x_max, y_max = bounds_pen.bounds
    return float(x_min), float(y_min), float(x_max), float(y_max)


def is_unicode_scalar(codepoint: int) -> bool:
    return 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF


def internal_glyph_bbox(glyf: Any, glyph_name: str) -> tuple[float, float, float, float] | None:
    glyph = glyf[glyph_name]
    if glyph.numberOfContours == 0:
        return None
    if not all(hasattr(glyph, attr) for attr in ("xMin", "yMin", "xMax", "yMax")):
        glyph.recalcBounds(glyf)
    return (
        float(glyph.xMin),
        float(glyph.yMin),
        float(glyph.xMax),
        float(glyph.yMax),
    )
