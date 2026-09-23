# SPDX-License-Identifier: AGPL-3.0-only

"""Text-space to device-space box helpers shared by the geometry passes."""

from __future__ import annotations

from core_pdf.impl.model.geometry import transform_bbox
from core_pdf.impl.types import Rectangle

TextBasis = tuple[float, float, float, float, float, float]


def text_basis_rect(x0: float, y0: float, x1: float, y1: float, text_basis: TextBasis) -> Rectangle:
    base_x, base_y, a, b, c, d = text_basis
    return transform_bbox((x0, y0, x1, y1), (a, b, c, d, base_x, base_y))


def glyph_ink_rect(
    glyph_bbox: Rectangle | None,
    advance_start: float,
    fallback_bbox: Rectangle,
    text_basis: TextBasis,
    text_advance_scale: float,
    rise: float,
    font_scale: float,
) -> Rectangle:
    if glyph_bbox is None:
        return fallback_bbox
    gx0, gy0, gx1, gy1 = glyph_bbox
    if gx1 <= gx0 or gy1 <= gy0:
        return fallback_bbox
    text_x0 = advance_start + gx0 * text_advance_scale
    text_x1 = advance_start + gx1 * text_advance_scale
    text_y0 = rise + gy0 * font_scale
    text_y1 = rise + gy1 * font_scale
    rect = text_basis_rect(text_x0, text_y0, text_x1, text_y1, text_basis)
    fallback_height = fallback_bbox[3] - fallback_bbox[1]
    fallback_width = fallback_bbox[2] - fallback_bbox[0]
    rect_x0, rect_y0, rect_x1, rect_y1 = rect
    rect_height = rect_y1 - rect_y0
    rect_width = rect_x1 - rect_x0
    if rect_width <= 0.01 or rect_height <= 0.01:
        return fallback_bbox
    if fallback_width > 0.0 and rect_width > fallback_width * 4.0:
        return fallback_bbox
    if fallback_height > 0.0 and rect_height > fallback_height * 1.5:
        return fallback_bbox
    return rect


def transformed_text_line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text_basis: TextBasis,
) -> tuple[float, float, float, float]:
    base_x, base_y, a, b, c, d = text_basis
    return (
        base_x + x0 * a + y0 * c,
        base_y + x0 * b + y0 * d,
        base_x + x1 * a + y1 * c,
        base_y + x1 * b + y1 * d,
    )


def glyph_text_space_boxes(
    offset: float,
    advance: float,
    *,
    is_vertical: bool,
    rise: float,
    font_ascent: float,
    font_descent: float,
    position: tuple[float, float] = (0.0, 0.0),
) -> tuple[
    Rectangle,
    tuple[float, float, float, float],
]:
    if is_vertical:
        position_x, position_y = position
        start_y = rise + position_y - offset
        end_y = start_y - advance
        ar = font_ascent
        dr = font_descent
        x0 = position_x + (min(dr, ar))
        x1 = position_x + (max(dr, ar))
        y0 = min(end_y, start_y)
        y1 = max(start_y, end_y)
        return (
            (x0, y0, x1, y1),
            (0.0, start_y, 0.0, end_y),
        )
    ar = font_ascent + rise
    dr = font_descent + rise
    return (
        (offset, dr, offset + advance, ar),
        (offset, rise, offset + advance, rise),
    )
