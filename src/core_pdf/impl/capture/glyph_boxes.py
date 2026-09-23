# SPDX-License-Identifier: AGPL-3.0-only

"""Text-space to device-space box helpers shared by the geometry passes.

These are what the split-unicode fallback in capture_glyphs still needs;
the ink box its sibling used to compute is the compiled kernel's job now.
"""

from __future__ import annotations

from core_pdf.impl.model.geometry import transform_bbox
from core_pdf.impl.types import Rectangle

TextBasis = tuple[float, float, float, float, float, float]


def text_basis_rect(x0: float, y0: float, x1: float, y1: float, text_basis: TextBasis) -> Rectangle:
    base_x, base_y, a, b, c, d = text_basis
    return transform_bbox((x0, y0, x1, y1), (a, b, c, d, base_x, base_y))


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
