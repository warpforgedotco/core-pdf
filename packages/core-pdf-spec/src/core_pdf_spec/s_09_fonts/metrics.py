"""Prescribed PDF font metrics and text-space displacement arithmetic."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf_spec.s_09_fonts.data.core14 import FONT_DATA


def standard_14_widths(
    base_font_name: str | None, decode_table: tuple[str, ...] | None
) -> Mapping[int, float] | None:
    """Return built-in glyph widths for a standard 14 font, if this is one.

    9.6.2.2 lets the standard 14 fonts omit FirstChar, LastChar, Widths and
    FontDescriptor entirely, and requires a conforming reader to supply the
    metrics itself -- the special treatment is deprecated for writers from
    PDF 1.5 but readers "shall still provide" it. Without this, every glyph in
    such a font falls back to MissingWidth and text advances at a full em,
    which stretches a line of Times to roughly twice its true width.

    The supplied mapping contains literal encoded glyph characters.
    """
    if base_font_name is None or decode_table is None:
        return None
    entry = FONT_DATA.get(base_font_name)
    if not isinstance(entry, dict):
        return None
    char_widths = entry.get("widths")
    if not isinstance(char_widths, dict):
        return None
    sparse: dict[int, float] = {}
    for code in range(min(len(decode_table), 256)):
        text = decode_table[code]
        width = char_widths.get(text)
        if width is not None:
            sparse[code] = float(width)
    if not sparse:
        return None
    return sparse


def glyph_advance_vector(
    width: float,
    *,
    vertical: bool,
    font_size: float,
    char_space: float,
    word_space: float,
    horizontal_scale: float,
    encoded_space: bool,
) -> tuple[float, float]:
    spacing = char_space + (word_space if encoded_space else 0.0)
    displacement = width * font_size / 1000.0 + spacing
    return (0.0, displacement) if vertical else (displacement * horizontal_scale / 100.0, 0.0)


def text_adjustment_vector(
    adjustment: float,
    *,
    vertical: bool,
    font_size: float,
    horizontal_scale: float,
) -> tuple[float, float]:
    """PDF 32000-1 9.4.4: TJ adjusts only the current writing direction.

    Horizontal scaling applies only to a horizontal displacement.
    """
    displacement = -adjustment * font_size / 1000.0
    return (0.0, displacement) if vertical else (displacement * horizontal_scale / 100.0, 0.0)


__all__ = [
    "standard_14_widths",
    "glyph_advance_vector",
    "text_adjustment_vector",
]
