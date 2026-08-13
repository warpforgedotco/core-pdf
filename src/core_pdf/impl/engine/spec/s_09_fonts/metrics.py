# SPDX-License-Identifier: AGPL-3.0-only
"""Native PDF font metric helpers."""

from __future__ import annotations

import contextlib
from typing import Any

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.cmap_widths import (
    FontWidthMap,
    SparseFontWidthMap,
    scale_font_widths,
)
from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import FONT_DATA
from core_pdf.impl.engine.spec.s_09_fonts.helpers import LIGATURE_TEXT_OVERRIDES
from core_pdf.impl.engine.spec.s_09_fonts.widths import (
    get_descendant,
    require_font_float,
)

LIGATURE_TEXT_TO_CHAR = {text: char for char, text in LIGATURE_TEXT_OVERRIDES.items()}


def standard_14_widths(
    base_font_name: str | None, decode_table: tuple[str, ...] | None
) -> FontWidthMap | None:
    """Return built-in glyph widths for a standard 14 font, if this is one.

    9.6.2.2 lets the standard 14 fonts omit FirstChar, LastChar, Widths and
    FontDescriptor entirely, and requires a conforming reader to supply the
    metrics itself -- the special treatment is deprecated for writers from
    PDF 1.5 but readers "shall still provide" it. Without this, every glyph in
    such a font falls back to MissingWidth and text advances at a full em,
    which stretches a line of Times to roughly twice its true width.

    The shipped tables are keyed by the character a code denotes, so the
    font's own encoding is what turns them into a per-code map.
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
        if width is None:
            # The tables are keyed by the character the glyph draws, and
            # decoding has already expanded ligatures, so "fi" has to be
            # folded back to find the single glyph's advance.
            ligature = LIGATURE_TEXT_TO_CHAR.get(text)
            if ligature is not None:
                width = char_widths.get(ligature)
        if width is not None:
            sparse[code] = float(width)
    if not sparse:
        return None
    return SparseFontWidthMap(sparse)


def parse_font_metrics(
    font_dict: dict[str, Any],
    subtype: str | None,
    base_font_name: str | None,
    widths: FontWidthMap,
) -> tuple[float, float]:
    ascent, descent = 800.0, -200.0
    descriptor = lookup_dict_key(font_dict, "FontDescriptor")
    if subtype == "Type3" and not isinstance(descriptor, dict):
        # Type 3 fonts need not have a FontDescriptor. Their FontBBox is in
        # glyph space and supplies the vertical metrics directly; using the
        # generic Latin fallback shifts every layout box below its baseline.
        font_bbox = lookup_dict_key(font_dict, "FontBBox")
        if isinstance(font_bbox, (list, tuple)) and len(font_bbox) >= 4:
            with contextlib.suppress(ValueError):
                descent = require_font_float(font_bbox[1], "invalid Type3 FontBBox")
                ascent = require_font_float(font_bbox[3], "invalid Type3 FontBBox")
    if subtype == "Type0":
        descendant = get_descendant(font_dict)
        if isinstance(descendant, dict):
            desc_descriptor = lookup_dict_key(descendant, "FontDescriptor")
            descriptor = desc_descriptor or descriptor

    if base_font_name in FONT_DATA and not widths:
        entry = FONT_DATA[base_font_name]
        props = entry["props"]
        ascent_value = props.get("Ascent")
        if ascent_value is not None:
            if type(ascent_value) is not int and type(ascent_value) is not float:
                raise ValueError("invalid font Ascent")
            ascent = float(ascent_value)
        descent_value = props.get("Descent")
        if descent_value is not None:
            if type(descent_value) is not int and type(descent_value) is not float:
                raise ValueError("invalid font Descent")
            descent = float(descent_value)

    if isinstance(descriptor, dict):
        descriptor_ascent = lookup_dict_key(descriptor, "Ascent")
        if descriptor_ascent is not None:
            with contextlib.suppress(ValueError):
                ascent = require_font_float(descriptor_ascent, "invalid font Ascent")
        descriptor_descent = lookup_dict_key(descriptor, "Descent")
        if descriptor_descent is not None:
            with contextlib.suppress(ValueError):
                descent = require_font_float(descriptor_descent, "invalid font Descent")
    # ISO 32000 defines Descent below the baseline and therefore as non-positive.
    # Some PDF producers (notably PScript5.dll) serialize its magnitude instead.
    # Normalize that widespread malformed form once at the font boundary so every
    # geometry consumer sees a consistent text coordinate system.
    if descent > 0:
        descent = -descent
    return ascent, descent


def adjust_type3_widths(font_dict: dict[str, Any], widths: FontWidthMap) -> FontWidthMap:
    font_matrix = lookup_dict_key(font_dict, "FontMatrix")
    if isinstance(font_matrix, (list, tuple)) and len(font_matrix) >= 1:
        try:
            fm_a = require_font_float(font_matrix[0], "invalid FontMatrix")
        except ValueError:
            fm_a = 0.001
    else:
        fm_a = 0.001
    width_scale = fm_a * 1000.0
    if abs(width_scale - 1.0) > 1e-6:
        return scale_font_widths(widths, width_scale)
    return widths
