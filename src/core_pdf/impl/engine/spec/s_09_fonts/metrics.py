# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import FONT_DATA
from core_pdf.impl.engine.spec.s_09_fonts.widths import (
    get_descendant,
    require_font_float,
)
from core_pdf.impl.third_party.cid.widths import FontWidthMap, scale_font_widths


def parse_font_metrics(
    font_dict: dict[str, Any],
    subtype: str | None,
    base_font_name: str | None,
    widths: FontWidthMap,
) -> tuple[float, float]:
    ascent, descent = 800.0, -200.0
    descriptor = lookup_dict_key(font_dict, "FontDescriptor")
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
            try:
                ascent = require_font_float(descriptor_ascent, "invalid font Ascent")
            except ValueError:
                pass
        descriptor_descent = lookup_dict_key(descriptor, "Descent")
        if descriptor_descent is not None:
            try:
                descent = require_font_float(descriptor_descent, "invalid font Descent")
            except ValueError:
                pass
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
