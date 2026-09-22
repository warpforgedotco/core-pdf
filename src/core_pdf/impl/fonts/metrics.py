# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from core_adobe_fonts.afm.core14 import FONT_DATA as PDF_FONT_DATA
from core_adobe_fonts.afm.core14 import Core14FontMetrics
from core_pdf.impl.fonts.helpers import LIGATURE_TEXT_OVERRIDES
from core_pdf.impl.fonts.widths import get_descendant
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict
from core_pdf_spec.s_09_fonts.metrics import standard_14_widths as pdf_standard_14_widths

LIGATURE_TEXT_TO_CHAR = {text: char for char, text in LIGATURE_TEXT_OVERRIDES.items()}


def standard_14_widths(
    base_font_name: str | None, decode_table: tuple[str, ...] | None
) -> Mapping[int, float] | None:
    literal = (
        tuple(LIGATURE_TEXT_TO_CHAR.get(text, text) for text in decode_table)
        if decode_table is not None
        else None
    )
    canonical = METRIC_RECORD_NAMES.get(base_font_name, base_font_name) if base_font_name else None
    return pdf_standard_14_widths(canonical, literal)


def parse_font_metrics(
    font_dict: dict[str, Any],
    subtype: str | None,
    base_font_name: str | None,
    widths: Mapping[int, float],
) -> tuple[float, float]:
    ascent, descent = 800.0, -200.0
    descriptor = font_dict.get("FontDescriptor")
    if subtype == "Type3" and not isinstance(descriptor, dict):
        font_bbox = font_dict.get("FontBBox")
        if isinstance(font_bbox, (list, tuple)) and len(font_bbox) >= 4:
            with contextlib.suppress(ValueError):
                bbox_descent = parse_float_strict(font_bbox[1], "invalid Type3 FontBBox")
                bbox_ascent = parse_float_strict(font_bbox[3], "invalid Type3 FontBBox")
                descent, ascent = bbox_descent, bbox_ascent
    if subtype == "Type0":
        descendant = get_descendant(font_dict)
        if isinstance(descendant, dict):
            desc_descriptor = descendant.get("FontDescriptor")
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

    descriptor_descent_applied = False
    if isinstance(descriptor, dict):
        descriptor_ascent = descriptor.get("Ascent")
        if descriptor_ascent is not None:
            with contextlib.suppress(ValueError):
                ascent = parse_float_strict(descriptor_ascent, "invalid font Ascent")
        descriptor_descent = descriptor.get("Descent")
        if descriptor_descent is not None:
            with contextlib.suppress(ValueError):
                descent = parse_float_strict(descriptor_descent, "invalid font Descent")
                descriptor_descent_applied = True
    if descriptor_descent_applied and descent > 0:
        descent = -descent
    return ascent, descent


def adjust_type3_widths(
    font_dict: dict[str, Any], widths: Mapping[int, float]
) -> Mapping[int, float]:
    font_matrix = font_dict.get("FontMatrix")
    if isinstance(font_matrix, (list, tuple)) and len(font_matrix) >= 1:
        try:
            fm_a = parse_float_strict(font_matrix[0], "invalid FontMatrix")
        except ValueError:
            fm_a = 0.001
    else:
        fm_a = 0.001
    width_scale = fm_a * 1000.0
    if abs(width_scale - 1.0) > 1e-6:
        return {code: width * width_scale for code, width in widths.items()}
    return widths


METRIC_RECORD_NAMES: dict[str, str] = {
    "Arial": "Helvetica",
    "Arial,Bold": "Helvetica-Bold",
    "Arial,BoldItalic": "Helvetica-BoldOblique",
    "Arial,Italic": "Helvetica-Oblique",
    "Courier": "Courier",
    "Courier-Bold": "Courier-Bold",
    "Courier-BoldOblique": "Courier-BoldOblique",
    "Courier-Oblique": "Courier-Oblique",
    "CourierNew": "Courier",
    "CourierNew,Bold": "Courier-Bold",
    "CourierNew,BoldItalic": "Courier-BoldOblique",
    "CourierNew,Italic": "Courier-Oblique",
    "Helvetica": "Helvetica",
    "Helvetica-Bold": "Helvetica-Bold",
    "Helvetica-BoldOblique": "Helvetica-BoldOblique",
    "Helvetica-Oblique": "Helvetica-Oblique",
    "Symbol": "Symbol",
    "Times-Bold": "Times-Bold",
    "Times-BoldItalic": "Times-BoldItalic",
    "Times-Italic": "Times-Italic",
    "Times-Roman": "Times-Roman",
    "TimesNewRoman": "Times-Roman",
    "TimesNewRoman,Bold": "Times-Bold",
    "TimesNewRoman,BoldItalic": "Times-BoldItalic",
    "TimesNewRoman,Italic": "Times-Italic",
    "ZapfDingbats": "ZapfDingbats",
}
FONT_DATA: dict[str, Core14FontMetrics] = {
    name: PDF_FONT_DATA[record] for name, record in METRIC_RECORD_NAMES.items()
}
