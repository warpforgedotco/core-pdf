# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from core_adobe_fonts.afm import core14
from core_adobe_fonts.afm.core14 import Core14FontMetrics
from core_pdf.impl.fonts_cmap import CMapDecoder
from core_pdf.impl.fonts_helpers import LIGATURE_TEXT_OVERRIDES
from core_pdf.impl.fonts_widths import (
    effective_descriptor,
    parse_font_widths,
    recover_descendant,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_float_strict,
    parse_int_strict,
    require_pdf_number,
)
from core_pdf_spec.s_09_fonts import metrics

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
    return metrics.standard_14_widths(canonical, literal)


def parse_font_metrics(
    font_dict: dict[str, Any],
    subtype: str | None,
    base_font_name: str | None,
    widths: Mapping[int, float],
) -> tuple[float, float]:
    ascent, descent = 800.0, -200.0
    descriptor = effective_descriptor(font_dict, subtype)
    if subtype == "Type3" and not isinstance(descriptor, dict):
        font_bbox = font_dict.get("FontBBox")
        if isinstance(font_bbox, (list, tuple)) and len(font_bbox) >= 4:
            with contextlib.suppress(ValueError):
                bbox_descent = parse_float_strict(font_bbox[1], "invalid Type3 FontBBox")
                bbox_ascent = parse_float_strict(font_bbox[3], "invalid Type3 FontBBox")
                descent, ascent = bbox_descent, bbox_ascent

    if base_font_name in FONT_DATA and not widths:
        entry = FONT_DATA[base_font_name]
        props = entry["props"]
        ascent_value = props.get("Ascent")
        if ascent_value is not None:
            ascent = require_pdf_number(ascent_value, "invalid font Ascent")
        descent_value = props.get("Descent")
        if descent_value is not None:
            descent = require_pdf_number(descent_value, "invalid font Descent")

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
    name: core14.FONT_DATA[record] for name, record in METRIC_RECORD_NAMES.items()
}


def font_is_vertical(
    font: dict[str, Any],
    subtype: str | None,
    base_encoding: str | None,
    base_font_name: str | None,
    cmap: CMapDecoder | None,
) -> bool:
    if (
        base_encoding == "V"
        or (base_encoding and base_encoding.endswith("-V"))
        or (base_font_name and base_font_name.endswith("-V"))
        or (cmap is not None and cmap.wmode == 1)
    ):
        return True
    descendant = recover_descendant(font) if subtype == "Type0" else None
    if descendant is None:
        return False
    wmode = descendant.get("WMode")
    if wmode is None:
        wmode = font.get("WMode", 0)
    try:
        return parse_int_strict(wmode, "invalid font WMode") == 1
    except ValueError:
        return False


class FontMetricsModel:
    __slots__ = (
        "widths",
        "default_width",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
        "is_vertical",
        "ascent",
        "descent",
    )

    widths: Mapping[int, float]
    default_width: float
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]
    is_vertical: bool
    ascent: float
    descent: float

    def __init__(
        self,
        font: dict[str, Any],
        subtype: str | None,
        *,
        base_encoding: str | None,
        cmap: CMapDecoder | None,
        encoding_decode_table: tuple[str, ...],
        base_font_name: str | None,
        is_cid_font: bool,
        is_type3: bool,
    ) -> None:
        font_metrics = parse_font_widths(font, subtype)
        widths = font_metrics.widths
        default_width = font_metrics.default_width
        is_vertical = font_is_vertical(font, subtype, base_encoding, base_font_name, cmap)
        ascent, descent = parse_font_metrics(font, subtype, base_font_name, widths)
        if is_type3:
            widths = adjust_type3_widths(font, widths)
        if not widths and not is_cid_font and not is_type3:
            builtin = standard_14_widths(base_font_name, encoding_decode_table)
            if builtin is not None:
                widths = builtin
                if font.get("MissingWidth") is None:
                    default_width = 0.0
        self.widths = widths
        self.default_width = default_width
        self.default_vertical_displacement_y = font_metrics.default_vertical_displacement_y
        self.default_vertical_origin_y = font_metrics.default_vertical_origin_y
        self.vertical_metrics = font_metrics.vertical_metrics
        self.is_vertical = is_vertical
        self.ascent = ascent
        self.descent = descent

    def glyph_width(self, code: int) -> float:
        return self.widths.get(code, self.default_width)

    def vertical_glyph_metric(self, code: int) -> tuple[float, float, float]:
        metric = self.vertical_metrics.get(code)
        if metric is None:
            metric = (
                self.default_vertical_displacement_y,
                self.glyph_width(code) / 2.0,
                self.default_vertical_origin_y,
            )
        return metric

    def vertical_glyph_position(self, code: int, *, font_size: float) -> tuple[float, float]:
        metric = self.vertical_glyph_metric(code)
        scale = font_size / 1000.0
        return (-metric[1] * scale, -metric[2] * scale)

    def glyph_advance_vector(
        self,
        code: int,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        encoded_space: bool,
    ) -> tuple[float, float]:
        width = self.vertical_glyph_metric(code)[0] if self.is_vertical else self.glyph_width(code)
        return metrics.glyph_advance_vector(
            width,
            vertical=self.is_vertical,
            font_size=font_size,
            char_space=char_space,
            word_space=word_space,
            horizontal_scale=horizontal_scale,
            encoded_space=encoded_space,
        )
