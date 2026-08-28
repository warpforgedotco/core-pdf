# SPDX-License-Identifier: AGPL-3.0-only
"""Native font width parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.spec.s_09_fonts.cmap_widths import (
    FontWidthMap,
    SparseFontWidthMap,
    internal_clipped_cid_bounds,
    internal_MAX_CID,
    internal_MIN_CID,
    parse_cid_widths,
)


def require_font_int(value: Any, message: str) -> int:
    if type(value) is int:
        return value
    if type(value) is bool:
        raise ValueError(message)
    parsed = parse_int(value, None)
    if parsed is None:
        raise ValueError(message)
    return parsed


def require_font_float(value: Any, message: str) -> float:
    if type(value) is float:
        return value
    if type(value) is int:
        return float(value)
    if type(value) is bool:
        raise ValueError(message)
    parsed = parse_float(value, None)
    if parsed is None:
        raise ValueError(message)
    return parsed


def parse_optional_font_float(value: Any, default: float) -> float:
    if type(value) is float:
        return value
    if type(value) is int:
        return float(value)
    if value is None or type(value) is bool:
        return default
    parsed = parse_float(value, None)
    return default if parsed is None else parsed


def get_descendant(font: dict[Any, Any]) -> dict[Any, Any] | None:
    descendant_fonts = lookup_dict_key(font, "DescendantFonts")
    if isinstance(descendant_fonts, (list, tuple)) and descendant_fonts:
        candidate = descendant_fonts[0]
        if isinstance(candidate, dict):
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class FontMetrics:
    widths: FontWidthMap
    default_width: float
    is_vertical: bool
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]


def parse_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    widths: FontWidthMap = SparseFontWidthMap()
    missing_width = lookup_dict_key(font, "MissingWidth")
    if missing_width is None:
        default_width = 1000.0
    else:
        default_width = parse_optional_font_float(missing_width, 1000.0)
    is_vertical = False
    default_vertical_displacement_y = -1000.0
    default_vertical_origin_y = 880.0
    vertical_metrics: dict[int, tuple[float, float, float]] = {}
    descriptor = lookup_dict_key(font, "FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_dw = lookup_dict_key(descendant, "DW")
            if descendant_dw is not None:
                default_width = parse_optional_font_float(descendant_dw, default_width)
            dw2 = lookup_dict_key(descendant, "DW2")
            if isinstance(dw2, (list, tuple)) and len(dw2) >= 2:
                default_vertical_origin_y = parse_optional_font_float(dw2[0], 880.0)
                default_vertical_displacement_y = parse_optional_font_float(dw2[1], -1000.0)
            w2 = lookup_dict_key(descendant, "W2")
            if isinstance(w2, (list, tuple)):
                index = 0
                while index + 1 < len(w2):
                    try:
                        first = require_font_int(w2[index], "invalid CID vertical widths")
                    except ValueError:
                        index += 1
                        continue
                    values = w2[index + 1]
                    if isinstance(values, (list, tuple)):
                        for offset in range(0, len(values) // 3):
                            cid = first + offset
                            if internal_MIN_CID <= cid <= internal_MAX_CID:
                                vertical_metrics[cid] = (
                                    parse_optional_font_float(
                                        values[offset * 3], default_vertical_displacement_y
                                    ),
                                    parse_optional_font_float(values[offset * 3 + 1], 0.0),
                                    parse_optional_font_float(values[offset * 3 + 2], 0.0),
                                )
                        index += 2
                    else:
                        if index + 4 < len(w2):
                            try:
                                last = require_font_int(values, "invalid CID vertical widths")
                                width = parse_optional_font_float(
                                    w2[index + 2], default_vertical_displacement_y
                                )
                                vx = parse_optional_font_float(w2[index + 3], 0.0)
                                vy = parse_optional_font_float(w2[index + 4], 0.0)
                                bounds = internal_clipped_cid_bounds(first, last)
                                if bounds is not None:
                                    clipped_first, clipped_last = bounds
                                    for cid in range(clipped_first, clipped_last + 1):
                                        vertical_metrics[cid] = (width, vx, vy)
                            except ValueError:
                                pass
                        index += 5
            wmode = lookup_dict_key(descendant, "WMode")
            if wmode is None:
                wmode = lookup_dict_key(font, "WMode")
            if wmode is None:
                wmode = 0
            try:
                wmode_int = require_font_int(wmode, "invalid font WMode")
            except ValueError:
                wmode_int = 0
            is_vertical = wmode_int == 1
            widths = parse_cid_widths(lookup_dict_key(descendant, "W"))
            descriptor = lookup_dict_key(descendant, "FontDescriptor")

    if isinstance(descriptor, dict):
        desc_missing_width = lookup_dict_key(descriptor, "MissingWidth")
        if desc_missing_width is not None:
            default_width = parse_optional_font_float(desc_missing_width, default_width)

    if subtype == "Type0":
        return FontMetrics(
            widths=widths,
            default_width=default_width,
            is_vertical=is_vertical,
            default_vertical_displacement_y=default_vertical_displacement_y,
            default_vertical_origin_y=default_vertical_origin_y,
            vertical_metrics=vertical_metrics,
        )

    first_char_val = lookup_dict_key(font, "FirstChar")
    if first_char_val is None:
        first_char = 0
    else:
        try:
            first_char = require_font_int(first_char_val, "invalid font FirstChar")
        except ValueError:
            first_char = 0
    last_char_val = lookup_dict_key(font, "LastChar")
    last_char = None
    if last_char_val is not None:
        try:
            last_char = require_font_int(last_char_val, "invalid font LastChar")
        except ValueError:
            last_char = None
    font_widths = lookup_dict_key(font, "Widths")
    if isinstance(font_widths, (list, tuple)):
        sparse_widths: dict[int, float] = {}
        for index, width in enumerate(font_widths):
            code = first_char + index
            if last_char is not None and code > last_char:
                break
            sparse_widths[code] = parse_optional_font_float(width, default_width)
        widths = SparseFontWidthMap(sparse_widths)
    elif font_widths is not None:
        raise ValueError("invalid font widths array")
    return FontMetrics(
        widths=widths,
        default_width=default_width,
        is_vertical=is_vertical,
        default_vertical_displacement_y=default_vertical_displacement_y,
        default_vertical_origin_y=default_vertical_origin_y,
        vertical_metrics=vertical_metrics,
    )
