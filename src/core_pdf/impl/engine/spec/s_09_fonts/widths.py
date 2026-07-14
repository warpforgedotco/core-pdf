# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_float, parse_int
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.third_party.cid.widths import (
    FontWidthMap,
    SparseFontWidthMap,
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


def parse_font_width(value: Any, default_width: float) -> float:
    if type(value) is float:
        return value
    if type(value) is int:
        return float(value)
    if value is None or type(value) is bool:
        return default_width
    parsed = parse_float(value, None)
    return default_width if parsed is None else parsed


def get_descendant(font: dict[Any, Any]) -> dict[Any, Any] | None:
    descendant_fonts = lookup_dict_key(font, "DescendantFonts")
    if isinstance(descendant_fonts, (list, tuple)) and descendant_fonts:
        candidate = descendant_fonts[0]
        if isinstance(candidate, dict):
            return candidate
    return None


def parse_font_widths(
    font: dict[Any, Any], subtype: str | None
) -> tuple[FontWidthMap, float, bool]:
    widths: FontWidthMap = SparseFontWidthMap()
    missing_width = lookup_dict_key(font, "MissingWidth")
    if missing_width is None:
        default_width = 1000.0
    else:
        default_width = parse_optional_font_float(missing_width, 1000.0)
    is_vertical = False
    descriptor = lookup_dict_key(font, "FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_dw = lookup_dict_key(descendant, "DW")
            if descendant_dw is not None:
                default_width = parse_optional_font_float(descendant_dw, default_width)
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

    if isinstance(descriptor, dict):
        desc_missing_width = lookup_dict_key(descriptor, "MissingWidth")
        if desc_missing_width is not None:
            default_width = parse_optional_font_float(desc_missing_width, default_width)

    if subtype == "Type0":
        return widths, default_width, is_vertical

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
            sparse_widths[code] = parse_font_width(width, default_width)
        widths = SparseFontWidthMap(sparse_widths)
    elif font_widths is not None:
        raise ValueError("invalid font widths array")
    return widths, default_width, is_vertical
