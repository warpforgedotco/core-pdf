# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import replace
from typing import Any

from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_float,
    parse_float_strict,
    parse_int_strict,
)
from core_pdf_spec.s_09_fonts.widths import (
    MAX_CID,
    MIN_CID,
    CompactCIDWidthMap,
    FontMetrics,
)
from core_pdf_spec.s_09_fonts.widths import parse_font_widths as pdf_font_widths


def get_descendant(font: dict[Any, Any]) -> dict[Any, Any] | None:
    descendant_fonts = font.get("DescendantFonts")
    if isinstance(descendant_fonts, (list, tuple)) and descendant_fonts:
        candidate = descendant_fonts[0]
        if isinstance(candidate, dict):
            return candidate
    return None


def recover_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    widths: Mapping[int, float] = {}
    missing_width = font.get("MissingWidth")
    if missing_width is None:
        default_width = 1000.0
        default_width_explicit = False
    else:
        default_width = parse_float(missing_width, 1000.0)
        default_width_explicit = True
    default_vertical_displacement_y = -1000.0
    default_vertical_origin_y = 880.0
    vertical_metrics: dict[int, tuple[float, float, float]] = {}
    descriptor = font.get("FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_dw = descendant.get("DW")
            if descendant_dw is not None:
                default_width = parse_float(descendant_dw, default_width)
                default_width_explicit = True
            dw2 = descendant.get("DW2")
            if isinstance(dw2, (list, tuple)) and len(dw2) >= 2:
                default_vertical_origin_y = parse_float(dw2[0], 880.0)
                default_vertical_displacement_y = parse_float(dw2[1], -1000.0)
            w2 = descendant.get("W2")
            if isinstance(w2, (list, tuple)):
                index = 0
                while index + 1 < len(w2):
                    try:
                        first = parse_int_strict(w2[index], "invalid CID vertical widths")
                    except ValueError:
                        index += 1
                        continue
                    values = w2[index + 1]
                    if isinstance(values, (list, tuple)):
                        for offset in range(len(values) // 3):
                            cid = first + offset
                            if MIN_CID <= cid <= MAX_CID:
                                vertical_metrics[cid] = (
                                    parse_float(
                                        values[offset * 3], default_vertical_displacement_y
                                    ),
                                    parse_float(values[offset * 3 + 1], 0.0),
                                    parse_float(values[offset * 3 + 2], 0.0),
                                )
                        index += 2
                    else:
                        if index + 4 < len(w2):
                            try:
                                last = parse_int_strict(values, "invalid CID vertical widths")
                                width = parse_float(w2[index + 2], default_vertical_displacement_y)
                                vx = parse_float(w2[index + 3], 0.0)
                                vy = parse_float(w2[index + 4], 0.0)
                                bounds = clipped_cid_bounds(first, last)
                                if bounds is not None:
                                    clipped_first, clipped_last = bounds
                                    for cid in range(clipped_first, clipped_last + 1):
                                        vertical_metrics[cid] = (width, vx, vy)
                            except ValueError:
                                pass
                        index += 5
            widths = parse_cid_widths(descendant.get("W"))
            descriptor = descendant.get("FontDescriptor")

    if isinstance(descriptor, dict) and subtype != "Type0":
        desc_missing_width = descriptor.get("MissingWidth")
        if desc_missing_width is not None:
            default_width = parse_float(desc_missing_width, default_width)
            default_width_explicit = True

    if subtype != "Type0":
        first_char_val = font.get("FirstChar")
        if first_char_val is None:
            first_char = 0
        else:
            try:
                first_char = parse_int_strict(first_char_val, "invalid font FirstChar")
            except ValueError:
                first_char = 0
        last_char_val = font.get("LastChar")
        last_char = None
        if last_char_val is not None:
            try:
                last_char = parse_int_strict(last_char_val, "invalid font LastChar")
            except ValueError:
                last_char = None
        font_widths = font.get("Widths")
        if isinstance(font_widths, (list, tuple)):
            sparse_widths: dict[int, float] = {}
            for index, width in enumerate(font_widths):
                code = first_char + index
                if last_char is not None and code > last_char:
                    break
                sparse_widths[code] = parse_float(width, default_width)
            widths = sparse_widths
        elif font_widths is not None:
            raise ValueError("invalid font widths array")
    return FontMetrics(
        widths=widths,
        default_width=default_width,
        default_width_explicit=default_width_explicit,
        default_vertical_displacement_y=default_vertical_displacement_y,
        default_vertical_origin_y=default_vertical_origin_y,
        vertical_metrics=vertical_metrics,
    )


def parse_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    if font.get("MissingWidth") is not None:
        return recover_font_widths(font, subtype)
    try:
        metrics = pdf_font_widths(font, subtype)
    except ValueError, TypeError, IndexError:
        return recover_font_widths(font, subtype)
    if subtype == "Type0":
        return metrics
    if not metrics.default_width_explicit:
        return replace(metrics, default_width=1000.0)
    return metrics


def clipped_cid_bounds(first: int, last: int) -> tuple[int, int] | None:
    if last < first or last < MIN_CID or first > MAX_CID:
        return None
    return (max(first, MIN_CID), min(last, MAX_CID))


def require_cid_int(value: Any, message: str) -> int:
    if type(value) is int:
        return value
    if type(value) is bool:
        raise ValueError(message)
    if type(value) is float and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    if isinstance(value, bytes):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError(message)


def parse_cid_widths(value: Any) -> Mapping[int, float]:
    if value is None:
        return {}
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    if len(value) == 2 and type(value[0]) is int:
        contiguous_widths = value[1]
        if isinstance(contiguous_widths, (list, tuple)):
            first = value[0]
            bounds = clipped_cid_bounds(
                first,
                first + len(contiguous_widths) - 1,
            )
            if bounds is None:
                return {}
            clipped_first, clipped_last = bounds
            offset = clipped_first - first
            count = clipped_last - clipped_first + 1
            with suppress(ValueError):
                return CompactCIDWidthMap(
                    clipped_first,
                    tuple(
                        parse_float_strict(width, "invalid CID widths array")
                        for width in contiguous_widths[offset : offset + count]
                    ),
                )
    widths: dict[int, float] = {}
    index = 0
    while index < len(value):
        try:
            first = require_cid_int(value[index], "invalid CID widths array")
        except ValueError:
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        nxt = value[index]
        if isinstance(nxt, (list, tuple)):
            code = first
            for w in nxt:
                if MIN_CID <= code <= MAX_CID:
                    with suppress(ValueError):
                        widths[code] = parse_float_strict(w, "invalid CID widths array")
                code += 1
            index += 1
        else:
            if index + 1 >= len(value):
                break
            try:
                last = require_cid_int(nxt, "invalid CID widths array")
                width = parse_float_strict(value[index + 1], "invalid CID widths array")
            except ValueError:
                index += 2
                continue
            bounds = clipped_cid_bounds(first, last)
            if bounds is None:
                index += 2
                continue
            clipped_first, clipped_last = bounds
            for i in range(clipped_first, clipped_last + 1):
                widths[i] = width
            index += 2
    return widths
