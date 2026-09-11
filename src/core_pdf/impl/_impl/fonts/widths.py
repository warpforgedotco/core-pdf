# SPDX-License-Identifier: AGPL-3.0-only
"""Native font width parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from core_pdf.impl._impl.fonts.cmap_widths import (
    MAX_CID,
    MIN_CID,
    internal_clipped_cid_bounds,
    parse_cid_widths,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_float,
    parse_int_strict,
)
from core_pdf_spec.s_09_fonts.widths import FontMetrics
from core_pdf_spec.s_09_fonts.widths import parse_font_widths as pdf_font_widths


def get_descendant(font: dict[Any, Any]) -> dict[Any, Any] | None:
    descendant_fonts = font.get("DescendantFonts")
    if isinstance(descendant_fonts, (list, tuple)) and descendant_fonts:
        candidate = descendant_fonts[0]
        if isinstance(candidate, dict):
            return candidate
    return None


def internal_recover_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    widths: Mapping[int, float] = {}
    missing_width = font.get("MissingWidth")
    if missing_width is None:
        # Not the Table 122 default of 0: a font that omits MissingWidth and
        # also omits a code from /Widths is broken, and advancing by zero piles
        # its glyphs on one spot. Deliberate leniency, applied only when the
        # document says nothing.
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
                        for offset in range(0, len(values) // 3):
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
                                bounds = internal_clipped_cid_bounds(first, last)
                                if bounds is not None:
                                    clipped_first, clipped_last = bounds
                                    for cid in range(clipped_first, clipped_last + 1):
                                        vertical_metrics[cid] = (width, vx, vy)
                            except ValueError:
                                pass
                        index += 5
            widths = parse_cid_widths(descendant.get("W"))
            descriptor = descendant.get("FontDescriptor")

    # ISO 32000-1 Table 122 scopes MissingWidth to "character codes whose widths
    # are not specified in a font dictionary's Widths array". A CIDFont has no
    # Widths array -- 9.7.4.3 gives it W/DW instead, and Table 117 makes DW the
    # default width -- so a descriptor MissingWidth must not override DW.
    if isinstance(descriptor, dict) and subtype != "Type0":
        desc_missing_width = descriptor.get("MissingWidth")
        if desc_missing_width is not None:
            default_width = parse_float(desc_missing_width, default_width)
            default_width_explicit = True

    if subtype == "Type0":
        return FontMetrics(
            widths=widths,
            default_width=default_width,
            default_width_explicit=default_width_explicit,
            default_vertical_displacement_y=default_vertical_displacement_y,
            default_vertical_origin_y=default_vertical_origin_y,
            vertical_metrics=vertical_metrics,
        )

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
    # Top-level MissingWidth and malformed entries are historical input recovery.
    if font.get("MissingWidth") is not None:
        return internal_recover_font_widths(font, subtype)
    try:
        metrics = pdf_font_widths(font, subtype)
    except (ValueError, TypeError, IndexError):
        return internal_recover_font_widths(font, subtype)
    if subtype == "Type0":
        return metrics
    if not metrics.default_width_explicit:
        return replace(metrics, default_width=1000.0)
    return metrics
