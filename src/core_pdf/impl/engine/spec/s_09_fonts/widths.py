"""Font width parsing logic."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.engine.spec.s_07_syntax.primitives import parse_float, parse_int


def get_descendant(font: dict[str, Any]) -> dict[str, Any] | None:
    descendant_fonts = font.get("DescendantFonts")
    if isinstance(descendant_fonts, (list, tuple)) and descendant_fonts:
        candidate = descendant_fonts[0]
        if isinstance(candidate, dict):
            return candidate
    return None


def parse_cid_widths(value: Any) -> dict[int, float]:
    widths: dict[int, float] = {}
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    index = 0
    while index < len(value):
        first = value[index]
        if not isinstance(first, int):
            raise ValueError("invalid CID widths array")
        index += 1
        if index >= len(value):
            raise ValueError("invalid CID widths array")
        nxt = value[index]
        if isinstance(nxt, (list, tuple)):
            for i, w in enumerate(nxt):
                widths[first + i] = parse_float(w, 0.0)
            index += 1
        else:
            if not isinstance(nxt, int) or index + 1 >= len(value):
                raise ValueError("invalid CID widths array")
            width = parse_float(value[index + 1], 0.0)
            for i in range(first, nxt + 1):
                widths[i] = width
            index += 2
    return widths


def parse_font_widths(
    font: dict[str, Any], subtype: str | None
) -> tuple[dict[int, float], float, bool]:
    widths: dict[int, float] = {}
    missing_width = font.get("MissingWidth")
    if missing_width is None:
        default_width = 1000.0
    else:
        default_width = parse_float(missing_width, 1000.0)
        if not isinstance(missing_width, (int, float)):
            raise ValueError("invalid font MissingWidth")
    is_vertical = False
    descriptor = font.get("FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_dw = descendant.get("DW")
            if descendant_dw is not None:
                if not isinstance(descendant_dw, (int, float)):
                    raise ValueError("invalid descendant width")
                default_width = parse_float(descendant_dw, default_width)
            wmode = descendant.get("WMode", font.get("WMode"))
            if wmode is None:
                wmode = 0
            elif not isinstance(wmode, int):
                raise ValueError("invalid font WMode")
            is_vertical = parse_int(wmode, 0) == 1
            widths.update(parse_cid_widths(descendant.get("W")))
            descriptor = descendant.get("FontDescriptor")
            if isinstance(descriptor, dict):
                desc_missing_width = descriptor.get("MissingWidth")
                if desc_missing_width is not None:
                    if not isinstance(desc_missing_width, (int, float)):
                        raise ValueError("invalid font MissingWidth")
                    default_width = parse_float(desc_missing_width, default_width)

    if isinstance(descriptor, dict):
        desc_missing_width = descriptor.get("MissingWidth")
        if desc_missing_width is not None:
            if not isinstance(desc_missing_width, (int, float)):
                raise ValueError("invalid font MissingWidth")
            default_width = parse_float(desc_missing_width, default_width)
        return widths, default_width, is_vertical

    if subtype == "Type0":
        return widths, default_width, is_vertical

    first_char_val = font.get("FirstChar")
    if first_char_val is None:
        first_char = 0
    elif isinstance(first_char_val, int):
        first_char = first_char_val
    else:
        raise ValueError("invalid font FirstChar")
    font_widths = font.get("Widths")
    if isinstance(font_widths, (list, tuple)):
        for index, width in enumerate(font_widths):
            if width is None:
                widths[first_char + index] = default_width
            elif isinstance(width, (int, float)):
                widths[first_char + index] = parse_float(width, default_width)
            else:
                raise ValueError("invalid font widths array")
    elif font_widths is not None:
        raise ValueError("invalid font widths array")
    return widths, default_width, is_vertical
