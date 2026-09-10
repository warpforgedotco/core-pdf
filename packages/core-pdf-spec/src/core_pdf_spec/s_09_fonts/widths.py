"""PDF font width and vertical-metric dictionary semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict, parse_int_strict
from core_pdf_spec.s_09_fonts.cmap_widths import (
    FontWidthMap,
    SparseFontWidthMap,
    parse_cid_widths,
)
from core_pdf_spec.s_09_fonts.dictionaries import (
    get_descendant,
    internal_font_descriptor,
)


@dataclass(frozen=True, slots=True)
class FontMetrics:
    widths: FontWidthMap
    default_width: float
    # Whether the document actually stated the default (MissingWidth, or DW for
    # a CIDFont). ISO 32000-1 Table 122 defaults MissingWidth to 0, so a stated
    # 0 must survive: it means unlisted codes have no advance, and substituting
    # a full em for it shifts every following glyph on the line.
    default_width_explicit: bool
    is_vertical: bool
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]


def parse_font_widths(font: dict[Any, Any], subtype: str | None) -> FontMetrics:
    vertical: dict[int, tuple[float, float, float]] = {}
    vy, dy = 880.0, -1000.0
    if subtype == "Type0":
        descendant = get_descendant(font)
        if descendant is None:
            raise ValueError("missing descendant font")
        default = parse_float_strict(descendant.get("DW", 1000), "invalid CID DW")
        dw2 = descendant.get("DW2", [880, -1000])
        if not isinstance(dw2, (list, tuple)) or len(dw2) != 2:
            raise ValueError("invalid CID DW2")
        vy, dy = (parse_float_strict(item, "invalid CID DW2") for item in dw2)
        w2 = descendant.get("W2", [])
        if not isinstance(w2, (list, tuple)):
            raise ValueError("invalid CID W2")
        index = 0
        while index < len(w2):
            first = parse_int_strict(w2[index], "invalid CID W2")
            if not 0 <= first <= 65535 or index + 1 >= len(w2):
                raise ValueError("invalid CID W2")
            item = w2[index + 1]
            if isinstance(item, (list, tuple)):
                if len(item) % 3 or first + len(item) // 3 > 65536:
                    raise ValueError("invalid CID W2 range")
                for offset in range(len(item) // 3):
                    a, b, c = item[offset * 3 : offset * 3 + 3]
                    vertical[first + offset] = (
                        parse_float_strict(a, "invalid W2"),
                        parse_float_strict(b, "invalid W2"),
                        parse_float_strict(c, "invalid W2"),
                    )
                index += 2
            else:
                last = parse_int_strict(item, "invalid CID W2")
                if not first <= last <= 65535 or index + 4 >= len(w2):
                    raise ValueError("invalid CID W2 range")
                a, b, c = w2[index + 2 : index + 5]
                metric = (
                    parse_float_strict(a, "invalid W2"),
                    parse_float_strict(b, "invalid W2"),
                    parse_float_strict(c, "invalid W2"),
                )
                vertical.update((cid, metric) for cid in range(first, last + 1))
                index += 5
        return FontMetrics(
            parse_cid_widths(descendant.get("W")),
            default,
            "DW" in descendant,
            False,
            dy,
            vy,
            vertical,
        )
    descriptor = internal_font_descriptor(font.get("FontDescriptor")) or {}
    default = parse_float_strict(descriptor.get("MissingWidth", 0), "invalid MissingWidth")
    values = font.get("Widths", [])
    if not isinstance(values, (list, tuple)):
        raise ValueError("invalid font widths array")
    if values and ("FirstChar" not in font or "LastChar" not in font):
        raise ValueError("missing font widths range")
    first = parse_int_strict(font.get("FirstChar", 0), "invalid FirstChar")
    last = parse_int_strict(font.get("LastChar", first + len(values) - 1), "invalid LastChar")
    if values and (not 0 <= first <= last <= 255 or len(values) != last - first + 1):
        raise ValueError("invalid font widths range")
    widths = SparseFontWidthMap(
        {
            first + index: parse_float_strict(value, "invalid font width")
            for index, value in enumerate(values)
        }
    )
    return FontMetrics(widths, default, "MissingWidth" in descriptor, False, dy, vy, vertical)


__all__ = ["get_descendant", "FontMetrics", "parse_font_widths"]
