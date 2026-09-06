from __future__ import annotations

from contextlib import suppress
from typing import Any

from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float_strict
from core_pdf.impl.spec.s_09_fonts.cmap_widths import (
    CompactCIDWidthMap,
    FontWidthMap,
    SparseFontWidthMap,
    internal_MAX_CID,
    internal_MIN_CID,
)


def internal_clipped_cid_bounds(first: int, last: int) -> tuple[int, int] | None:
    """Return the valid part of a recovery-parsed CID range.

    PDF CIDs are limited to 0 through 65535. Width arrays are parsed
    tolerantly elsewhere in this module, so a partially overlapping range
    keeps its valid portion while a reversed or wholly invalid range is
    ignored.
    """
    if last < first or last < internal_MIN_CID or first > internal_MAX_CID:
        return None
    return (max(first, internal_MIN_CID), min(last, internal_MAX_CID))


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


def parse_cid_widths(value: Any) -> FontWidthMap:
    if value is None:
        return SparseFontWidthMap()
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid CID widths array")
    if len(value) == 2 and type(value[0]) is int:
        contiguous_widths = value[1]
        if isinstance(contiguous_widths, (list, tuple)) and set(
            map(type, contiguous_widths)
        ).issubset({int, float}):
            first = value[0]
            bounds = internal_clipped_cid_bounds(
                first,
                first + len(contiguous_widths) - 1,
            )
            if bounds is None:
                return SparseFontWidthMap()
            clipped_first, clipped_last = bounds
            offset = clipped_first - first
            count = clipped_last - clipped_first + 1
            return CompactCIDWidthMap(
                clipped_first,
                tuple(contiguous_widths[offset : offset + count]),
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
                if internal_MIN_CID <= code <= internal_MAX_CID:
                    if type(w) is int:
                        widths[code] = float(w)
                    elif type(w) is float:
                        widths[code] = w
                    else:
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
            bounds = internal_clipped_cid_bounds(first, last)
            if bounds is None:
                index += 2
                continue
            clipped_first, clipped_last = bounds
            for i in range(clipped_first, clipped_last + 1):
                widths[i] = width
            index += 2
    return SparseFontWidthMap(widths)
