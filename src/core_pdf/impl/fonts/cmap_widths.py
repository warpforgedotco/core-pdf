from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict
from core_pdf_spec.s_09_fonts.widths import (
    MAX_CID,
    MIN_CID,
    CompactCIDWidthMap,
)


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
