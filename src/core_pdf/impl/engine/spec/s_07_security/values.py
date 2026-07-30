# SPDX-License-Identifier: AGPL-3.0-only
"""Native security dictionary value coercions."""

from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_int,
)


def get_int(val: object, default: int = 0) -> int:
    parsed = parse_int(val, None)
    if parsed is None:
        if val is None:
            return default
        raise ValueError(f"invalid integer value: {val!r}")
    return parsed


def get_uint(val: object, n_bits: int = 32) -> int:
    value = get_int(val, 0)
    if value >= 0:
        return value
    return value + (1 << n_bits)


def get_name(val: object) -> str:
    return normalize_pdf_name(val, "") or ""
