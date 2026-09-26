# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Sequence

from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace


def color_space_paints(spec: ColorSpace) -> bool:
    while spec.kind in {"Indexed", "Pattern"} and spec.base is not None:
        spec = spec.base
    if spec.kind == "Separation":
        return spec.colorants != ("None",)
    if spec.kind == "DeviceN":
        return not spec.colorants or any(name != "None" for name in spec.colorants)
    return True


def normalize_color_components(spec: ColorSpace, components: Sequence[object]) -> tuple[float, ...]:
    if spec.kind == "Pattern":
        raise ValueError("Pattern color requires a pattern selection")
    count = len(spec.component_ranges)
    if count == 0:
        raise ValueError("unsupported color space")
    if len(components) != count:
        raise ValueError("invalid color component operands")
    values = tuple(
        require_pdf_number(value, "color component must be a finite PDF number")
        for value in components
    )
    if spec.kind == "Indexed":
        return (float(indexed_color_index(values[0], spec.hival)),)
    ranges = spec.component_ranges
    return tuple(
        max(low, min(high, value)) for value, (low, high) in zip(values, ranges, strict=True)
    )


def initial_color_components(spec: ColorSpace) -> tuple[float, ...] | None:
    count = len(spec.component_ranges)
    if count == 0 and spec.kind != "Pattern":
        raise ValueError("unsupported color space")
    if spec.kind == "Pattern":
        return None
    if spec.kind == "DeviceCMYK":
        return (0.0, 0.0, 0.0, 1.0)
    initial = 1.0 if spec.kind in {"Separation", "DeviceN"} else 0.0
    return normalize_color_components(spec, (initial,) * count)


def indexed_color_index(value: float, hival: int) -> int:
    return max(0, min(hival, int(value + 0.5)))


__all__ = (
    "color_space_paints",
    "normalize_color_components",
    "initial_color_components",
)
