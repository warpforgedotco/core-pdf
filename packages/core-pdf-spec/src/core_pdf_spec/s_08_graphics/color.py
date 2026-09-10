# SPDX-License-Identifier: AGPL-3.0-only
"""PDF color-space semantics, independent of an output device."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf_spec.s_08_graphics.pdf_function import compile_pdf_function


def color_component_count(spec: ImageColorSpec) -> int:
    """The numeric component count defined by the PDF color-space family."""
    if spec.kind in {"DeviceGray", "CalGray", "Indexed", "Separation"}:
        return 1
    if spec.kind in {"DeviceRGB", "CalRGB", "Lab"}:
        return 3
    if spec.kind == "DeviceCMYK":
        return 4
    if spec.kind in {"ICCBased", "DeviceN"}:
        if type(spec.channels) is not int or spec.channels <= 0:
            raise ValueError("invalid color component count")
        if spec.kind == "ICCBased" and spec.channels not in {1, 3, 4}:
            raise ValueError("invalid ICCBased channel count")
        return spec.channels
    if spec.kind == "Pattern":
        if spec.pattern_base is None:
            return 0
        if spec.pattern_base.kind == "Pattern":
            raise ValueError("Pattern cannot be its own underlying color space")
        return color_component_count(spec.pattern_base)
    raise ValueError("unsupported color space")


def internal_color_number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("color component must be a PDF number")
    return parse_float_strict(value, "color component must be a finite PDF number")


def internal_color_ranges(spec: ImageColorSpec, count: int) -> tuple[tuple[float, float], ...]:
    if spec.kind == "Lab":
        raw = spec.params.get("Range", (-100, 100, -100, 100))
        size = 4
    elif spec.kind == "ICCBased":
        raw = spec.params.get("Range", (0, 1) * count)
        size = 2 * count
    else:
        return ((0.0, 1.0),) * count
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ValueError("invalid color component Range")
    values = tuple(internal_color_number(value) for value in raw)
    ranges = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low > high for low, high in ranges):
        raise ValueError("invalid color component Range")
    return ((0.0, 100.0), *ranges) if spec.kind == "Lab" else ranges


def normalize_color_components(
    spec: ImageColorSpec, components: Sequence[object]
) -> tuple[float, ...]:
    """Validate component types/counts and apply their PDF-defined limits.

    Pattern selection is separate; pass its underlying space for stencil colors.
    """
    if spec.kind == "Pattern":
        raise ValueError("Pattern color requires a pattern selection")
    count = color_component_count(spec)
    if len(components) != count:
        raise ValueError("invalid color component operands")
    values = tuple(internal_color_number(value) for value in components)
    if spec.kind == "Indexed":
        return (float(internal_indexed_color_index(values[0], spec.hival)),)
    ranges = internal_color_ranges(spec, count)
    return tuple(
        max(low, min(high, value)) for value, (low, high) in zip(values, ranges, strict=True)
    )


def initial_color_components(spec: ImageColorSpec) -> tuple[float, ...] | None:
    """ISO 32000-1 Table 74: the color established by CS/cs.

    None represents the initial Pattern selection, which paints nothing.
    """
    count = color_component_count(spec)
    if spec.kind == "Pattern":
        return None
    if spec.kind == "DeviceCMYK":
        return (0.0, 0.0, 0.0, 1.0)
    initial = 1.0 if spec.kind in {"Separation", "DeviceN"} else 0.0
    return normalize_color_components(spec, (initial,) * count)


def internal_indexed_color_index(value: float, hival: int) -> int:
    """Round an Indexed component half up, then clamp it to the palette bounds."""
    return max(0, min(hival, int(value + 0.5)))


def indexed_color_components(
    spec: ImageColorSpec, value: float, components: int
) -> tuple[float, ...]:
    if spec.lookup is None or components <= 0:
        raise ValueError("invalid Indexed color lookup")
    index = internal_indexed_color_index(value, spec.hival)
    entry = spec.lookup[index * components : (index + 1) * components]
    if len(entry) != components:
        raise ValueError("invalid Indexed color lookup")
    return tuple(sample / 255 for sample in entry)


def tint_color_components(spec: ImageColorSpec, components: Sequence[float]) -> tuple[float, ...]:
    if spec.tint_fn is None:
        raise ValueError("missing tint transform")
    return compile_pdf_function(spec.tint_fn)(*(max(0.0, min(1.0, value)) for value in components))


def calgray_to_xyz(
    value: float, gamma: float, white_point: tuple[float, float, float]
) -> tuple[float, float, float]:
    if gamma <= 0 or white_point[0] <= 0 or white_point[1] != 1 or white_point[2] <= 0:
        raise ValueError("invalid CalGray parameters")
    gray = max(0.0, min(1.0, value)) ** gamma
    return (white_point[0] * gray, white_point[1] * gray, white_point[2] * gray)


__all__ = (
    "color_component_count",
    "normalize_color_components",
    "initial_color_components",
    "indexed_color_components",
    "tint_color_components",
    "calgray_to_xyz",
)
