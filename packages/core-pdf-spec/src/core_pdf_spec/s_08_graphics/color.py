# SPDX-License-Identifier: AGPL-3.0-only
"""PDF color-space semantics, independent of an output device."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf_spec.s_07_syntax_primitives.coercion import require_pdf_number
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace
from core_pdf_spec.s_08_graphics.pdf_function import compile_pdf_function


def color_space_paints(spec: ColorSpace) -> bool:
    """Whether this space can paint any colorant (ISO 32000-2, 8.6.6.4-5).

    Separation None and nonempty all-None DeviceN spaces discard their output,
    including through an Indexed palette or an uncolored Pattern. The tint
    function is irrelevant in these cases. A mixed DeviceN space still passes
    every component, including None components, to its alternate tint function.
    """
    while spec.kind in {"Indexed", "Pattern"} and spec.base is not None:
        spec = spec.base
    if spec.kind == "Separation":
        return spec.colorants != ("None",)
    if spec.kind == "DeviceN":
        return not spec.colorants or any(name != "None" for name in spec.colorants)
    return True


def normalize_color_components(spec: ColorSpace, components: Sequence[object]) -> tuple[float, ...]:
    """Validate component types/counts and apply their PDF-defined limits.

    Pattern selection is separate; pass its underlying space for stencil colors.
    """
    if spec.kind == "Pattern":
        raise ValueError("Pattern color requires a pattern selection")
    count = len(spec.component_ranges)
    if count == 0 and spec.kind != "Pattern":
        raise ValueError("unsupported color space")
    if len(components) != count:
        raise ValueError("invalid color component operands")
    values = tuple(
        require_pdf_number(value, "color component must be a finite PDF number")
        for value in components
    )
    if spec.kind == "Indexed":
        return (float(internal_indexed_color_index(values[0], spec.hival)),)
    ranges = spec.component_ranges
    return tuple(
        max(low, min(high, value)) for value, (low, high) in zip(values, ranges, strict=True)
    )


def initial_color_components(spec: ColorSpace) -> tuple[float, ...] | None:
    """ISO 32000-1 Table 74: the color established by CS/cs.

    None represents the initial Pattern selection, which paints nothing.
    """
    count = len(spec.component_ranges)
    if count == 0 and spec.kind != "Pattern":
        raise ValueError("unsupported color space")
    if spec.kind == "Pattern":
        return None
    if spec.kind == "DeviceCMYK":
        return (0.0, 0.0, 0.0, 1.0)
    initial = 1.0 if spec.kind in {"Separation", "DeviceN"} else 0.0
    return normalize_color_components(spec, (initial,) * count)


def internal_indexed_color_index(value: float, hival: int) -> int:
    """Round an Indexed component half up, then clamp it to the palette bounds."""
    return max(0, min(hival, int(value + 0.5)))


def indexed_color_components(spec: ColorSpace, value: float) -> tuple[float, ...]:
    if spec.base is None:
        raise ValueError("invalid Indexed base color space")
    components = len(spec.base.component_ranges)
    if spec.lookup is None or components <= 0:
        raise ValueError("invalid Indexed color lookup")
    index = internal_indexed_color_index(
        require_pdf_number(value, "invalid Indexed color component"), spec.hival
    )
    entry = spec.lookup[index * components : (index + 1) * components]
    if len(entry) != components:
        raise ValueError("invalid Indexed color lookup")
    return tuple(
        low + sample / 255 * (high - low)
        for sample, (low, high) in zip(entry, spec.base.component_ranges, strict=True)
    )


def tint_color_components(spec: ColorSpace, components: Sequence[float]) -> tuple[float, ...]:
    if spec.tint_fn is None:
        raise ValueError("missing tint transform")
    values = normalize_color_components(spec, components)
    result = compile_pdf_function(spec.tint_fn)(*values)
    if spec.alternate is None or len(result) != len(spec.alternate.component_ranges):
        raise ValueError("invalid tint transform output count")
    return normalize_color_components(spec.alternate, result)


def calgray_to_xyz(
    value: float, gamma: float, white_point: tuple[float, float, float]
) -> tuple[float, float, float]:
    if gamma <= 0 or white_point[0] <= 0 or white_point[1] != 1 or white_point[2] <= 0:
        raise ValueError("invalid CalGray parameters")
    gray = max(0.0, min(1.0, value)) ** gamma
    return (white_point[0] * gray, white_point[1] * gray, white_point[2] * gray)


__all__ = (
    "color_space_paints",
    "normalize_color_components",
    "initial_color_components",
    "indexed_color_components",
    "tint_color_components",
    "calgray_to_xyz",
)
