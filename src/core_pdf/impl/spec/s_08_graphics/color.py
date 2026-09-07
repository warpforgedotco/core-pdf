# SPDX-License-Identifier: AGPL-3.0-only
"""PDF color-space semantics, independent of an output device."""

from __future__ import annotations

from collections.abc import Sequence

from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf.impl.spec.s_08_graphics.pdf_function import internal_compile_pdf_function


def indexed_color_components(
    spec: ImageColorSpec, value: float, components: int
) -> tuple[float, ...]:
    if spec.lookup is None or components <= 0:
        raise ValueError("invalid Indexed color lookup")
    index = max(0, min(spec.hival, round(value)))
    entry = spec.lookup[index * components : (index + 1) * components]
    if len(entry) != components:
        raise ValueError("invalid Indexed color lookup")
    return tuple(sample / 255 for sample in entry)


def tint_color_components(spec: ImageColorSpec, components: Sequence[float]) -> tuple[float, ...]:
    if spec.tint_fn is None:
        raise ValueError("missing tint transform")
    return internal_compile_pdf_function(spec.tint_fn)(
        *(max(0.0, min(1.0, value)) for value in components)
    )


def calgray_to_xyz(
    value: float, gamma: float, white_point: tuple[float, float, float]
) -> tuple[float, float, float]:
    if gamma <= 0 or white_point[0] <= 0 or white_point[1] != 1 or white_point[2] <= 0:
        raise ValueError("invalid CalGray parameters")
    gray = max(0.0, min(1.0, value)) ** gamma
    return (white_point[0] * gray, white_point[1] * gray, white_point[2] * gray)
