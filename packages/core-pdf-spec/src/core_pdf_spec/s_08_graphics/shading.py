# SPDX-License-Identifier: AGPL-3.0-only
"""PDF axial/radial shading descriptors and parameter semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from core_pdf_spec.s_08_graphics.pdf_function import (
    PdfFunctionEvaluator,
    compile_pdf_function,
    internal_number_array,
)


@dataclass(frozen=True, slots=True)
class ShadingSpec:
    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_space: object
    bbox: tuple[float, float, float, float] | None
    evaluator: PdfFunctionEvaluator = field(repr=False, compare=False)


def parse_shading(
    dictionary: object,
    *,
    compile_function: Callable[[object], PdfFunctionEvaluator] = compile_pdf_function,
) -> ShadingSpec:
    if not isinstance(dictionary, dict):
        raise ValueError("invalid shading dictionary")
    shading_type = dictionary.get("ShadingType")
    if type(shading_type) is not int or shading_type not in {2, 3}:
        raise ValueError("unsupported shading type")
    coords = internal_number_array(dictionary.get("Coords"))
    if len(coords) != (4 if shading_type == 2 else 6):
        raise ValueError("invalid shading coordinates")
    domain = internal_number_array(dictionary.get("Domain", (0.0, 1.0)))
    if len(domain) != 2:
        raise ValueError("invalid shading domain")
    extend = dictionary.get("Extend", (False, False))
    if (
        not isinstance(extend, (list, tuple))
        or len(extend) != 2
        or any(type(value) is not bool for value in extend)
    ):
        raise ValueError("invalid shading extension")
    color_space = dictionary.get("ColorSpace")
    if color_space is None:
        raise ValueError("missing shading color space")
    bbox_values = internal_number_array(dictionary.get("BBox"))
    if dictionary.get("BBox") is not None and len(bbox_values) != 4:
        raise ValueError("invalid shading bbox")
    bbox = (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3]) if bbox_values else None
    return ShadingSpec(
        shading_type,
        coords,
        (domain[0], domain[1]),
        cast(bool, extend[0]),
        cast(bool, extend[1]),
        color_space,
        bbox,
        compile_function(dictionary.get("Function")),
    )


__all__ = (
    "ShadingSpec",
    "parse_shading",
)
