# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize PDF shading dictionaries and functions for raster consumers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core_pdf.impl._impl.graphics.color_spec import describe_color_space
from core_pdf.impl._impl.graphics.functions import (
    internal_compile_pdf_function,
    internal_number_array,
)
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf.impl.spec.s_08_graphics.pdf_function import internal_PdfFunctionEvaluator
from core_pdf.impl.spec.s_08_graphics.shading import parse_shading


@dataclass(frozen=True, slots=True)
class PreparedShading:
    """PDF-independent numeric shading parameters consumed by the renderer."""

    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_model: str
    bbox: tuple[float, float, float, float] | None
    internal_evaluator: Callable[[float], tuple[float, ...]] = field(repr=False, compare=False)

    def evaluate(self, value: float) -> tuple[float, ...]:
        return self.internal_evaluator(value)


def prepare_shading(
    dictionary: object,
    *,
    compile_function: Callable[
        [object], internal_PdfFunctionEvaluator
    ] = internal_compile_pdf_function,
) -> PreparedShading | None:
    """Normalize one axial or radial PDF shading dictionary."""
    if not isinstance(dictionary, dict):
        return None
    shading_type = parse_int(dictionary.get("ShadingType"), 0)
    if shading_type not in {2, 3}:
        return None
    coords = internal_number_array(dictionary.get("Coords"))
    if (shading_type == 2 and len(coords) < 4) or (shading_type == 3 and len(coords) < 6):
        return None
    domain_values = internal_number_array(dictionary.get("Domain"))
    domain = (domain_values[0], domain_values[1]) if len(domain_values) >= 2 else (0.0, 1.0)
    extend = dictionary.get("Extend")
    extend_start = isinstance(extend, (list, tuple)) and len(extend) > 0 and extend[0] is True
    extend_end = isinstance(extend, (list, tuple)) and len(extend) > 1 and extend[1] is True
    bbox_values = internal_number_array(dictionary.get("BBox"))
    bbox = (
        (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        if len(bbox_values) >= 4
        else None
    )
    normalized = dict(dictionary)
    normalized.update(
        Coords=coords[: 4 if shading_type == 2 else 6],
        Domain=domain,
        Extend=(extend_start, extend_end),
        ColorSpace=dictionary.get("ColorSpace") or "DeviceRGB",
    )
    if bbox is None:
        normalized.pop("BBox", None)
    else:
        normalized["BBox"] = bbox
    try:
        spec = parse_shading(normalized, compile_function=compile_function)
    except ValueError:
        return None
    return PreparedShading(
        spec.shading_type,
        coords,
        spec.domain,
        spec.extend_start,
        spec.extend_end,
        describe_color_space(dictionary.get("ColorSpace")) or "DeviceRGB",
        spec.bbox,
        spec.evaluator,
    )


__all__ = ("PreparedShading", "prepare_shading")
