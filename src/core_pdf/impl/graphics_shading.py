# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import ClassVar

from core_pdf.impl.caches import IdentityCache
from core_pdf.impl.graphics_color import color_operands_to_srgb
from core_pdf.impl.graphics_color_spec import (
    DEVICE_KINDS,
    parse_color_space,
    raw_color_space_paints,
)
from core_pdf.impl.graphics_functions import (
    compile_pdf_function,
    number_array,
)
from core_pdf.impl.pdf_values import lenient_int
from core_pdf.impl.types import GeneratedRecord
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator
from core_pdf_spec.s_08_graphics.shading import parse_shading


class PreparedShading(GeneratedRecord):
    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_model: str
    bbox: tuple[float, float, float, float] | None
    evaluator: Callable[[float], tuple[float, ...]]
    color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING

    __repr_fields__: ClassVar[tuple[str, ...]] = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_model",
        "bbox",
        "color_rendering",
    )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.shading_type == other.shading_type
            and self.coords == other.coords
            and self.domain == other.domain
            and self.extend_start == other.extend_start
            and self.extend_end == other.extend_end
            and self.color_model == other.color_model
            and self.bbox == other.bbox
            and self.color_rendering == other.color_rendering
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.shading_type,
                self.coords,
                self.domain,
                self.extend_start,
                self.extend_end,
                self.color_model,
                self.bbox,
                self.color_rendering,
            )
        )


type ShadingEvaluatorCache = IdentityCache[
    tuple[PdfFunctionEvaluator, Callable[[float], tuple[float, ...]], str]
]


def prepare_shading(
    dictionary: object,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    evaluators: ShadingEvaluatorCache | None = None,
) -> PreparedShading | None:
    if not isinstance(dictionary, dict):
        return None
    if not raw_color_space_paints(dictionary.get("ColorSpace")):
        return None
    shading_type = lenient_int(dictionary.get("ShadingType"), 0)
    if shading_type not in {2, 3}:
        return None
    coords = number_array(dictionary.get("Coords"))
    if (shading_type == 2 and len(coords) < 4) or (shading_type == 3 and len(coords) < 6):
        return None
    domain_values = number_array(dictionary.get("Domain"))
    domain = (domain_values[0], domain_values[1]) if len(domain_values) >= 2 else (0.0, 1.0)
    extend = dictionary.get("Extend")
    extend_start = isinstance(extend, (list, tuple)) and len(extend) > 0 and extend[0] is True
    extend_end = isinstance(extend, (list, tuple)) and len(extend) > 1 and extend[1] is True
    bbox_values = number_array(dictionary.get("BBox"))
    bbox = (
        (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3])
        if len(bbox_values) >= 4
        else None
    )
    normalized = dict(dictionary)
    normalized.update(
        ShadingType=shading_type,
        Coords=coords[: 4 if shading_type == 2 else 6],
        Domain=domain,
        Extend=(extend_start, extend_end),
        ColorSpace=dictionary.get("ColorSpace") or "DeviceRGB",
    )
    if bbox is None:
        normalized.pop("BBox", None)
    else:
        normalized["BBox"] = bbox
    function = normalized.get("Function")
    color_space = normalized["ColorSpace"]
    key = (id(function), id(color_space), rendering)
    cached = evaluators.get_key(function, key) if evaluators is not None else None
    if cached is not None:
        compiled, evaluator, color_model = cached
        try:
            spec = parse_shading(normalized, compile_function=lambda _function: compiled)
        except ValueError:
            return None
    else:
        try:
            spec = parse_shading(normalized, compile_function=compile_pdf_function)
            space = parse_color_space(spec.color_space)
        except ValueError:
            return None
        evaluator, color_model = shading_color_evaluator(spec.evaluator, space, rendering)
        if evaluators is not None:
            evaluators.put_key(
                function, key, (spec.evaluator, evaluator, color_model), (color_space,)
            )
    return PreparedShading(
        spec.shading_type,
        coords,
        spec.domain,
        spec.extend_start,
        spec.extend_end,
        color_model,
        spec.bbox,
        evaluator,
        rendering,
    )


def shading_color_evaluator(
    function: PdfFunctionEvaluator, space: ColorSpace, rendering: ColorRendering
) -> tuple[Callable[[float], tuple[float, ...]], str]:
    if space.kind in DEVICE_KINDS:
        return function, space.kind

    @lru_cache(maxsize=8192)
    def convert(value: float) -> tuple[float, ...]:
        components = function(value)
        return color_operands_to_srgb(space, components, rendering=rendering) or components

    return convert, "DeviceRGB"


__all__ = (
    "PreparedShading",
    "ShadingEvaluatorCache",
    "prepare_shading",
)
