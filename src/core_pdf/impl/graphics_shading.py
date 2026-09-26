# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import ClassVar

from core_pdf.impl.graphics_color import color_operands_to_srgb
from core_pdf.impl.graphics_color_spec import parse_color_space, raw_color_space_paints
from core_pdf.impl.graphics_functions import (
    compile_pdf_function,
    number_array,
)
from core_pdf.impl.types import Record, frozen_setattr
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator
from core_pdf_spec.s_08_graphics.shading import parse_shading


class PreparedShading(Record):
    __slots__ = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_model",
        "bbox",
        "evaluator",
        "color_rendering",
    )

    shading_type: int
    coords: tuple[float, ...]
    domain: tuple[float, float]
    extend_start: bool
    extend_end: bool
    color_model: str
    bbox: tuple[float, float, float, float] | None
    evaluator: Callable[[float], tuple[float, ...]]
    color_rendering: ColorRendering

    __fields__: ClassVar[tuple[str, ...]] = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_model",
        "bbox",
        "evaluator",
        "color_rendering",
    )
    __repr_fields__: ClassVar[tuple[str, ...]] = tuple(
        name for name in __fields__ if name != "evaluator"
    )
    __match_args__ = (
        "shading_type",
        "coords",
        "domain",
        "extend_start",
        "extend_end",
        "color_model",
        "bbox",
        "evaluator",
        "color_rendering",
    )

    def __init__(
        self,
        shading_type: int,
        coords: tuple[float, ...],
        domain: tuple[float, float],
        extend_start: bool,
        extend_end: bool,
        color_model: str,
        bbox: tuple[float, float, float, float] | None,
        evaluator: Callable[[float], tuple[float, ...]],
        color_rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    ) -> None:
        frozen_setattr(self, "shading_type", shading_type)
        frozen_setattr(self, "coords", coords)
        frozen_setattr(self, "domain", domain)
        frozen_setattr(self, "extend_start", extend_start)
        frozen_setattr(self, "extend_end", extend_end)
        frozen_setattr(self, "color_model", color_model)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "evaluator", evaluator)
        frozen_setattr(self, "color_rendering", color_rendering)

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


# What a shading's colours depend on -- its Function and ColorSpace objects,
# held so their ids stay theirs, and the rendering -- mapped to the compiled
# function, the colour evaluator built over it, and its colour model.
type ShadingEvaluatorKey = tuple[int, int, ColorRendering]
type ShadingEvaluatorCache = dict[
    ShadingEvaluatorKey,
    tuple[object, object, PdfFunctionEvaluator, Callable[[float], tuple[float, ...]], str],
]


def prepare_shading(
    dictionary: object,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    evaluators: ShadingEvaluatorCache | None = None,
) -> PreparedShading | None:
    """The shading's geometry and colour evaluator, or None if it cannot paint.

    With `evaluators`, shadings that share a Function and ColorSpace object
    -- the copies a tiled sh makes, one per tile with its own Coords and
    BBox -- share one compiled function and one colour evaluator, with its
    colour cache, instead of compiling them again.
    """
    if not isinstance(dictionary, dict):
        return None
    if not raw_color_space_paints(dictionary.get("ColorSpace")):
        return None
    shading_type = parse_int(dictionary.get("ShadingType"), 0, python_syntax=True)
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
    cached = evaluators.get(key) if evaluators is not None else None
    if cached is not None and cached[0] is function and cached[1] is color_space:
        compiled = cached[2]
        try:
            # Coords, Domain, Extend and BBox are still checked; only the
            # compile is skipped.
            spec = parse_shading(normalized, compile_function=lambda _function: compiled)
        except ValueError:
            return None
        evaluator, color_model = cached[3], cached[4]
    else:
        try:
            spec = parse_shading(normalized, compile_function=compile_pdf_function)
            space = parse_color_space(spec.color_space)
        except ValueError:
            return None
        evaluator, color_model = shading_color_evaluator(spec.evaluator, space, rendering)
        if evaluators is not None:
            evaluators[key] = (function, color_space, spec.evaluator, evaluator, color_model)
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
    """The shading's colour at a parameter value, and the colour model it is in."""
    if space.kind in {"DeviceGray", "DeviceRGB", "DeviceCMYK"}:
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
