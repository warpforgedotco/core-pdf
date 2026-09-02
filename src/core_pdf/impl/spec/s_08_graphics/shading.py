# SPDX-License-Identifier: AGPL-3.0-only
"""Normalize PDF shading dictionaries and functions for raster consumers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int
from core_pdf.impl.spec.s_08_graphics.color_kernels import (
    evaluate_sampled_tint_function,
)
from core_pdf.impl.spec.s_08_graphics.image_metadata import (
    image_color_space_name,
)


def internal_number_array(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    output: list[float] = []
    for item in value:
        parsed = parse_float(item, None)
        if parsed is None:
            return ()
        output.append(parsed)
    return tuple(output)


def internal_compile_pdf_function(function: Any) -> Callable[[float], tuple[float, ...]]:
    """Resolve a PDF Function's constants once, returning its evaluator.

    The renderer evaluates a shading per distinct parametric value, and past
    its 256-entry colour cache a diagonal or radial gradient produces a
    near-unique value per pixel. Re-reading N, C0, C1 (or Bounds/Encode) from
    the PDF dictionary on each of those was most of the cost.
    """
    if isinstance(function, (list, tuple)):
        # ISO 32000-1 Table 78: Function is "A 1-in, n-out function or an array
        # of n 1-in, 1-out functions (where n is the number of colour components
        # in the shading dictionary's colour space)". An array matched no branch
        # below and fell through to the scalar return, which the renderer then
        # painted as a grey ramp -- silently, so nothing upstream could see the
        # colour had been lost.
        parts = [internal_compile_pdf_function(entry) for entry in function]

        def evaluate_array(value: float) -> tuple[float, ...]:
            outputs: list[float] = []
            for part in parts:
                outputs.extend(part(value))
            return tuple(outputs) if outputs else (value,)

        return evaluate_array

    if isinstance(function, PdfStream):
        function_type = parse_int(function.dictionary.get("FunctionType"), -1)
        if function_type == 0:
            sampled = function

            def evaluate_sampled(value: float) -> tuple[float, ...]:
                try:
                    return tuple(evaluate_sampled_tint_function(sampled, value))
                except Exception:
                    return (value,)

            return evaluate_sampled
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = parse_int(function.get("FunctionType"), -1)
        dictionary = function
    else:
        return lambda value: (value,)

    if function_type == 2:
        exponent = parse_float(dictionary.get("N"), 1.0)
        c0 = list(internal_number_array(dictionary.get("C0")) or (0.0,))
        c1 = list(internal_number_array(dictionary.get("C1")) or (1.0,))
        count = max(len(c0), len(c1))
        if len(c0) < count:
            c0.extend([c0[-1] if c0 else 0.0] * (count - len(c0)))
        if len(c1) < count:
            c1.extend([c1[-1] if c1 else 1.0] * (count - len(c1)))
        start_values = tuple(c0)
        deltas = tuple(c1[index] - c0[index] for index in range(count))

        def evaluate_exponential(value: float) -> tuple[float, ...]:
            factor = value**exponent
            return tuple(
                start_value + factor * delta
                for start_value, delta in zip(start_values, deltas, strict=True)
            )

        return evaluate_exponential

    if function_type == 3:
        functions = dictionary.get("Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            return lambda value: (value,)
        bounds = internal_number_array(dictionary.get("Bounds"))
        encode = internal_number_array(dictionary.get("Encode"))
        parts = [internal_compile_pdf_function(entry) for entry in functions]

        def evaluate_stitching(value: float) -> tuple[float, ...]:
            index = 0
            while index < len(bounds) and value >= bounds[index]:
                index += 1
            low = bounds[index - 1] if index > 0 else 0.0
            high = bounds[index] if index < len(bounds) else 1.0
            enc0 = encode[index * 2] if index * 2 < len(encode) else 0.0
            enc1 = encode[index * 2 + 1] if index * 2 + 1 < len(encode) else 1.0
            encoded = enc0 if high == low else enc0 + (value - low) * (enc1 - enc0) / (high - low)
            return parts[min(index, len(parts) - 1)](encoded)

        return evaluate_stitching

    return lambda value: (value,)


def internal_evaluate_pdf_function(function: Any, value: float) -> tuple[float, ...]:
    """Evaluate the PDF Function forms supported by the rasterizer."""
    return internal_compile_pdf_function(function)(value)


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


def prepare_shading(dictionary: object) -> PreparedShading | None:
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
    return PreparedShading(
        shading_type=shading_type,
        coords=coords,
        domain=domain,
        extend_start=extend_start,
        extend_end=extend_end,
        color_model=image_color_space_name(dictionary.get("ColorSpace")) or "DeviceRGB",
        bbox=bbox,
        internal_evaluator=internal_compile_pdf_function(dictionary.get("Function")),
    )


__all__ = ("PreparedShading", "prepare_shading")
