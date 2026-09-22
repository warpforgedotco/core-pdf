# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from typing import Any, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
)
from core_pdf_spec.s_08_graphics.calculator import internal_compile_calculator_function

PdfFunctionEvaluator = Callable[..., tuple[float, ...]]


def with_pdf_function_range(
    evaluate: PdfFunctionEvaluator,
    range_values: Any,
    *,
    output_count: int | None = None,
) -> PdfFunctionEvaluator:
    if range_values is None:
        return evaluate
    values = require_pdf_number_array(range_values, "invalid PDF function range")
    if not values or len(values) % 2:
        raise ValueError("invalid PDF function range")
    ranges = tuple(zip(values[::2], values[1::2], strict=True))
    if any(lower > upper for lower, upper in ranges):
        raise ValueError("invalid PDF function range")
    if output_count is not None and len(ranges) != output_count:
        raise ValueError("PDF function range has incorrect output count")

    def evaluate_clipped(*inputs: float) -> tuple[float, ...]:
        outputs = evaluate(*inputs)
        if len(outputs) != len(ranges):
            raise ValueError("PDF function range has incorrect output count")
        return tuple(
            max(lower, min(upper, output))
            for output, (lower, upper) in zip(outputs, ranges, strict=True)
        )

    return evaluate_clipped


def internal_scalar_domain(dictionary: dict[Any, Any]) -> tuple[float, float]:
    domain_obj = dictionary.get("Domain")
    if domain_obj is None:
        raise ValueError("missing PDF function domain")
    domain = require_pdf_number_array(domain_obj, "invalid PDF function domain")
    if len(domain) != 2 or domain[1] < domain[0]:
        raise ValueError("invalid PDF function domain")
    return (domain[0], domain[1])


def internal_compile_sampled_function(function: PdfStream) -> PdfFunctionEvaluator:
    dictionary = function.dictionary
    if internal_function_type(dictionary) != 0:
        raise ValueError("invalid sampled function")
    if type(dictionary.get("BitsPerSample")) is not int or dictionary["BitsPerSample"] != 8:
        raise ValueError("unsupported sampled function bit depth")

    size_obj = dictionary.get("Size")
    domain_obj = dictionary.get("Domain")
    range_obj = dictionary.get("Range")
    if not isinstance(size_obj, (list, tuple)):
        raise ValueError("invalid sampled function size")
    if any(type(value) is not int for value in size_obj):
        raise ValueError("invalid sampled function size")
    sizes = tuple(cast(int, value) for value in size_obj)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("invalid sampled function size")
    domain_values = require_pdf_number_array(domain_obj, "invalid PDF function domain")
    if len(domain_values) != len(sizes) * 2:
        raise ValueError("invalid sampled function domain")
    domains = tuple(zip(domain_values[::2], domain_values[1::2], strict=True))
    if any(upper <= lower for lower, upper in domains):
        raise ValueError("invalid sampled function domain")

    range_values = require_pdf_number_array(range_obj, "invalid sampled function range")
    if not range_values or len(range_values) % 2:
        raise ValueError("invalid sampled function range")
    ranges = tuple(zip(range_values[::2], range_values[1::2], strict=True))
    if any(upper < lower for lower, upper in ranges):
        raise ValueError("invalid sampled function range")

    order = dictionary.get("Order")
    if order is None:
        order = 1
    if type(order) is not int or order not in {1, 3}:
        raise ValueError("invalid sampled function interpolation order")
    if order == 3 and any(size >= 4 for size in sizes):
        raise ValueError("unsupported cubic sampled function interpolation")

    decode_obj = dictionary.get("Decode")
    if decode_obj is None:
        decodes = tuple(ranges)
    else:
        decode_values = require_pdf_number_array(decode_obj, "invalid sampled function decode")
        if len(decode_values) != len(ranges) * 2:
            raise ValueError("invalid sampled function decode")
        decodes = tuple(
            (decode_values[index * 2], decode_values[index * 2 + 1]) for index in range(len(ranges))
        )

    encode_obj = dictionary.get("Encode")
    encode_values = (
        require_pdf_number_array(encode_obj, "invalid sampled function encode")
        if encode_obj is not None
        else ()
    )
    if encode_obj is not None and len(encode_values) != len(sizes) * 2:
        raise ValueError("invalid sampled function encode")
    encodes: list[tuple[float, float]] = []
    for input_index, size in enumerate(sizes):
        lower = 0.0
        upper = float(size - 1)
        if encode_values:
            lower = encode_values[input_index * 2]
            upper = encode_values[input_index * 2 + 1]
        encodes.append((lower, upper))

    sample_count = 1
    for size in sizes:
        sample_count *= size
    samples = function.data
    if len(samples) < sample_count * len(ranges):
        raise ValueError("invalid sampled function data")

    def evaluate(*inputs: float) -> tuple[float, ...]:
        if len(inputs) != len(sizes):
            raise ValueError("invalid sampled function input count")
        corners: list[tuple[int, float]] = [(0, 1.0)]
        stride = 1
        for raw_input, size, domain, encode in zip(inputs, sizes, domains, encodes, strict=True):
            domain_min, domain_max = domain
            clipped = max(domain_min, min(domain_max, raw_input))
            normalized = (clipped - domain_min) / (domain_max - domain_min)
            encoded = encode[0] + normalized * (encode[1] - encode[0])
            encoded = max(0.0, min(float(size - 1), encoded))
            lower_index = int(encoded)
            upper_index = min(size - 1, lower_index + 1)
            fraction = encoded - lower_index
            if upper_index == lower_index or fraction == 0.0:
                corners = [
                    (sample_index + lower_index * stride, weight)
                    for sample_index, weight in corners
                ]
            else:
                corners = [
                    corner
                    for sample_index, weight in corners
                    for corner in (
                        (sample_index + lower_index * stride, weight * (1.0 - fraction)),
                        (sample_index + upper_index * stride, weight * fraction),
                    )
                ]
            stride *= size

        return tuple(
            max(
                range_min,
                min(
                    range_max,
                    decode_min
                    + sum(
                        samples[sample_index * len(ranges) + output_index] * weight
                        for sample_index, weight in corners
                    )
                    / 255.0
                    * (decode_max - decode_min),
                ),
            )
            for output_index, ((range_min, range_max), (decode_min, decode_max)) in enumerate(
                zip(ranges, decodes, strict=True)
            )
        )

    return evaluate


def internal_function_type(dictionary: dict[Any, Any]) -> int:
    return require_pdf_integer(dictionary.get("FunctionType"), "invalid PDF function type")


def compile_pdf_function(
    function: Any, *, compile_nested: Callable[[Any], PdfFunctionEvaluator] | None = None
) -> PdfFunctionEvaluator:
    compile_child = compile_nested or compile_pdf_function
    if isinstance(function, (list, tuple)):
        if function and all(isinstance(part, (dict, PdfStream)) for part in function):
            parts = tuple(compile_child(part) for part in function)

            def evaluate_array(*inputs: float) -> tuple[float, ...]:
                outputs: list[float] = []
                for part in parts:
                    outputs.extend(part(*inputs))
                return tuple(outputs) if outputs else tuple(inputs)

            return evaluate_array
        raise ValueError("invalid PDF function array")

    if isinstance(function, PdfStream):
        function_type = internal_function_type(function.dictionary)
        if function_type == 4:
            return internal_compile_calculator_function(function)
        if function_type == 0:
            try:
                return internal_compile_sampled_function(function)
            except Exception as exc:
                raise ValueError("invalid sampled PDF function") from exc
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = internal_function_type(function)
        dictionary = function
    else:
        raise ValueError("invalid PDF function")

    if function_type == 2:
        exponent = require_pdf_number(dictionary.get("N"), "invalid exponential PDF function")
        domain_min, domain_max = internal_scalar_domain(dictionary)
        c0 = list(
            require_pdf_number_array(
                dictionary.get("C0", (0.0,)), "invalid exponential function components"
            )
        )
        c1 = list(
            require_pdf_number_array(
                dictionary.get("C1", (1.0,)), "invalid exponential function components"
            )
        )
        if not c0 or not c1:
            raise ValueError("invalid exponential function components")
        if len(c0) != len(c1):
            raise ValueError("mismatched exponential function components")
        count = len(c0)
        start_values = tuple(c0)
        deltas = tuple(c1[index] - c0[index] for index in range(count))

        def evaluate_exponential(*inputs: float) -> tuple[float, ...]:
            if len(inputs) != 1:
                raise ValueError("invalid exponential function input count")
            value = max(domain_min, min(domain_max, inputs[0]))
            factor = math.pow(value, exponent)
            return tuple(
                start_value + factor * delta
                for start_value, delta in zip(start_values, deltas, strict=True)
            )

        return with_pdf_function_range(
            evaluate_exponential, dictionary.get("Range"), output_count=count
        )

    if function_type == 3:
        functions = dictionary.get("Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            raise ValueError("invalid stitching PDF function")
        domain_min, domain_max = internal_scalar_domain(dictionary)
        bounds_obj = dictionary.get("Bounds")
        if not isinstance(bounds_obj, (list, tuple)):
            raise ValueError("invalid stitching function bounds")
        bounds = require_pdf_number_array(bounds_obj, "invalid stitching function bounds")
        encode = require_pdf_number_array(
            dictionary.get("Encode"), "invalid stitching function parameters"
        )
        parts = tuple(compile_child(entry) for entry in functions)
        if len(bounds) != len(parts) - 1 or len(encode) != len(parts) * 2:
            raise ValueError("invalid stitching function parameters")
        previous = domain_min
        for bound in bounds:
            if not previous < bound <= domain_max:
                raise ValueError("invalid stitching function bounds")
            previous = bound

        def evaluate_stitching(*inputs: float) -> tuple[float, ...]:
            if len(inputs) != 1:
                raise ValueError("invalid stitching function input count")
            value = max(domain_min, min(domain_max, inputs[0]))
            index = bisect_right(bounds, value)
            low = bounds[index - 1] if index > 0 else domain_min
            high = bounds[index] if index < len(bounds) else domain_max
            enc0 = encode[index * 2]
            enc1 = encode[index * 2 + 1]
            encoded = enc0 if high == low else enc0 + (value - low) * (enc1 - enc0) / (high - low)
            return parts[index](encoded)

        return with_pdf_function_range(evaluate_stitching, dictionary.get("Range"))

    raise ValueError(f"unsupported PDF function type: {function_type}")


__all__ = (
    "PdfFunctionEvaluator",
    "with_pdf_function_range",
    "compile_pdf_function",
)
