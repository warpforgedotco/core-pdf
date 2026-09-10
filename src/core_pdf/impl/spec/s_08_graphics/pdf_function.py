# SPDX-License-Identifier: AGPL-3.0-only
"""Compile PDF functions into reusable numeric evaluators."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int
from core_pdf.impl.spec.s_08_graphics.pdf_calculator import compile_calculator_function

internal_PdfFunctionEvaluator = Callable[..., tuple[float, ...]]


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


def internal_scalar_domain(dictionary: dict[Any, Any]) -> tuple[float, float]:
    """Return the one-input domain used by Type 2 and Type 3 functions."""
    domain_obj = dictionary.get("Domain")
    if domain_obj is None:
        raise ValueError("missing PDF function domain")
    domain = internal_number_array(domain_obj)
    if len(domain) < 2 or domain[1] < domain[0]:
        raise ValueError("invalid PDF function domain")
    return (domain[0], domain[1])


def internal_compile_sampled_function(
    function: PdfStream, *, coordinate_rounding: Callable[[float], float] | None = None
) -> internal_PdfFunctionEvaluator:
    """Compile an 8-bit sampled function, decoding its stream exactly once."""
    dictionary = function.dictionary
    if parse_int(dictionary.get("FunctionType"), -1) != 0:
        raise ValueError("invalid sampled function")
    if parse_int(dictionary.get("BitsPerSample"), 0) != 8:
        raise ValueError("unsupported sampled function bit depth")

    size_obj = dictionary.get("Size")
    domain_obj = dictionary.get("Domain")
    range_obj = dictionary.get("Range")
    if not isinstance(size_obj, (list, tuple)):
        raise ValueError("invalid sampled function size")
    if not isinstance(domain_obj, (list, tuple)):
        raise ValueError("invalid sampled function domain")
    if not isinstance(range_obj, (list, tuple)) or len(range_obj) < 2:
        raise ValueError("invalid sampled function range")

    sizes = tuple(parse_int(value, 0) or 0 for value in size_obj)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("invalid sampled function size")
    if len(domain_obj) < len(sizes) * 2:
        raise ValueError("invalid sampled function domain")

    domains: list[tuple[float, float]] = []
    for input_index in range(len(sizes)):
        lower = parse_float(domain_obj[input_index * 2], None)
        upper = parse_float(domain_obj[input_index * 2 + 1], None)
        if lower is None or upper is None or upper <= lower:
            raise ValueError("invalid sampled function domain")
        domains.append((lower, upper))

    ranges: list[tuple[float, float]] = []
    for output_index in range(len(range_obj) // 2):
        lower = parse_float(range_obj[output_index * 2], None)
        upper = parse_float(range_obj[output_index * 2 + 1], None)
        if lower is None or upper is None:
            raise ValueError("invalid sampled function range")
        ranges.append((lower, upper))

    decode_obj = dictionary.get("Decode")
    if decode_obj is None:
        decodes = tuple(ranges)
    else:
        decode_values = internal_number_array(decode_obj)
        if len(decode_values) < len(ranges) * 2:
            raise ValueError("invalid sampled function decode")
        decodes = tuple(
            (decode_values[index * 2], decode_values[index * 2 + 1]) for index in range(len(ranges))
        )

    encode_obj = dictionary.get("Encode")
    encode_values = internal_number_array(encode_obj) if encode_obj is not None else ()
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
            if coordinate_rounding is not None:
                encoded = coordinate_rounding(encoded)
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


def internal_compile_pdf_function(
    function: Any, *, compile_nested: Callable[[Any], internal_PdfFunctionEvaluator] | None = None
) -> internal_PdfFunctionEvaluator:
    """Normalize a supported PDF Function into a reusable evaluator."""
    compile_child = compile_nested or internal_compile_pdf_function
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
        function_type = parse_int(function.dictionary.get("FunctionType"), -1)
        if function_type == 0:
            try:
                return internal_compile_sampled_function(function)
            except Exception as exc:
                raise ValueError("invalid sampled PDF function") from exc
        if function_type == 4:
            return compile_calculator_function(
                function.data,
                internal_number_array(function.dictionary.get("Domain")),
                internal_number_array(function.dictionary.get("Range")),
            )
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = parse_int(function.get("FunctionType"), -1)
        dictionary = function
    else:
        raise ValueError("invalid PDF function")

    if function_type == 2:
        exponent = parse_float(dictionary.get("N"), None)
        if exponent is None:
            raise ValueError("invalid exponential PDF function")
        domain_min, domain_max = internal_scalar_domain(dictionary)
        c0 = list(internal_number_array(dictionary.get("C0", (0.0,))))
        c1 = list(internal_number_array(dictionary.get("C1", (1.0,))))
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

        return evaluate_exponential

    if function_type == 3:
        functions = dictionary.get("Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            raise ValueError("invalid stitching PDF function")
        domain_min, domain_max = internal_scalar_domain(dictionary)
        bounds = internal_number_array(dictionary.get("Bounds"))
        encode = internal_number_array(dictionary.get("Encode"))
        parts = tuple(compile_child(entry) for entry in functions)
        if len(bounds) != len(parts) - 1 or len(encode) != len(parts) * 2:
            raise ValueError("invalid stitching function parameters")

        def evaluate_stitching(*inputs: float) -> tuple[float, ...]:
            if len(inputs) != 1:
                raise ValueError("invalid stitching function input count")
            value = max(domain_min, min(domain_max, inputs[0]))
            index = 0
            while index < len(bounds) and value >= bounds[index]:
                index += 1
            low = bounds[index - 1] if index > 0 else domain_min
            high = bounds[index] if index < len(bounds) else domain_max
            enc0 = encode[index * 2]
            enc1 = encode[index * 2 + 1]
            encoded = enc0 if high == low else enc0 + (value - low) * (enc1 - enc0) / (high - low)
            return parts[index](encoded)

        return evaluate_stitching

    raise ValueError(f"unsupported PDF function type: {function_type}")
