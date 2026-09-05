# SPDX-License-Identifier: AGPL-3.0-only
"""Compile PDF functions into reusable numeric evaluators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int

internal_PdfFunctionEvaluator = Callable[..., tuple[float, ...]]


def internal_identity_function(*inputs: float) -> tuple[float, ...]:
    return tuple(inputs)


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


def internal_compile_sampled_function(function: PdfStream) -> internal_PdfFunctionEvaluator:
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
        if lower is None or upper is None or upper == lower:
            raise ValueError("invalid sampled function domain")
        domains.append((lower, upper))

    ranges: list[tuple[float, float]] = []
    for output_index in range(len(range_obj) // 2):
        lower = parse_float(range_obj[output_index * 2], None)
        upper = parse_float(range_obj[output_index * 2 + 1], None)
        if lower is None or upper is None:
            raise ValueError("invalid sampled function range")
        ranges.append((lower, upper))

    encode_obj = dictionary.get("Encode")
    encodes: list[tuple[float, float]] = []
    for input_index, size in enumerate(sizes):
        lower = 0.0
        upper = float(size - 1)
        if isinstance(encode_obj, (list, tuple)) and len(encode_obj) >= len(sizes) * 2:
            parsed_lower = parse_float(encode_obj[input_index * 2], None)
            parsed_upper = parse_float(encode_obj[input_index * 2 + 1], None)
            if parsed_lower is not None and parsed_upper is not None:
                lower = parsed_lower
                upper = parsed_upper
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
        sample_index = 0
        stride = 1
        for index, (raw_input, size, domain, encode) in enumerate(
            zip(inputs, sizes, domains, encodes, strict=True)
        ):
            domain_min, domain_max = domain
            clipped = max(domain_min, min(domain_max, raw_input))
            normalized = (clipped - domain_min) / (domain_max - domain_min)
            encoded = encode[0] + normalized * (encode[1] - encode[0])
            position = max(0, min(size - 1, int(round(encoded))))
            if index > 0:
                stride *= sizes[index - 1]
            sample_index += position * stride

        base = sample_index * len(ranges)
        return tuple(
            max(lower, min(upper, lower + samples[base + index] / 255.0 * (upper - lower)))
            for index, (lower, upper) in enumerate(ranges)
        )

    return evaluate


def internal_compile_pdf_function(function: Any) -> internal_PdfFunctionEvaluator:
    """Normalize a supported PDF Function into a reusable evaluator."""
    if isinstance(function, (list, tuple)):
        if function and all(
            isinstance(part, (dict, PdfStream)) or callable(part) for part in function
        ):
            parts = tuple(internal_compile_pdf_function(part) for part in function)

            def evaluate_array(*inputs: float) -> tuple[float, ...]:
                outputs: list[float] = []
                for part in parts:
                    outputs.extend(part(*inputs))
                return tuple(outputs) if outputs else tuple(inputs)

            return evaluate_array
        try:
            constants = tuple(float(value) for value in function)
        except (TypeError, ValueError):
            return internal_identity_function
        return lambda *inputs: constants

    if callable(function):

        def evaluate_callable(*inputs: float) -> tuple[float, ...]:
            result = function(*inputs)
            if isinstance(result, (list, tuple)):
                return tuple(float(value) for value in result)
            return (float(result),)

        return evaluate_callable

    if isinstance(function, PdfStream):
        function_type = parse_int(function.dictionary.get("FunctionType"), -1)
        if function_type == 0:
            try:
                return internal_compile_sampled_function(function)
            except Exception:
                return internal_identity_function
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = parse_int(function.get("FunctionType"), -1)
        dictionary = function
    else:
        return internal_identity_function

    if function_type == 2:
        exponent = parse_float(dictionary.get("N"), 1.0)
        if exponent is None:
            return internal_identity_function
        c0 = list(internal_number_array(dictionary.get("C0")) or (0.0,))
        c1 = list(internal_number_array(dictionary.get("C1")) or (1.0,))
        count = max(len(c0), len(c1))
        if len(c0) < count:
            c0.extend([c0[-1] if c0 else 0.0] * (count - len(c0)))
        if len(c1) < count:
            c1.extend([c1[-1] if c1 else 1.0] * (count - len(c1)))
        start_values = tuple(c0)
        deltas = tuple(c1[index] - c0[index] for index in range(count))

        def evaluate_exponential(*inputs: float) -> tuple[float, ...]:
            if len(inputs) != 1:
                raise ValueError("invalid exponential function input count")
            factor = inputs[0] ** exponent
            return tuple(
                start_value + factor * delta
                for start_value, delta in zip(start_values, deltas, strict=True)
            )

        return evaluate_exponential

    if function_type == 3:
        functions = dictionary.get("Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            return internal_identity_function
        bounds = internal_number_array(dictionary.get("Bounds"))
        encode = internal_number_array(dictionary.get("Encode"))
        parts = tuple(internal_compile_pdf_function(entry) for entry in functions)

        def evaluate_stitching(*inputs: float) -> tuple[float, ...]:
            if len(inputs) != 1:
                raise ValueError("invalid stitching function input count")
            value = inputs[0]
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

    return internal_identity_function


def internal_evaluate_pdf_function(function: Any, *inputs: float) -> tuple[float, ...]:
    """Evaluate a supported PDF Function without retaining its compiled form."""
    return internal_compile_pdf_function(function)(*inputs)
