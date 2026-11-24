# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.10: functions: sampled, exponential, stitching and PostScript calculator."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from typing import Any

lazy import numpy

from core_pdf.impl.postscript.calculator import bounds, compile_calculator
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    require_pdf_integer,
    require_pdf_number,
    require_pdf_number_array,
    require_pdf_number_pairs,
)

PdfFunctionEvaluator = Callable[..., tuple[float, ...]]
# The batch form of an evaluator: from a float64 array of inputs, (n,) for one input each or
# (n, m) for m, to a float64 array of their outputs, one row each, found with the same float
# operations in the same order, so equal bit for bit to calling the evaluator on each. It
# raises where the evaluator would raise for one of them, or where it cannot be sure of the
# same result (a NaN), so the caller then evaluates one at a time.
type BatchEvaluator = Callable[[Any], Any]
# The bit depths a sampled function's samples may have (7.10.2).
SAMPLE_BITS = frozenset((1, 2, 4, 8, 12, 16, 24, 32))


def batch_evaluator(evaluate: object) -> BatchEvaluator | None:
    """The batch form of evaluate, or None when it has none."""
    return getattr(evaluate, "batch", None)


def with_batch(evaluate: PdfFunctionEvaluator, batch: BatchEvaluator) -> PdfFunctionEvaluator:
    setattr(evaluate, "batch", batch)  # noqa: B010 - an attribute of a function object
    return evaluate


def clipped(values: Any, lower: Any, upper: Any) -> Any:
    """max(lower, min(upper, value)) of each value, as Python finds it, -0.0 and NaN included:
    min keeps its first argument unless the second is less, and max unless it is greater."""
    below = numpy.where(values < upper, values, upper)
    return numpy.where(below > lower, below, lower)


def clamped_inputs(inputs: Any, low: float, high: float) -> Any:
    """max(low, min(high, value)) of each input of a one-input function, which must all be
    finite."""
    values = numpy.asarray(inputs, dtype=numpy.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not numpy.isfinite(values).all():
        raise ValueError("a batch of PDF function inputs must be finite numbers")
    return clipped(values, low, high)


def calculator_bounds(value: object, kind: str) -> tuple[tuple[float, float], ...]:
    return bounds(require_pdf_number_pairs(value, f"invalid calculator {kind}"), kind)


def with_pdf_function_range(
    evaluate: PdfFunctionEvaluator, range_values: Any, *, output_count: int | None = None
) -> PdfFunctionEvaluator:
    if range_values is None:
        return evaluate
    ranges = require_pdf_number_pairs(range_values, "invalid PDF function range")
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

    inner = batch_evaluator(evaluate)
    if inner is None:
        return evaluate_clipped
    lowers = numpy.asarray([lower for lower, _ in ranges], dtype=numpy.float64)
    uppers = numpy.asarray([upper for _, upper in ranges], dtype=numpy.float64)

    def batch_clipped(inputs: Any) -> Any:
        outputs = inner(inputs)
        # A NaN output is left to one at a time, which finds min(upper, NaN) is upper.
        if outputs.shape[1] != len(ranges) or numpy.isnan(outputs).any():
            raise ValueError("PDF function range has incorrect output count")
        return clipped(outputs, lowers, uppers)

    return with_batch(evaluate_clipped, batch_clipped)


def scalar_domain(dictionary: dict[Any, Any]) -> tuple[float, float]:
    domain_obj = dictionary.get("Domain")
    if domain_obj is None:
        raise ValueError("missing PDF function domain")
    domain = require_pdf_number_array(domain_obj, "invalid PDF function domain")
    if len(domain) != 2 or domain[1] < domain[0]:
        raise ValueError("invalid PDF function domain")
    return (domain[0], domain[1])


def sample_values(data: bytes, bits: int, count: int) -> Any:
    """The first count samples of bits each, packed with no padding (7.10.2), as uint64."""
    if len(data) * 8 < count * bits:
        raise ValueError("invalid sampled function data")
    if bits in {8, 16, 32}:
        return numpy.frombuffer(data, dtype=f">u{bits // 8}", count=count).astype(numpy.uint64)
    packed = numpy.frombuffer(data, dtype=numpy.uint8, count=(count * bits + 7) // 8)
    digits = numpy.unpackbits(packed)[: count * bits].reshape(count, bits).astype(numpy.uint64)
    return digits @ (numpy.uint64(1) << numpy.arange(bits - 1, -1, -1, dtype=numpy.uint64))


def neumaier_sum(terms: list[Any]) -> Any:
    """sum() of the terms, elementwise, as Python (3.12 and later) sums floats: with Neumaier's
    compensation, added at the end when it is finite and not zero. Every term must be finite,
    as a sampled function's are."""
    total = terms[0] + 0.0
    compensation = numpy.zeros_like(total)
    for term in terms[1:]:
        step = total + term
        compensation += numpy.where(
            numpy.abs(total) >= numpy.abs(term), (total - step) + term, (term - step) + total
        )
        total = step
    add = (compensation != 0.0) & numpy.isfinite(compensation)
    return numpy.where(add, total + compensation, total)


def compile_sampled_function(function: PdfStream) -> PdfFunctionEvaluator:
    dictionary = function.dictionary
    if require_function_type(dictionary) != 0:
        raise ValueError("invalid sampled function")
    bits = dictionary.get("BitsPerSample")
    if type(bits) is not int or bits not in SAMPLE_BITS:
        raise ValueError("unsupported sampled function bit depth")

    size_obj = dictionary.get("Size")
    domain_obj = dictionary.get("Domain")
    range_obj = dictionary.get("Range")
    if not isinstance(size_obj, (list, tuple)):
        raise ValueError("invalid sampled function size")
    if any(type(value) is not int for value in size_obj):
        raise ValueError("invalid sampled function size")
    sizes: tuple[int, ...] = tuple(size_obj)  # type: ignore[arg-type]  # ty: ignore[invalid-assignment]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("invalid sampled function size")
    domains = require_pdf_number_pairs(
        domain_obj,
        "invalid PDF function domain",
        count=len(sizes),
        length_message="invalid sampled function domain",
    )
    if any(upper <= lower for lower, upper in domains):
        raise ValueError("invalid sampled function domain")

    ranges = require_pdf_number_pairs(range_obj, "invalid sampled function range")
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
        decodes = require_pdf_number_pairs(
            decode_obj, "invalid sampled function decode", count=len(ranges)
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
    table = sample_values(function.data, bits, sample_count * len(ranges))
    samples: list[int] = table.tolist()
    # The largest sample, which Decode maps to its upper bound: 255.0 for 8 bits.
    maximum = float((1 << bits) - 1)

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
                    / maximum
                    * (decode_max - decode_min),
                ),
            )
            for output_index, ((range_min, range_max), (decode_min, decode_max)) in enumerate(
                zip(ranges, decodes, strict=True)
            )
        )

    rows = table.astype(numpy.float64).reshape(sample_count, len(ranges))
    strides = [1]
    for size in sizes[:-1]:
        strides.append(strides[-1] * size)
    decode_lows = numpy.asarray([low for low, _ in decodes], dtype=numpy.float64)
    decode_spans = numpy.asarray([high - low for low, high in decodes], dtype=numpy.float64)
    range_lows = numpy.asarray([low for low, _ in ranges], dtype=numpy.float64)
    range_highs = numpy.asarray([high for _, high in ranges], dtype=numpy.float64)

    def batch_sampled(inputs: Any) -> Any:
        values = numpy.asarray(inputs, dtype=numpy.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != len(sizes):
            raise ValueError("invalid sampled function input count")
        lowers: list[Any] = []
        uppers: list[Any] = []
        fractions: list[Any] = []
        with numpy.errstate(all="ignore"):
            for column, size, (domain_min, domain_max), (start, end) in zip(
                values.T, sizes, domains, encodes, strict=True
            ):
                normalized = (clipped(column, domain_min, domain_max) - domain_min) / (
                    domain_max - domain_min
                )
                encoded = clipped(start + normalized * (end - start), 0.0, float(size - 1))
                lower = encoded.astype(numpy.int64)
                lowers.append(lower)
                uppers.append(numpy.minimum(size - 1, lower + 1))
                fractions.append(encoded - lower)
            # Every corner of the cell, in the order one at a time visits them: the first
            # input's index the slowest to change. Along an input whose fraction is 0, one at
            # a time keeps the lower corner alone; here the upper one has weight 0.0, and its
            # 0.0 term changes neither the sum nor its compensation.
            count = len(sizes)
            terms: list[Any] = []
            for corner in range(1 << count):
                offset = numpy.zeros(len(values), dtype=numpy.int64)
                weight = numpy.ones(len(values), dtype=numpy.float64)
                for axis in range(count):
                    if corner >> (count - 1 - axis) & 1:
                        offset += uppers[axis] * strides[axis]
                        weight = weight * fractions[axis]
                    else:
                        offset += lowers[axis] * strides[axis]
                        weight = weight * (1.0 - fractions[axis])
                terms.append(rows[offset] * weight[:, None])
            decoded = decode_lows + neumaier_sum(terms) / maximum * decode_spans
        return clipped(decoded, range_lows, range_highs)

    return with_batch(evaluate, batch_sampled)


def require_function_type(dictionary: dict[Any, Any]) -> int:
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

            part_batches = [
                batch for batch in (batch_evaluator(part) for part in parts) if batch is not None
            ]
            if len(part_batches) != len(parts):
                return evaluate_array

            def batch_array(inputs: Any) -> Any:
                found = [batch(inputs) for batch in part_batches]
                outputs = numpy.concatenate(found, axis=1)
                if not outputs.shape[1]:
                    raise ValueError("PDF function array has no outputs")
                return outputs

            return with_batch(evaluate_array, batch_array)
        raise ValueError("invalid PDF function array")

    if isinstance(function, PdfStream):
        function_type = require_function_type(function.dictionary)
        if function_type == 4:
            domains = calculator_bounds(function.dictionary.get("Domain"), "domain")
            ranges = calculator_bounds(function.dictionary.get("Range"), "range")
            return compile_calculator(function.data, domains, ranges)
        if function_type == 0:
            try:
                return compile_sampled_function(function)
            except Exception as exc:
                raise ValueError("invalid sampled PDF function") from exc
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = require_function_type(function)
        dictionary = function
    else:
        raise ValueError("invalid PDF function")

    if function_type == 2:
        exponent = require_pdf_number(dictionary.get("N"), "invalid exponential PDF function")
        domain_min, domain_max = scalar_domain(dictionary)
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

        starts = numpy.asarray(start_values, dtype=numpy.float64)
        steps = numpy.asarray(deltas, dtype=numpy.float64)

        def batch_exponential(inputs: Any) -> Any:
            values = clamped_inputs(inputs, domain_min, domain_max)
            if exponent == 1:
                # math.pow(value, 1.0) is value.
                factors = values
            else:
                # math.pow for each, raising as one call would; numpy.power may differ in an ulp.
                factors = numpy.asarray(
                    [math.pow(value, exponent) for value in values.tolist()], dtype=numpy.float64
                )
            return starts + factors[:, None] * steps

        return with_pdf_function_range(
            with_batch(evaluate_exponential, batch_exponential),
            dictionary.get("Range"),
            output_count=count,
        )

    if function_type == 3:
        functions = dictionary.get("Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            raise ValueError("invalid stitching PDF function")
        domain_min, domain_max = scalar_domain(dictionary)
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

        part_batches = [
            batch for batch in (batch_evaluator(part) for part in parts) if batch is not None
        ]
        if len(part_batches) != len(parts):
            return with_pdf_function_range(evaluate_stitching, dictionary.get("Range"))
        lows = numpy.asarray([domain_min, *bounds], dtype=numpy.float64)
        highs = numpy.asarray([*bounds, domain_max], dtype=numpy.float64)
        bound_array = numpy.asarray(bounds, dtype=numpy.float64)
        encode_starts = numpy.asarray(encode[0::2], dtype=numpy.float64)
        encode_ends = numpy.asarray(encode[1::2], dtype=numpy.float64)

        def batch_stitching(inputs: Any) -> Any:
            values = clamped_inputs(inputs, domain_min, domain_max)
            index = numpy.searchsorted(bound_array, values, side="right")
            low = lows[index]
            high = highs[index]
            enc0 = encode_starts[index]
            enc1 = encode_ends[index]
            with numpy.errstate(divide="ignore", invalid="ignore"):
                encoded = numpy.where(
                    high == low, enc0, enc0 + (values - low) * (enc1 - enc0) / (high - low)
                )
            outputs: Any = None
            for part in numpy.unique(index).tolist():
                chosen = index == part
                found = part_batches[part](encoded[chosen])
                if outputs is None:
                    outputs = numpy.empty((len(values), found.shape[1]), dtype=numpy.float64)
                elif found.shape[1] != outputs.shape[1]:
                    raise ValueError("stitched PDF functions differ in output count")
                outputs[chosen] = found
            if outputs is None:
                return numpy.empty((0, 0), dtype=numpy.float64)
            return outputs

        return with_pdf_function_range(
            with_batch(evaluate_stitching, batch_stitching), dictionary.get("Range")
        )

    raise ValueError(f"unsupported PDF function type: {function_type}")


__all__ = (
    "BatchEvaluator",
    "PdfFunctionEvaluator",
    "batch_evaluator",
    "compile_pdf_function",
    "with_pdf_function_range",
)
