# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any

import core_pdf_spec.s_08_graphics.pdf_function as strict
from core_pdf.impl.runtime.scalars import parse_float, parse_int
from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float as parse_pdf_float
from core_pdf_spec.s_08_graphics.pdf_function import (
    PdfFunctionEvaluator,
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


def internal_sampled_number_array(values: list[Any] | tuple[Any, ...]) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        parsed = parse_pdf_float(value, None)
        if parsed is None:
            return ()
        output.append(parsed)
    return tuple(output)


def internal_compile_unordered_stitching(
    functions: list[Any],
    domain: tuple[float, float],
    bounds: tuple[float, ...],
    encode: tuple[float, ...],
) -> PdfFunctionEvaluator:
    parts = tuple(internal_compile_pdf_function(entry) for entry in functions)
    domain_min, domain_max = domain

    def evaluate(*inputs: float) -> tuple[float, ...]:
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

    return evaluate


def internal_compile_pdf_function(function: Any) -> PdfFunctionEvaluator:
    if callable(function):

        def evaluate_callable(*inputs: float) -> tuple[float, ...]:
            result = function(*inputs)
            return (
                tuple(float(value) for value in result)
                if isinstance(result, (list, tuple))
                else (float(result),)
            )

        return evaluate_callable
    if isinstance(function, (list, tuple)):
        if function and all(
            isinstance(part, (dict, PdfStream)) or callable(part) for part in function
        ):
            parts = tuple(internal_compile_pdf_function(part) for part in function)

            def evaluate_array(*inputs: float) -> tuple[float, ...]:
                return tuple(value for part in parts for value in part(*inputs)) or tuple(inputs)

            return evaluate_array
        try:
            constants = tuple(float(value) for value in function)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid PDF function") from exc
        if not constants:
            raise ValueError("invalid PDF function")
        return lambda *_inputs: constants
    source_dictionary = function.dictionary if isinstance(function, PdfStream) else function
    if not isinstance(source_dictionary, dict):
        raise ValueError("invalid PDF function")
    dictionary = dict(source_dictionary)
    kind = parse_int(dictionary.get("FunctionType"), -1)
    dictionary["FunctionType"] = kind
    if kind in {2, 3}:
        dictionary.pop("Range", None)
    for name in ("Domain", "Range", "Decode"):
        values = internal_number_array(dictionary.get(name))
        if values:
            dictionary[name] = values
    if kind == 0:
        dictionary["Order"] = 1
        dictionary["BitsPerSample"] = parse_int(dictionary.get("BitsPerSample"), 0)
        sizes = dictionary.get("Size")
        if isinstance(sizes, (list, tuple)):
            sizes = tuple(parse_int(value, 0) or 0 for value in sizes)
            dictionary["Size"] = sizes
            domain = dictionary.get("Domain")
            if isinstance(domain, (list, tuple)):
                dictionary["Domain"] = internal_sampled_number_array(domain[: 2 * len(sizes)])
            raw_encode = dictionary.get("Encode")
            encodes: list[float] = []
            for index, size in enumerate(sizes):
                lower, upper = 0.0, float(size - 1)
                if isinstance(raw_encode, (list, tuple)) and len(raw_encode) >= len(sizes) * 2:
                    parsed_lower = parse_float(raw_encode[index * 2], None)
                    parsed_upper = parse_float(raw_encode[index * 2 + 1], None)
                    if parsed_lower is not None and parsed_upper is not None:
                        lower, upper = parsed_lower, parsed_upper
                encodes.extend((lower, upper))
            dictionary["Encode"] = encodes
        raw_range = dictionary.get("Range")
        if isinstance(raw_range, (list, tuple)):
            ranges = internal_sampled_number_array(raw_range[: len(raw_range) // 2 * 2])
            dictionary["Range"] = tuple(
                value
                for lower, upper in zip(ranges[::2], ranges[1::2], strict=True)
                for value in (lower, max(lower, upper))
            )
            if dictionary.get("Decode") is None:
                dictionary["Decode"] = ranges
            else:
                decodes = internal_number_array(dictionary["Decode"])
                if decodes and all(math.isfinite(value) for value in decodes):
                    dictionary["Decode"] = decodes[: len(ranges)]
    if kind in {2, 3} and dictionary.get("Domain") is None:
        dictionary["Domain"] = [0.0, 1.0]
    elif kind in {2, 3}:
        dictionary["Domain"] = internal_number_array(dictionary.get("Domain"))[:2]
    if kind == 2:
        if dictionary.get("N") is None:
            dictionary["N"] = 1.0
        else:
            dictionary["N"] = parse_float(dictionary["N"], None)
        c0 = list(internal_number_array(dictionary.get("C0")) or (0.0,))
        c1 = list(internal_number_array(dictionary.get("C1")) or (1.0,))
        count = max(len(c0), len(c1))
        c0.extend([c0[-1]] * (count - len(c0)))
        c1.extend([c1[-1]] * (count - len(c1)))
        dictionary.update(C0=c0, C1=c1)
    if kind == 3:
        functions = dictionary.get("Functions")
        if isinstance(functions, (list, tuple)) and functions:
            bounds = internal_number_array(dictionary.get("Bounds"))
            count = len(bounds) + 1
            functions = list(functions[:count])
            functions.extend([functions[-1]] * (count - len(functions)))
            encode = internal_number_array(dictionary.get("Encode"))
            encode = tuple(
                encode[index] if index < len(encode) else float(index % 2)
                for index in range(2 * count)
            )
            dictionary.update(
                Functions=functions,
                Bounds=bounds,
                Encode=encode,
            )
            domain = internal_number_array(dictionary["Domain"])
            if (
                len(domain) == 2
                and domain[0] <= domain[1]
                and all(math.isfinite(value) for value in (*domain, *bounds, *encode))
                and any(
                    lower >= upper or upper > domain[1]
                    for lower, upper in zip((domain[0], *bounds), bounds)
                )
            ):
                return internal_compile_unordered_stitching(
                    functions, (domain[0], domain[1]), bounds, encode
                )
    prepared = (
        function.replace(dictionary=dictionary, spec=function.spec)
        if isinstance(function, PdfStream)
        else dictionary
    )
    if kind in {0, 4} and isinstance(prepared, PdfStream):
        try:
            decoded = prepared.data
        except (
            TypeError,
            ValueError,
            ArithmeticError,
            FilterError,
            PdfParseError,
            PdfUnsupportedError,
        ) as error:
            label = "sampled" if kind == 0 else "calculator"
            raise ValueError(f"invalid {label} PDF function") from error
        prepared = prepared.replace(raw_data=decoded, spec=None, decoder=None)
    return strict.compile_pdf_function(prepared, compile_nested=internal_compile_pdf_function)
