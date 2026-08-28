# SPDX-License-Identifier: AGPL-3.0-only
"""Native image sample and decode-array kernels."""

from __future__ import annotations

from functools import lru_cache

import numpy

from core_pdf.impl.runtime.array_views import ByteBuffer, uint8_view
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float, parse_int
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec

ImageDict = dict[str, object]
ImageBuffer = ByteBuffer


@lru_cache(maxsize=1024)
def decode_translation_tables_8bit(
    pairs: tuple[tuple[float, float], ...],
) -> tuple[bytes, ...]:
    values = numpy.arange(256, dtype=numpy.float64) / 255.0
    tables: list[bytes] = []
    for dmin, dmax in pairs:
        span = dmax - dmin
        decoded = numpy.rint((dmin + values * span) * 255.0)
        tables.append(numpy.clip(decoded, 0, 255).astype(numpy.uint8).tobytes())
    return tuple(tables)


def apply_decode_array_8bit(
    samples: ImageBuffer,
    pairs: tuple[tuple[float, float], ...],
) -> numpy.ndarray:
    tables = decode_translation_tables_8bit(pairs)
    values = uint8_view(samples)
    if len(tables) == 1:
        return numpy.frombuffer(tables[0], dtype=numpy.uint8)[values]
    components = len(tables)
    result = numpy.empty(len(values), dtype=numpy.uint8)
    for index, table in enumerate(tables):
        result[index::components] = numpy.frombuffer(table, dtype=numpy.uint8)[
            values[index::components]
        ]
    return result


@lru_cache(maxsize=1024)
def decode_translation_tables_subbyte(
    max_sample: int,
    pairs: tuple[tuple[float, float], ...],
) -> tuple[bytes, ...]:
    values = numpy.arange(256, dtype=numpy.float64)
    normalized = values / max_sample if max_sample > 0 else numpy.zeros(256)
    tables: list[bytes] = []
    for dmin, dmax in pairs:
        span = dmax - dmin
        decoded = numpy.rint((dmin + normalized * span) * 255.0)
        tables.append(numpy.clip(decoded, 0, 255).astype(numpy.uint8).tobytes())
    return tuple(tables)


def apply_decode_array_subbyte(
    samples: ImageBuffer,
    pairs: tuple[tuple[float, float], ...],
    max_sample: int,
) -> numpy.ndarray:
    tables = decode_translation_tables_subbyte(max_sample, pairs)
    values = uint8_view(samples)
    if len(tables) == 1:
        return numpy.frombuffer(tables[0], dtype=numpy.uint8)[values]
    components = len(tables)
    result = numpy.empty(len(values), dtype=numpy.uint8)
    for index, table in enumerate(tables):
        result[index::components] = numpy.frombuffer(table, dtype=numpy.uint8)[
            values[index::components]
        ]
    return result


def image_dimension(image_dict: ImageDict | ImageColorSpec, key: str) -> int:
    if not isinstance(image_dict, dict):
        return 0
    value = lookup_dict_key(image_dict, key)
    if type(value) is bool:
        return 0
    return parse_int(value, 0) or 0


def image_component_count(spec: ImageColorSpec) -> int:
    if spec.kind in {"DeviceGray", "CalGray", "Indexed", "Separation"}:
        return 1
    if spec.kind in {"DeviceRGB", "Lab", "CalRGB"}:
        return 3
    if spec.kind == "DeviceCMYK":
        return 4
    if spec.kind in {"ICCBased", "DeviceN"}:
        return max(1, spec.channels)
    return 1


def evaluate_sampled_tint_function(function: PdfStream, *inputs: float) -> list[float]:
    dictionary = function.dictionary
    if parse_int(lookup_dict_key(dictionary, "FunctionType"), -1) != 0:
        raise ValueError("invalid sampled function")
    if parse_int(lookup_dict_key(dictionary, "BitsPerSample"), 0) != 8:
        raise ValueError("unsupported sampled function bit depth")
    size_obj = lookup_dict_key(dictionary, "Size")
    domain_obj = lookup_dict_key(dictionary, "Domain")
    range_obj = lookup_dict_key(dictionary, "Range")
    if not isinstance(size_obj, (list, tuple)):
        raise ValueError("invalid sampled function size")
    if not isinstance(domain_obj, (list, tuple)):
        raise ValueError("invalid sampled function domain")
    if not isinstance(range_obj, (list, tuple)) or len(range_obj) < 2:
        raise ValueError("invalid sampled function range")
    input_count = len(size_obj)
    if input_count != len(inputs):
        raise ValueError("invalid sampled function input count")
    if len(domain_obj) < input_count * 2:
        raise ValueError("invalid sampled function domain")
    output_count = len(range_obj) // 2
    sizes = [parse_int(value, 0) or 0 for value in size_obj]
    if any(size <= 0 for size in sizes):
        raise ValueError("invalid sampled function size")
    sample_count = 1
    for size in sizes:
        sample_count *= size
    samples = function.data
    if len(samples) < sample_count * output_count:
        raise ValueError("invalid sampled function data")
    encode_obj = lookup_dict_key(dictionary, "Encode")
    encoded_positions: list[int] = []
    for input_index, raw_input in enumerate(inputs):
        domain_min = parse_float(domain_obj[input_index * 2], None)
        domain_max = parse_float(domain_obj[input_index * 2 + 1], None)
        if domain_min is None or domain_max is None or domain_max == domain_min:
            raise ValueError("invalid sampled function domain")
        clipped = max(domain_min, min(domain_max, raw_input))
        normalized = (clipped - domain_min) / (domain_max - domain_min)
        encode_min = 0.0
        encode_max = float(sizes[input_index] - 1)
        if isinstance(encode_obj, (list, tuple)) and len(encode_obj) >= input_count * 2:
            parsed_min = parse_float(encode_obj[input_index * 2], None)
            parsed_max = parse_float(encode_obj[input_index * 2 + 1], None)
            if parsed_min is not None and parsed_max is not None:
                encode_min = parsed_min
                encode_max = parsed_max
        encoded = encode_min + normalized * (encode_max - encode_min)
        encoded_positions.append(max(0, min(sizes[input_index] - 1, int(round(encoded)))))
    sample_index = 0
    stride = 1
    for input_index, position in enumerate(encoded_positions):
        if input_index > 0:
            stride *= sizes[input_index - 1]
        sample_index += position * stride
    result: list[float] = []
    base = sample_index * output_count
    for output_index in range(output_count):
        range_min = parse_float(range_obj[output_index * 2], None)
        range_max = parse_float(range_obj[output_index * 2 + 1], None)
        if range_min is None or range_max is None:
            raise ValueError("invalid sampled function range")
        decoded = range_min + (samples[base + output_index] / 255.0) * (range_max - range_min)
        result.append(max(range_min, min(range_max, decoded)))
    return result


def unpack_subbyte_image_samples(
    data: ImageBuffer,
    bits_per_component: int,
    width: int,
    height: int,
    components: int,
) -> ImageBuffer:
    if bits_per_component not in {1, 2, 4}:
        return data
    if width <= 0 or height <= 0 or components <= 0:
        raise ValueError("invalid image dimensions")
    samples_per_row = width * components
    row_bytes = (samples_per_row * bits_per_component + 7) // 8
    if len(data) < row_bytes * height:
        raise ValueError("invalid image sample data")
    output = numpy.empty(width * height * components, dtype=numpy.uint8)
    output_row_bytes = samples_per_row
    chunk_rows = max(1, 4_000_000 // max(1, row_bytes))
    weights = 1 << numpy.arange(bits_per_component - 1, -1, -1, dtype=numpy.uint16)
    for row_start in range(0, height, chunk_rows):
        row_count = min(chunk_rows, height - row_start)
        packed = uint8_view(
            data,
            count=row_count * row_bytes,
            offset=row_start * row_bytes,
        ).reshape(row_count, row_bytes)
        unpacked = numpy.unpackbits(packed, axis=1, bitorder="big")
        unpacked = unpacked[:, : samples_per_row * bits_per_component].reshape(
            row_count,
            samples_per_row,
            bits_per_component,
        )
        samples = (
            (unpacked * weights)
            .sum(axis=2, dtype=numpy.uint16)
            .astype(
                numpy.uint8,
                copy=False,
            )
        )
        output_start = row_start * output_row_bytes
        output[output_start : output_start + row_count * output_row_bytes] = samples.reshape(-1)
    return output
