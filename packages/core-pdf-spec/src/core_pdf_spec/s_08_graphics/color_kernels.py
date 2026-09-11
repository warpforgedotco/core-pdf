# SPDX-License-Identifier: AGPL-3.0-only
"""PDF sample packing and Decode mapping, before device quantization."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf_spec.s_07_filters.predictors import internal_unpack_subbyte_rows


def unpack_subbyte_image_samples(
    data: bytes | memoryview | numpy.ndarray[Any, Any],
    bits_per_component: int,
    width: int,
    height: int,
    components: int,
) -> numpy.ndarray[Any, Any]:
    if bits_per_component not in {1, 2, 4} or min(width, height, components) <= 0:
        raise ValueError("invalid image sample layout")
    row_samples = width * components
    row_bytes = (row_samples * bits_per_component + 7) // 8
    packed = (
        numpy.asarray(data, dtype=numpy.uint8).reshape(-1)
        if isinstance(data, numpy.ndarray)
        else numpy.frombuffer(data, dtype=numpy.uint8)
    )
    if len(packed) < row_bytes * height:
        raise ValueError("invalid image sample data")
    return internal_unpack_subbyte_rows(
        packed[: row_bytes * height].reshape(height, row_bytes), row_samples, bits_per_component
    ).reshape(-1)


def decode_sample_values(
    samples: numpy.ndarray[Any, Any], pairs: tuple[tuple[float, float], ...], max_sample: int
) -> numpy.ndarray[Any, Any]:
    if max_sample <= 0 or not pairs:
        raise ValueError("invalid image Decode mapping")
    values = numpy.asarray(samples, dtype=numpy.float64).reshape(-1, len(pairs)) / max_sample
    minimum = numpy.asarray([pair[0] for pair in pairs])
    maximum = numpy.asarray([pair[1] for pair in pairs])
    return minimum + values * (maximum - minimum)


def unpack_image_samples(
    data: bytes | memoryview | numpy.ndarray[Any, Any],
    bits_per_component: int,
    width: int,
    height: int,
    components: int,
) -> numpy.ndarray[Any, Any]:
    """ISO 32000-1, 8.9.3: unpack unsigned samples, with 16-bit words MSB first."""
    if bits_per_component in {1, 2, 4}:
        return unpack_subbyte_image_samples(data, bits_per_component, width, height, components)
    if bits_per_component not in {8, 16} or min(width, height, components) <= 0:
        raise ValueError("invalid image sample layout")
    packed = (
        numpy.asarray(data, dtype=numpy.uint8).reshape(-1)
        if isinstance(data, numpy.ndarray)
        else numpy.frombuffer(data, dtype=numpy.uint8)
    )
    count = width * height * components
    length = count * (bits_per_component // 8)
    if len(packed) < length:
        raise ValueError("invalid image sample data")
    values = numpy.ascontiguousarray(packed[:length])
    return values if bits_per_component == 8 else values.view(">u2").astype(numpy.uint16)


def color_key_alpha(
    samples: numpy.ndarray[Any, Any], mask: tuple[int, ...], max_sample: int
) -> numpy.ndarray[Any, Any]:
    """ISO 32000-1/2 8.9.6.4: mask inclusive source integers before Decode."""
    integers = numpy.asarray(samples)
    if (
        integers.ndim != 2
        or not numpy.issubdtype(integers.dtype, numpy.integer)
        or max_sample <= 0
        or integers.shape[1] == 0
        or len(mask) != integers.shape[1] * 2
        or any(type(value) is not int or value < 0 or value > max_sample for value in mask)
        or numpy.any(integers < 0)
        or numpy.any(integers > max_sample)
    ):
        raise ValueError("invalid image color key mask")
    ranges = numpy.asarray(mask, dtype=numpy.int64).reshape(-1, 2)
    if numpy.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError("invalid image color key mask")
    hidden = ((integers >= ranges[:, 0]) & (integers <= ranges[:, 1])).all(axis=1)
    return numpy.where(hidden, 0, 255).astype(numpy.uint8)


__all__ = (
    "color_key_alpha",
    "unpack_subbyte_image_samples",
    "decode_sample_values",
    "unpack_image_samples",
)
