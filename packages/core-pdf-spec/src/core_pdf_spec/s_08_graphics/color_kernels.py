# SPDX-License-Identifier: AGPL-3.0-only
"""PDF sample packing and Decode mapping, before device quantization."""

from __future__ import annotations

from typing import Any

import numpy


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
    bits = numpy.unpackbits(
        packed[: row_bytes * height].reshape(height, row_bytes), axis=1, bitorder="big"
    )[:, : row_samples * bits_per_component]
    groups = bits.reshape(height, row_samples, bits_per_component)
    shifts = numpy.arange(bits_per_component - 1, -1, -1, dtype=numpy.uint8)
    return numpy.sum(groups << shifts, axis=2, dtype=numpy.uint8).reshape(-1)


def decode_sample_values(
    samples: numpy.ndarray[Any, Any], pairs: tuple[tuple[float, float], ...], max_sample: int
) -> numpy.ndarray[Any, Any]:
    if max_sample <= 0 or not pairs:
        raise ValueError("invalid image Decode mapping")
    values = numpy.asarray(samples, dtype=numpy.float64).reshape(-1, len(pairs)) / max_sample
    minimum = numpy.asarray([pair[0] for pair in pairs])
    maximum = numpy.asarray([pair[1] for pair in pairs])
    return minimum + values * (maximum - minimum)


__all__ = (
    "unpack_subbyte_image_samples",
    "decode_sample_values",
)
