# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, TypeAlias

import numpy

from core_jbig2.bitmap import uint8_view as uint8_view

ByteBuffer: TypeAlias = bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
UInt8Array = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


def readonly(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    array.flags.writeable = False
    return array


def finite_median(values: numpy.ndarray[Any, Any]) -> float:
    size = values.size
    if size == 0:
        raise ValueError("finite_median requires at least one value")
    middle = size // 2
    if size & 1:
        partitioned = numpy.partition(values, middle)
        return float(partitioned[middle])
    partitioned = numpy.partition(values, (middle - 1, middle))
    return (float(partitioned[middle - 1]) + float(partitioned[middle])) * 0.5


def nearest_indices(output_count: int, source_count: int) -> numpy.ndarray[Any, Any]:
    if output_count <= 0 or source_count <= 0:
        return numpy.empty(0, dtype=numpy.intp)
    indexes = numpy.arange(output_count, dtype=numpy.intp)
    result = numpy.minimum(source_count - 1, (indexes * source_count) // output_count)
    return readonly(result)


def contiguous_bytes(array: numpy.ndarray[Any, Any]) -> memoryview:
    return memoryview(numpy.ascontiguousarray(array)).cast("B")


def uint8_image_view(
    buffer: ByteBuffer,
    shape: tuple[int, ...],
    *,
    allow_trailing: bool = False,
) -> UInt8Array:
    view = uint8_view(buffer)
    expected = 1
    for dimension in shape:
        expected *= dimension
    view_len = len(view)
    if view_len < expected:
        raise ValueError("buffer is smaller than requested image shape")
    if not allow_trailing and view_len != expected:
        raise ValueError("buffer is larger than requested image shape")
    if view_len != expected:
        view = view[:expected]
    return view.reshape(shape)


__all__ = (
    "ByteBuffer",
    "UInt8Array",
    "contiguous_bytes",
    "finite_median",
    "nearest_indices",
    "readonly",
    "uint8_image_view",
    "uint8_view",
)
