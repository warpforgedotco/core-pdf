# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import numpy

from core_jbig2.bitmap import uint8_view

ByteBuffer: TypeAlias = bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
UInt8Array = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
BoolArray = numpy.ndarray[Any, numpy.dtype[numpy.bool_]]


def readonly(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    array.flags.writeable = False
    return array


def finite_median(values: numpy.ndarray[Any, Any]) -> float:
    size = values.size
    if size == 0:
        raise ValueError("finite_median requires at least one value")
    middle = size // 2
    # numpy.partition on a flattened copy, without its Python-level dispatch.
    partitioned = values.flatten()
    if size & 1:
        partitioned.partition(middle)
        return float(partitioned[middle])
    partitioned.partition((middle - 1, middle))
    return (float(partitioned[middle - 1]) + float(partitioned[middle])) * 0.5


def make_column(
    values: Iterable[Any] | None,
    dtype: Any,
    default: Callable[[], numpy.ndarray[Any, Any]] | None = None,
) -> numpy.ndarray[Any, Any]:
    if values is None:
        if default is None:
            raise ValueError("observation column is required")
        return default()
    return numpy.array(
        values if isinstance(values, (list, tuple, range, numpy.ndarray)) else tuple(values),
        dtype=dtype,
    )


def validate_selection_mask(mask: BoolArray, size: int) -> None:
    if mask.dtype != numpy.bool_:
        raise TypeError("observation selection mask must have boolean dtype")
    if mask.shape != (size,):
        raise ValueError("observation selection mask must have shape (n,)")


def nearest_indices(output_count: int, source_count: int) -> numpy.ndarray[Any, Any]:
    if output_count <= 0 or source_count <= 0:
        return numpy.empty(0, dtype=numpy.intp)
    indexes = numpy.arange(output_count, dtype=numpy.intp)
    result = numpy.minimum(source_count - 1, (indexes * source_count) // output_count)
    return readonly(result)


def contiguous_bytes(array: numpy.ndarray[Any, Any]) -> memoryview:
    return memoryview(numpy.ascontiguousarray(array)).cast("B")


def uint8_image_view(buffer: ByteBuffer, shape: tuple[int, ...]) -> UInt8Array:
    view = uint8_view(buffer)
    expected = 1
    for dimension in shape:
        expected *= dimension
    view_len = len(view)
    if view_len < expected:
        raise ValueError("buffer is smaller than requested image shape")
    if view_len != expected:
        raise ValueError("buffer is larger than requested image shape")
    return view.reshape(shape)


__all__ = (
    "BoolArray",
    "ByteBuffer",
    "UInt8Array",
    "contiguous_bytes",
    "finite_median",
    "make_column",
    "nearest_indices",
    "readonly",
    "uint8_image_view",
    "uint8_view",
    "validate_selection_mask",
)
