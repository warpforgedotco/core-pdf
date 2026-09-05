# SPDX-License-Identifier: AGPL-3.0-only
"""Shared zero-copy views over byte-oriented image buffers."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy

ByteBuffer: TypeAlias = bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
UInt8Array = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


def readonly(array: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    """Mark an array immutable and return it, so callers can seal in an expression."""
    array.flags.writeable = False
    return array


def finite_median(values: numpy.ndarray[Any, Any]) -> float:
    """Return the median of a non-empty finite 1D array without NaN dispatch."""
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
    """Return bounded nearest-neighbour source indexes for an output axis."""
    if output_count <= 0 or source_count <= 0:
        return numpy.empty(0, dtype=numpy.intp)
    indexes = numpy.arange(output_count, dtype=numpy.intp)
    result = numpy.minimum(source_count - 1, (indexes * source_count) // output_count)
    return readonly(result)


def unit_sample_positions(output_count: int) -> numpy.ndarray[Any, Any]:
    """Return normalized pixel positions used for sampled-mask interpolation."""
    if output_count <= 0:
        return numpy.empty(0, dtype=numpy.float64)
    result = numpy.arange(output_count, dtype=numpy.float64) / output_count
    return readonly(result)


def contiguous_bytes(array: numpy.ndarray[Any, Any]) -> memoryview:
    """Expose a C-contiguous array as bytes, copying only at the serialization boundary."""
    return memoryview(numpy.ascontiguousarray(array)).cast("B")


def uint8_view(
    buffer: ByteBuffer,
    *,
    count: int = -1,
    offset: int = 0,
) -> UInt8Array:
    """Return a flat uint8 view, copying only non-contiguous/wrong-typed arrays."""
    if isinstance(buffer, numpy.ndarray):
        array = numpy.asarray(buffer)
        if array.dtype.type is numpy.uint8:
            if array.flags.c_contiguous:
                view = array.reshape(-1)
            else:
                view = numpy.ascontiguousarray(array).reshape(-1)
        else:
            view = numpy.asarray(array, dtype=numpy.uint8).reshape(-1)
        if offset:
            view = view[offset:]
        if count >= 0:
            view = view[:count]
        return view
    return numpy.frombuffer(buffer, dtype=numpy.uint8, count=count, offset=offset)


def uint8_image_view(
    buffer: ByteBuffer,
    shape: tuple[int, ...],
    *,
    allow_trailing: bool = False,
) -> UInt8Array:
    """Return a shaped uint8 view, validating the required byte count."""
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
    "unit_sample_positions",
)
