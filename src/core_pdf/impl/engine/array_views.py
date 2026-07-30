# SPDX-License-Identifier: AGPL-3.0-only
"""Shared zero-copy views over byte-oriented image buffers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeAlias

import numpy

ByteBuffer: TypeAlias = bytes | bytearray | memoryview | numpy.ndarray[Any, Any]
UInt8Array = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


@lru_cache(maxsize=256)
def nearest_indices(output_count: int, source_count: int) -> numpy.ndarray[Any, Any]:
    """Return bounded nearest-neighbour source indexes for an output axis."""
    if output_count <= 0 or source_count <= 0:
        return numpy.empty(0, dtype=numpy.intp)
    indexes = numpy.arange(output_count, dtype=numpy.intp)
    result = numpy.minimum(source_count - 1, (indexes * source_count) // output_count)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=256)
def unit_sample_positions(output_count: int) -> numpy.ndarray[Any, Any]:
    """Return normalized pixel positions used for sampled-mask interpolation."""
    if output_count <= 0:
        return numpy.empty(0, dtype=numpy.float64)
    result = numpy.arange(output_count, dtype=numpy.float64) / output_count
    result.flags.writeable = False
    return result


def resample_nearest(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    """Resize a 2D or 3D sample array with bounded nearest-neighbour lookup."""
    if height <= 0 or width <= 0:
        raise ValueError("resampling dimensions must be positive")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must be a 2D or 3D array")
    if samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("samples must have positive spatial dimensions")
    if samples.shape[:2] == (height, width) and samples.flags.c_contiguous:
        return samples
    y_indexes = nearest_indices(height, samples.shape[0])
    x_indexes = nearest_indices(width, samples.shape[1])
    return numpy.ascontiguousarray(samples[y_indexes[:, None], x_indexes])


def contiguous_bytes(array: numpy.ndarray[Any, Any]) -> memoryview:
    """Expose a C-contiguous array as bytes, copying only at the serialization boundary."""
    return memoryview(numpy.ascontiguousarray(array)).cast("B")


def typed_view(
    buffer: ByteBuffer,
    dtype: numpy.dtype[Any] | str,
    *,
    count: int = -1,
    offset: int = 0,
) -> numpy.ndarray[Any, Any]:
    """Return a one-dimensional typed view over a byte-oriented buffer.

    Byte buffers remain borrowed.  Array inputs are flattened when compatible and
    copied only when their dtype or layout cannot support the requested view.
    ``count`` and ``offset`` are expressed in elements for array inputs and bytes
    for byte buffers, matching NumPy's ``frombuffer`` contract.
    """
    target = numpy.dtype(dtype)
    if isinstance(buffer, numpy.ndarray):
        array = numpy.asarray(buffer)
        if array.dtype == target and array.flags.c_contiguous:
            view = array.reshape(-1)
        else:
            view = numpy.ascontiguousarray(array, dtype=target).reshape(-1)
        if offset:
            view = view[offset:]
        if count >= 0:
            view = view[:count]
        return view
    return numpy.frombuffer(buffer, dtype=target, count=count, offset=offset)


def uint8_view(
    buffer: ByteBuffer,
    *,
    count: int = -1,
    offset: int = 0,
) -> UInt8Array:
    """Return a flat uint8 view, copying only non-contiguous/wrong-typed arrays."""
    if isinstance(buffer, numpy.ndarray):
        array = numpy.asarray(buffer)
        if array.dtype != numpy.dtype(numpy.uint8):
            view = numpy.asarray(array, dtype=numpy.uint8).reshape(-1)
        elif not array.flags.c_contiguous:
            view = numpy.ascontiguousarray(array).reshape(-1)
        else:
            view = array.reshape(-1)
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
    if len(view) < expected:
        raise ValueError("buffer is smaller than requested image shape")
    if not allow_trailing and len(view) != expected:
        raise ValueError("buffer is larger than requested image shape")
    return view[:expected].reshape(shape)


__all__ = (
    "ByteBuffer",
    "UInt8Array",
    "contiguous_bytes",
    "nearest_indices",
    "resample_nearest",
    "typed_view",
    "uint8_image_view",
    "uint8_view",
    "unit_sample_positions",
)
