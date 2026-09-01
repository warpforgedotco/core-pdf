# SPDX-License-Identifier: AGPL-3.0-only
"""Shared zero-copy views over byte-oriented image buffers."""

from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=256)
def nearest_indices(output_count: int, source_count: int) -> numpy.ndarray[Any, Any]:
    """Return bounded nearest-neighbour source indexes for an output axis."""
    if output_count <= 0 or source_count <= 0:
        return numpy.empty(0, dtype=numpy.intp)
    indexes = numpy.arange(output_count, dtype=numpy.intp)
    result = numpy.minimum(source_count - 1, (indexes * source_count) // output_count)
    return readonly(result)


@lru_cache(maxsize=256)
def unit_sample_positions(output_count: int) -> numpy.ndarray[Any, Any]:
    """Return normalized pixel positions used for sampled-mask interpolation."""
    if output_count <= 0:
        return numpy.empty(0, dtype=numpy.float64)
    result = numpy.arange(output_count, dtype=numpy.float64) / output_count
    return readonly(result)


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
    rows = samples.take(y_indexes, axis=0)
    return numpy.ascontiguousarray(rows.take(x_indexes, axis=1))


@lru_cache(maxsize=256)
def internal_box_bounds(output_count: int, source_count: int) -> tuple[Any, Any]:
    """Return per-output bin start offsets and sample counts for one axis."""
    edges = numpy.arange(output_count + 1, dtype=numpy.intp) * source_count // output_count
    # A bin can collapse to zero samples when the ratio is close to 1. Widen it by
    # one sample so ``reduceat`` never reads a reversed slice.
    starts = numpy.minimum(edges[:-1], source_count - 1)
    stops = numpy.maximum(edges[1:], starts + 1)
    counts = (stops - starts).astype(numpy.float32)
    return readonly(starts), readonly(counts)


def internal_box_axis(
    samples: numpy.ndarray[Any, Any], output_count: int, axis: int
) -> numpy.ndarray[Any, Any]:
    """Average one axis into ``output_count`` bins without materializing a cast copy."""
    starts, counts = internal_box_bounds(output_count, samples.shape[axis])
    totals = numpy.add.reduceat(samples, starts, axis=axis, dtype=numpy.float32)
    shape = [1] * totals.ndim
    shape[axis] = output_count
    return totals / counts.reshape(shape)


def resample_box(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    """Resize by averaging each output pixel's source area.

    Point sampling a 300 DPI scan down to a pixel budget drops whole stroke rows,
    which is how ``lbs`` becomes ``ibs``. Averaging the covered area keeps the
    stroke energy that Tesseract's classifier needs. Only used for reductions;
    callers upscale with :func:`resample_bilinear`.
    """
    if height <= 0 or width <= 0:
        raise ValueError("resampling dimensions must be positive")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must be a 2D or 3D array")
    if samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("samples must have positive spatial dimensions")
    if height > samples.shape[0] or width > samples.shape[1]:
        raise ValueError("resample_box only reduces; use resample_bilinear to enlarge")
    if samples.shape[:2] == (height, width) and samples.flags.c_contiguous:
        return samples
    reduced = samples
    # Reduce the longer axis first so the second pass runs over fewer elements.
    if samples.shape[0] - height >= samples.shape[1] - width:
        if height != samples.shape[0]:
            reduced = internal_box_axis(reduced, height, 0)
        if width != samples.shape[1]:
            reduced = internal_box_axis(reduced, width, 1)
    else:
        if width != samples.shape[1]:
            reduced = internal_box_axis(reduced, width, 1)
        if height != samples.shape[0]:
            reduced = internal_box_axis(reduced, height, 0)
    if reduced is samples:
        return samples
    return numpy.ascontiguousarray(numpy.rint(reduced)).astype(samples.dtype, copy=False)


@lru_cache(maxsize=256)
def internal_bilinear_taps(output_count: int, source_count: int) -> tuple[Any, Any, Any]:
    """Return lower/upper source indexes and blend weights for one axis."""
    if source_count == 1:
        zeros = readonly(numpy.zeros(output_count, dtype=numpy.intp))
        weights = readonly(numpy.zeros(output_count, dtype=numpy.float32))
        return zeros, zeros, weights
    positions = (numpy.arange(output_count, dtype=numpy.float64) + 0.5) * (
        source_count / output_count
    ) - 0.5
    positions = numpy.clip(positions, 0.0, source_count - 1)
    lower = numpy.floor(positions).astype(numpy.intp)
    lower = numpy.minimum(lower, source_count - 2)
    upper = lower + 1
    weights = (positions - lower).astype(numpy.float32)
    return readonly(lower), readonly(upper), readonly(weights)


def internal_bilinear_axis(
    samples: numpy.ndarray[Any, Any], output_count: int, axis: int
) -> numpy.ndarray[Any, Any]:
    """Linearly interpolate one axis to ``output_count`` samples."""
    lower, upper, weights = internal_bilinear_taps(output_count, samples.shape[axis])
    shape = [1] * samples.ndim
    shape[axis] = output_count
    blend = weights.reshape(shape)
    low = samples.take(lower, axis=axis).astype(numpy.float32, copy=False)
    high = samples.take(upper, axis=axis).astype(numpy.float32, copy=False)
    return low + (high - low) * blend


def resample_bilinear(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    """Resize with separable linear interpolation.

    Replicating pixels to enlarge a scan gives Tesseract staircased stems; blending
    neighbours preserves the smooth edges its line classifier was trained on.
    """
    if height <= 0 or width <= 0:
        raise ValueError("resampling dimensions must be positive")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must be a 2D or 3D array")
    if samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("samples must have positive spatial dimensions")
    if samples.shape[:2] == (height, width) and samples.flags.c_contiguous:
        return samples
    resized = samples
    # Interpolate the shrinking axis first to keep the intermediate small.
    if height - samples.shape[0] <= width - samples.shape[1]:
        if height != samples.shape[0]:
            resized = internal_bilinear_axis(resized, height, 0)
        if width != samples.shape[1]:
            resized = internal_bilinear_axis(resized, width, 1)
    else:
        if width != samples.shape[1]:
            resized = internal_bilinear_axis(resized, width, 1)
        if height != samples.shape[0]:
            resized = internal_bilinear_axis(resized, height, 0)
    if resized is samples:
        return samples
    return numpy.ascontiguousarray(numpy.rint(resized)).astype(samples.dtype, copy=False)


def resample_smooth(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    """Resize using the filter that suits the direction of each axis change."""
    if height <= samples.shape[0] and width <= samples.shape[1]:
        return resample_box(samples, height, width)
    if height >= samples.shape[0] and width >= samples.shape[1]:
        return resample_bilinear(samples, height, width)
    intermediate_height = min(height, samples.shape[0])
    intermediate_width = min(width, samples.shape[1])
    reduced = resample_box(samples, intermediate_height, intermediate_width)
    return resample_bilinear(reduced, height, width)


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
    "resample_bilinear",
    "resample_box",
    "resample_nearest",
    "resample_smooth",
    "uint8_image_view",
    "uint8_view",
    "unit_sample_positions",
)
