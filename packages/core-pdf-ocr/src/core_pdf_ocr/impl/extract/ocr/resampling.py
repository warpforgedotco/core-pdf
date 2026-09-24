# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.array_views import nearest_indices, readonly


def validate_resampling_shape(samples: numpy.ndarray[Any, Any], height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("resampling dimensions must be positive")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must be a 2D or 3D array")
    if samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("samples must have positive spatial dimensions")


def resample_nearest(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    validate_resampling_shape(samples, height, width)
    if samples.shape[:2] == (height, width) and samples.flags.c_contiguous:
        return samples
    y_indexes = nearest_indices(height, samples.shape[0])
    x_indexes = nearest_indices(width, samples.shape[1])
    rows = samples.take(y_indexes, axis=0)
    return numpy.ascontiguousarray(rows.take(x_indexes, axis=1))


def box_bounds(output_count: int, source_count: int) -> tuple[Any, Any]:
    edges = numpy.arange(output_count + 1, dtype=numpy.intp) * source_count // output_count
    starts = numpy.minimum(edges[:-1], source_count - 1)
    stops = numpy.maximum(edges[1:], starts + 1)
    counts = (stops - starts).astype(numpy.float32)
    return readonly(starts), readonly(counts)


def box_axis(
    samples: numpy.ndarray[Any, Any], output_count: int, axis: int
) -> numpy.ndarray[Any, Any]:
    starts, counts = box_bounds(output_count, samples.shape[axis])
    totals = numpy.add.reduceat(samples, starts, axis=axis, dtype=numpy.float32)
    shape = [1] * totals.ndim
    shape[axis] = output_count
    return totals / counts.reshape(shape)


def resample_separable(
    samples: numpy.ndarray[Any, Any],
    height: int,
    width: int,
    resample_axis: Callable[[numpy.ndarray[Any, Any], int, int], numpy.ndarray[Any, Any]],
) -> numpy.ndarray[Any, Any]:
    axes = (0, 1) if height - samples.shape[0] <= width - samples.shape[1] else (1, 0)
    dimensions = (height, width)
    resized = samples
    for axis in axes:
        if dimensions[axis] != samples.shape[axis]:
            resized = resample_axis(resized, dimensions[axis], axis)
    if resized is samples:
        return samples
    return numpy.ascontiguousarray(numpy.rint(resized)).astype(samples.dtype, copy=False)


def resample_box(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    validate_resampling_shape(samples, height, width)
    if height > samples.shape[0] or width > samples.shape[1]:
        raise ValueError("resample_box only reduces; use resample_bilinear to enlarge")
    return resample_separable(samples, height, width, box_axis)


def bilinear_taps(output_count: int, source_count: int) -> tuple[Any, Any, Any]:
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


def bilinear_axis(
    samples: numpy.ndarray[Any, Any], output_count: int, axis: int
) -> numpy.ndarray[Any, Any]:
    lower, upper, weights = bilinear_taps(output_count, samples.shape[axis])
    shape = [1] * samples.ndim
    shape[axis] = output_count
    blend = weights.reshape(shape)
    low = samples.take(lower, axis=axis).astype(numpy.float32, copy=False)
    high = samples.take(upper, axis=axis).astype(numpy.float32, copy=False)
    return low + (high - low) * blend


def resample_bilinear(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    validate_resampling_shape(samples, height, width)
    return resample_separable(samples, height, width, bilinear_axis)


def resample_smooth(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    validate_resampling_shape(samples, height, width)
    if height <= samples.shape[0] and width <= samples.shape[1]:
        return resample_box(samples, height, width)
    if height >= samples.shape[0] and width >= samples.shape[1]:
        return resample_bilinear(samples, height, width)
    intermediate_height = min(height, samples.shape[0])
    intermediate_width = min(width, samples.shape[1])
    reduced = resample_box(samples, intermediate_height, intermediate_width)
    return resample_bilinear(reduced, height, width)
