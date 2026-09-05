# SPDX-License-Identifier: AGPL-3.0-only
"""Raster resampling for recognition inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.runtime.array_views import nearest_indices, readonly


def internal_validate_resampling_shape(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("resampling dimensions must be positive")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must be a 2D or 3D array")
    if samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("samples must have positive spatial dimensions")


def resample_nearest(
    samples: numpy.ndarray[Any, Any], height: int, width: int
) -> numpy.ndarray[Any, Any]:
    """Resize a 2D or 3D sample array with bounded nearest-neighbour lookup."""
    internal_validate_resampling_shape(samples, height, width)
    if samples.shape[:2] == (height, width) and samples.flags.c_contiguous:
        return samples
    y_indexes = nearest_indices(height, samples.shape[0])
    x_indexes = nearest_indices(width, samples.shape[1])
    rows = samples.take(y_indexes, axis=0)
    return numpy.ascontiguousarray(rows.take(x_indexes, axis=1))


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


def internal_resample_separable(
    samples: numpy.ndarray[Any, Any],
    height: int,
    width: int,
    resample_axis: Callable[[numpy.ndarray[Any, Any], int, int], numpy.ndarray[Any, Any]],
) -> numpy.ndarray[Any, Any]:
    """Apply two axis passes, keeping float intermediates until the final rounding."""
    # Shrink the most (or enlarge the least) first to keep the intermediate small.
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
    """Resize by averaging each output pixel's source area.

    Point sampling a 300 DPI scan down to a pixel budget drops whole stroke rows,
    which is how ``lbs`` becomes ``ibs``. Averaging the covered area keeps the
    stroke energy that Tesseract's classifier needs. Only used for reductions;
    callers upscale with :func:`resample_bilinear`.
    """
    internal_validate_resampling_shape(samples, height, width)
    if height > samples.shape[0] or width > samples.shape[1]:
        raise ValueError("resample_box only reduces; use resample_bilinear to enlarge")
    return internal_resample_separable(samples, height, width, internal_box_axis)


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
    internal_validate_resampling_shape(samples, height, width)
    return internal_resample_separable(samples, height, width, internal_bilinear_axis)


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
