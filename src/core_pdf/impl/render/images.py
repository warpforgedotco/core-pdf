# SPDX-License-Identifier: AGPL-3.0-only
"""Image resampling and mask kernels."""

from __future__ import annotations

from typing import Any

import numpy

AFFINE_BLIT_SCRATCH_BYTES = 1 << 20


def internal_box_downsample(
    samples: numpy.ndarray[Any, Any],
    source_width: int,
    source_height: int,
    channels: int,
    target_width: int,
    target_height: int,
) -> tuple[numpy.ndarray[Any, Any], int, int]:
    """Area-average an image down to about (target_width, target_height).

    Sampling a shrunk image with nearest-neighbour throws away most of it: a
    2544x3296 scan placed on a 612x792 page keeps roughly one source pixel in
    eighteen, so the thin rules and letter stems of a scanned form fall between
    samples and the page renders visibly faint. Averaging the block each output
    pixel covers keeps that ink.

    Bin edges are ``i * source // target`` so the blocks tile the source exactly
    even when the ratio is not integral; ``add.reduceat`` then sums each block in
    one pass per axis. Returns the reduced samples with their new dimensions, so
    the caller's existing nearest-neighbour map resamples an already-averaged
    image.
    """
    if target_width <= 0 or target_height <= 0:
        return samples, source_width, source_height
    if source_width <= target_width and source_height <= target_height:
        return samples, source_width, source_height
    target_width = min(target_width, source_width)
    target_height = min(target_height, source_height)
    grid = samples.reshape(source_height, source_width, channels)
    row_edges = (numpy.arange(target_height + 1, dtype=numpy.int64) * source_height) // (
        target_height
    )
    column_edges = (numpy.arange(target_width + 1, dtype=numpy.int64) * source_width) // (
        target_width
    )
    totals = numpy.add.reduceat(grid, row_edges[:-1], axis=0, dtype=numpy.uint32)
    totals = numpy.add.reduceat(totals, column_edges[:-1], axis=1, dtype=numpy.uint32)
    counts = numpy.diff(row_edges)[:, None, None] * numpy.diff(column_edges)[None, :, None]
    reduced = (totals // numpy.maximum(counts, 1)).astype(numpy.uint8)
    return reduced.reshape(-1), target_width, target_height


def internal_soft_mask_alpha_at(
    mask: numpy.ndarray[Any, Any] | None,
    u: float,
    v: float,
) -> int:
    if mask is None:
        return 255
    if mask.ndim != 2 or mask.size == 0:
        return 255
    mask_height, mask_width = mask.shape
    src_x = min(mask_width - 1, max(0, int(u * mask_width)))
    src_y = min(mask_height - 1, max(0, int((1.0 - v) * mask_height)))
    return int(mask[src_y, src_x])
