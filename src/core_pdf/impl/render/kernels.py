# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.runtime.array_views import UInt8Array

AFFINE_BLIT_SCRATCH_BYTES = 1 << 20


def internal_box_downsample(
    samples: numpy.ndarray[Any, Any],
    source_width: int,
    source_height: int,
    channels: int,
    target_width: int,
    target_height: int,
) -> tuple[numpy.ndarray[Any, Any], int, int]:
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


def internal_sample_image_plane(
    plane: UInt8Array, u: numpy.ndarray[Any, Any], v: numpy.ndarray[Any, Any]
) -> UInt8Array:
    height, width = plane.shape
    source_x = numpy.clip((u * width).astype(numpy.intp), 0, width - 1)
    source_y = numpy.clip(((1.0 - v) * height).astype(numpy.intp), 0, height - 1)
    return plane[source_y, source_x]


def internal_make_page_geometry(
    crop_x0: float, crop_y1: float, scale: float, width: int, height: int
) -> tuple[
    Callable[[float, float, float, float], tuple[int, int, int, int] | None],
    Callable[[float, float], tuple[int, int] | None],
]:

    def page_box_to_pixels(
        x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, int, int, int] | None:
        ix0 = max(0, min(width, math.floor((x0 - crop_x0) * scale)))
        ix1 = max(0, min(width, math.ceil((x1 - crop_x0) * scale)))
        iy0 = max(0, min(height, math.floor((crop_y1 - y1) * scale)))
        iy1 = max(0, min(height, math.ceil((crop_y1 - y0) * scale)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return ix0, iy0, ix1, iy1

    def page_x_to_pixel_span(start_x: float, end_x: float) -> tuple[int, int] | None:
        if end_x <= start_x:
            return None
        start = math.ceil((start_x - crop_x0) * scale - 0.5)
        end = math.ceil((end_x - crop_x0) * scale - 0.5)
        start = max(0, min(width, start))
        end = max(0, min(width, end))
        if end <= start:
            return None
        return start, end

    return page_box_to_pixels, page_x_to_pixel_span
