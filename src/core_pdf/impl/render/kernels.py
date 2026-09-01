# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-cutting raster coordinate kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy

RASTER_COORDINATE_CACHE_MAX_ENTRIES = 256


def internal_cached_raster_coordinates(
    cache: dict[tuple[int, int], numpy.ndarray[Any, Any]],
    start: int,
    stop: int,
) -> numpy.ndarray[Any, Any]:
    key = (start, stop)
    coordinates = cache.get(key)
    if coordinates is None:
        coordinates = numpy.arange(start, stop, dtype=numpy.float64)
        if len(cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
            cache[key] = coordinates
    return coordinates


def internal_make_page_geometry(
    crop_x0: float, crop_y1: float, scale: float, width: int, height: int
) -> tuple[
    Callable[[float, float, float, float], tuple[int, int, int, int] | None],
    Callable[[float, float], tuple[int, int] | None],
]:
    """Build the page-to-pixel converters, closed over a fixed page geometry.

    These run ~1.8M times over the corpus. Binding the geometry into a closure
    once keeps every read a `LOAD_DEREF`; holding them as instance attributes
    would add an attribute load per access on the hottest path in the rasterizer.
    """

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
