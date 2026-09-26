# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def stroke_segment_samples(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    origin_x: int,
    origin_y: int,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    seg_len2: float,
    half2: float,
    projection_extension: float,
    red: int,
    green: int,
    blue: int,
    alpha: int,
    allowed: bytes | bytearray | memoryview | None = ...,
    counts: numpy.ndarray[Any, numpy.dtype[numpy.uint8]] | None = ...,
) -> tuple[int, int, int, int] | None: ...
