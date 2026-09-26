# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def stroke_polylines(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    xs: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    ys: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    spans: list[tuple[int, int, bool]],
    outline: bool,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    clip: tuple[float, float, float, float] | None,
    line_width: float,
    red: int,
    green: int,
    blue: int,
    alpha: int,
    line_cap: int,
    line_join: int,
    first: int,
    exponent: float,
) -> tuple[tuple[int, int, int, int] | None, int, int]: ...
