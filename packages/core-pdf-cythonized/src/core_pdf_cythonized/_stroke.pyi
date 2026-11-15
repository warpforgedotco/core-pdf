# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable
from typing import Any

import numpy

Int64Array = numpy.ndarray[Any, numpy.dtype[numpy.int64]]

def stroke_polylines(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    xs: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    ys: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    spans: list[tuple[int, int, bool]],
    outline: bool,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    clip_mode: int,
    clip_box: tuple[float, float, float, float] | None,
    clip_empty: bool,
    rows_origin: int,
    row_offsets: Int64Array | None,
    row_spans: Int64Array | None,
    line_width: float,
    red: int,
    green: int,
    blue: int,
    alpha: int,
    native: bool,
    line_cap: int,
    line_join: int,
    miter_limit: float,
    exponent: float,
    on_line: Callable[[float, float, float, float], None],
    on_polygon: Callable[[tuple[float, ...]], None],
    on_square: Callable[[float, float], None],
    on_circle: Callable[[float, float], None],
    on_dot: Callable[[float, float], None],
    on_extend: Callable[[int, int, int, int], None],
) -> None: ...
