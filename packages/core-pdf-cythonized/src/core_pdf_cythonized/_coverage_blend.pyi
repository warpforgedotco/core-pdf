# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

Plane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]

def blend_coverage_counts(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    counts: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    left: int,
    top: int,
    allowed: bytes | bytearray | memoryview | None,
    red: int,
    green: int,
    blue: int,
    alpha: int,
    source_alpha: Plane | None,
    source_shape: Plane | None,
    track_shape: bool,
    shape_alpha: float,
    mode: int = ...,
    revised: bool = ...,
    stop_at_visible: bool = ...,
) -> tuple[tuple[int, int, int, int] | None, bool]: ...
