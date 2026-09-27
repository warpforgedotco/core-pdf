# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def blend_visible_rgba(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    visible: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    red: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    green: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    blue: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    alpha: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    mode: int,
) -> None: ...
