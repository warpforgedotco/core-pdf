# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

SPACE_RGB: int
SPACE_GRAY: int
SPACE_CMYK: int

def luminosity_channel(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    black: numpy.ndarray[Any, numpy.dtype[numpy.uint8]] | None,
    space: int,
    presence: bool,
) -> tuple[
    numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
]: ...
