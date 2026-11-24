# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

NAMED: int
ERASED: int
KEPT: int
GROUP: int

Bytes = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]
Floats = numpy.ndarray[Any, numpy.dtype[numpy.float32]]

def composite_spot_plate(
    values: Bytes,
    below: Bytes | None,
    inner: Bytes | None,
    previous: Bytes | None,
    layer: Bytes | None,
    marked: int,
    source: Bytes,
    result: Bytes,
    current: Bytes,
    backdrop_alpha: Bytes,
    element_alpha: Bytes,
    group_alpha: Floats | None,
    cover: Floats | None,
    mode: int,
) -> None: ...
def apply_spot_plate(pixels: Bytes, ink: Bytes) -> None: ...
