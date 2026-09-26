# SPDX-License-Identifier: AGPL-3.0-only

from array import array
from collections.abc import Callable, Sequence
from typing import Any

import numpy

Rectangle = tuple[float, float, float, float]

def flatten_path_commands(
    ops: bytes | bytearray,
    coords: Any,
    matrix: Sequence[float] | None,
    hypot: Callable[[float, float], float],
    line_rows: array[float] | None,
    line_width: float,
) -> tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    list[tuple[int, int, bool]],
    Rectangle | None,
    bool,
]: ...
def path_bounds(
    xs: numpy.ndarray[Any, Any], ys: numpy.ndarray[Any, Any], spans: list[tuple[int, int, bool]]
) -> tuple[Rectangle | None, bool]: ...
