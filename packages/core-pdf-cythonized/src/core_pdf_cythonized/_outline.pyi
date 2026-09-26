# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def outline_edges(
    xs: numpy.ndarray[Any, Any],
    ys: numpy.ndarray[Any, Any],
    spans: Any,
) -> tuple[numpy.ndarray[Any, Any] | None, list[tuple[int, int, bool]], bool]: ...
def translated_outline_edges(
    linear_x: numpy.ndarray[Any, Any],
    linear_y: numpy.ndarray[Any, Any],
    e: float,
    f: float,
    spans: Any,
) -> tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any] | None,
    list[tuple[int, int, bool]],
    bool,
    tuple[float, float, float, float] | None,
]: ...
