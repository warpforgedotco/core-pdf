# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def outline_edges(
    xs: numpy.ndarray[Any, Any], ys: numpy.ndarray[Any, Any], spans: Any
) -> tuple[numpy.ndarray[Any, Any] | None, list[tuple[int, int, bool]], bool]: ...
def glyph_outline_edges(
    xs: numpy.ndarray[Any, Any],
    ys: numpy.ndarray[Any, Any],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    spans: Any,
    simple: bool = False,
) -> tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any] | None,
    list[tuple[int, int, bool]],
    bool,
    tuple[float, float, float, float] | None,
]: ...
def outline_coordinate_arrays(
    contours: object,
) -> (
    tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], tuple[tuple[int, int], ...]] | None
): ...
def outline_simple(
    xs: numpy.ndarray[Any, Any], ys: numpy.ndarray[Any, Any], spans: Any
) -> bool: ...
