# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def shading_values(
    kind: int,
    coords: Any,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    allowed: numpy.ndarray[Any, Any],
    extend0: bool,
    extend1: bool,
    domain0: float,
    domain_span: float,
    half: float,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]: ...
def shading_blend(
    pixels: numpy.ndarray[Any, Any],
    ix0: int,
    iy0: int,
    painted: numpy.ndarray[Any, Any],
    color_index: numpy.ndarray[Any, Any],
    colors: numpy.ndarray[Any, Any],
    alpha_plane: numpy.ndarray[Any, Any] | None = ...,
) -> tuple[int, int, int, int] | None: ...
