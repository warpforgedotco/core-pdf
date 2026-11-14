# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Sequence
from typing import Any

import numpy

def function_values(
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    allowed: numpy.ndarray[Any, Any],
    inverse: Sequence[float] | None,
    bbox: Sequence[float] | None,
    to_domain: Sequence[float],
    domain: Sequence[float],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]: ...
def mesh_values(
    triangles: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    allowed: numpy.ndarray[Any, Any],
    forward: Sequence[float] | None = None,
    inverse: Sequence[float] | None = None,
    bbox: Sequence[float] | None = None,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]: ...
def patch_triangles(
    patches: numpy.ndarray[Any, Any],
    corners: numpy.ndarray[Any, Any],
    linear: Sequence[float],
    tolerance: float,
    color_tolerances: numpy.ndarray[Any, Any],
    max_divisions: int,
) -> numpy.ndarray[Any, Any]: ...
