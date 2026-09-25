# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def signed_area_coverage(
    edges: numpy.ndarray[Any, Any], width: int, height: int
) -> numpy.ndarray[Any, Any]: ...
def glyph_coverage_plane(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    width: int,
    height: int,
) -> numpy.ndarray[Any, Any] | None: ...
def glyph_alpha_planes(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    width: int,
    height: int,
    alpha: float,
    want_shape: bool,
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any] | None] | None: ...
