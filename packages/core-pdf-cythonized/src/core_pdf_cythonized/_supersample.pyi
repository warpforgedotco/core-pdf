# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def supersampled_coverage_plane(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    evenodd: bool,
) -> tuple[numpy.ndarray[Any, Any], int] | None: ...
