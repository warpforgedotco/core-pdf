# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def rect_coverage_plane(
    ix0: int,
    ix1: int,
    iy0: int,
    iy1: int,
    left: float,
    right: float,
    top: float,
    bottom: float,
    scale: int,
) -> numpy.ndarray[Any, Any]: ...
