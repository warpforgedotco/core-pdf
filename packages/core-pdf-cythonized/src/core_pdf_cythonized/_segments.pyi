# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def merge_collinear_rows(
    rows: numpy.ndarray[Any, Any],
    coordinate: int,
    start: int,
    end: int,
    tolerance: float,
    reach: float,
) -> numpy.ndarray[Any, Any]: ...
