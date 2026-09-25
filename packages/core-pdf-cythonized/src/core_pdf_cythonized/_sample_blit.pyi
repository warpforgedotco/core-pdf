# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def sample_opaque_pixels(
    target: numpy.ndarray[Any, Any],
    source: numpy.ndarray[Any, Any],
    source_y: numpy.ndarray[Any, Any],
    source_x: numpy.ndarray[Any, Any],
    valid_rows: numpy.ndarray[Any, Any],
    valid_columns: numpy.ndarray[Any, Any],
    transposed: bool,
) -> None: ...
