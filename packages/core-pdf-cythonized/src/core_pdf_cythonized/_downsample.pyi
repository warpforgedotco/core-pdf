# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def box_downsample_blocks(
    grid: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    row_edges: numpy.ndarray[Any, numpy.dtype[numpy.int64]],
    column_edges: numpy.ndarray[Any, numpy.dtype[numpy.int64]],
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]: ...
