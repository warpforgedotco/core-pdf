# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

Rows = numpy.ndarray[Any, numpy.dtype[numpy.uint16]]
Indices = numpy.ndarray[Any, numpy.dtype[numpy.uint32]]

def distinct_uint16_rows(samples: Rows, limit: int) -> tuple[Rows, Indices] | None: ...
def gather_uint8_rows(
    table: numpy.ndarray[Any, numpy.dtype[numpy.uint8]], indices: Indices
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]: ...
