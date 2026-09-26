# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

Int64Array = numpy.ndarray[Any, numpy.dtype[numpy.int64]]

def truetype_contours(
    glyf: bytes,
    loca: Int64Array,
    lsb: Int64Array,
    glyph_count: int,
    gid: int,
    scale: float,
) -> tuple[tuple[tuple[float, float], ...], ...] | None: ...
