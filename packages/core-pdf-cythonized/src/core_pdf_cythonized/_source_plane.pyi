# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def accumulate_source_plane(
    plane: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    coverage: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    scale: float,
) -> None: ...
