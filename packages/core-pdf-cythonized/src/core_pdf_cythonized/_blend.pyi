# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def blend_normal_alpha_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
    alpha: numpy.ndarray[Any, Any],
) -> None: ...
