# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def composite_nonisolated_blend(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rendered: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    opacity: float,
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None,
    mode: int,
    copy_unmasked: bool,
    revised: bool = ...,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]: ...
