# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

GroupSamples = numpy.ndarray[Any, numpy.dtype[numpy.float64]]

def composite_knockout_element(
    components: numpy.ndarray[Any, Any],
    alpha: numpy.ndarray[Any, Any],
    *,
    backdrop_components: numpy.ndarray[Any, Any],
    backdrop_alpha: numpy.ndarray[Any, Any],
    element_components: numpy.ndarray[Any, Any],
    element_alpha: numpy.ndarray[Any, Any],
    shape: numpy.ndarray[Any, Any],
    group_alpha: numpy.ndarray[Any, Any],
    element_group_alpha: numpy.ndarray[Any, Any],
    validate: bool = ...,
) -> tuple[GroupSamples, GroupSamples, GroupSamples]: ...
def composite_knockout_group(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    backdrop: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    element: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    group_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    element_alpha: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    shape: numpy.ndarray[Any, Any],
) -> None: ...
