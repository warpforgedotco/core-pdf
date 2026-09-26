# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def composite_elementary_normal(
    destination: numpy.ndarray[Any, Any],
    rendered: numpy.ndarray[Any, Any],
    source_alpha: numpy.ndarray[Any, Any],
) -> numpy.ndarray[Any, Any]: ...
def composite_masked_normal(
    destination: numpy.ndarray[Any, Any],
    rendered: numpy.ndarray[Any, Any],
    opacity: float,
    mask_codes: numpy.ndarray[Any, Any],
    mask_table: numpy.ndarray[Any, Any],
) -> numpy.ndarray[Any, Any]: ...
def composite_normal_group(
    destination: numpy.ndarray[Any, Any],
    rendered: numpy.ndarray[Any, Any],
    source_alpha_scale: float,
    target_alpha_scale: float = ...,
    effective_plane: bool = ...,
) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]] | None: ...
