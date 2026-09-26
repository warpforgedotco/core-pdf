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
def interleave_soft_mask(
    raster: numpy.ndarray[Any, Any],
    channels: int,
    mask: numpy.ndarray[Any, Any],
    mask_rows: numpy.ndarray[Any, Any],
    mask_columns: numpy.ndarray[Any, Any],
) -> numpy.ndarray[Any, Any]: ...
def alpha_channel(
    pixels: numpy.ndarray[Any, numpy.dtype[numpy.uint8]], presence: bool
) -> tuple[
    numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
]: ...
