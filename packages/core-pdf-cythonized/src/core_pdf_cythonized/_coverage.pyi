# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

class SimpleEdges(numpy.ndarray[Any, Any]): ...

def windings_mix(edges: numpy.ndarray[Any, Any]) -> bool: ...
def area_coverage_plane(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    width: int,
    height: int,
    mode: int = 0,
) -> tuple[numpy.ndarray[Any, Any], int, numpy.ndarray[Any, Any]] | None: ...
def signed_area_coverage(
    edges: numpy.ndarray[Any, Any], width: int, height: int
) -> numpy.ndarray[Any, Any]: ...
def glyph_coverage_plane(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: float,
    iy0: float,
    width: int,
    height: int,
) -> numpy.ndarray[Any, Any] | None: ...
def fill_glyph_coverage(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    width: int,
    height: int,
    rgba: tuple[int, int, int, int],
    target: numpy.ndarray[Any, Any],
    source_alpha: numpy.ndarray[Any, Any] | None,
    source_shape: numpy.ndarray[Any, Any] | None,
    shape_scale: float,
) -> bool | None: ...
def fill_glyph_coverage_at(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    width: int,
    height: int,
    rgba: tuple[int, int, int, int],
    target: numpy.ndarray[Any, Any],
    source_alpha: numpy.ndarray[Any, Any] | None,
    source_shape: numpy.ndarray[Any, Any] | None,
    shape_scale: float,
) -> bool | None: ...
def fill_glyph_knockout(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    width: int,
    height: int,
    rgba: tuple[int, int, int, int],
    destination: numpy.ndarray[Any, Any],
    backdrop: numpy.ndarray[Any, Any],
    group_alpha: numpy.ndarray[Any, Any],
    parent_shape: numpy.ndarray[Any, Any] | None,
    shape_scale: float,
) -> bool | None: ...
def fill_glyph_knockout_at(
    edges: numpy.ndarray[Any, Any],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    ix0: int,
    iy0: int,
    width: int,
    height: int,
    rgba: tuple[int, int, int, int],
    destination: numpy.ndarray[Any, Any],
    backdrop: numpy.ndarray[Any, Any],
    group_alpha: numpy.ndarray[Any, Any],
    parent_shape: numpy.ndarray[Any, Any] | None,
    shape_scale: float,
) -> bool | None: ...
