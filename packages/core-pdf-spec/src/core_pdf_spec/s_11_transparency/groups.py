# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

GroupSamples = numpy.ndarray[Any, numpy.dtype[numpy.float64]]


def unit_range(*arrays: numpy.ndarray[Any, Any]) -> bool:
    for values in arrays:
        if values.size and not (values.min() >= 0.0 and values.max() <= 1.0):
            return False
    return True


def remove_group_backdrop(
    components: numpy.ndarray[Any, Any],
    alpha: numpy.ndarray[Any, Any],
    backdrop_components: numpy.ndarray[Any, Any],
    backdrop_alpha: numpy.ndarray[Any, Any],
    group_alpha: numpy.ndarray[Any, Any],
    *,
    validate: bool = True,
) -> tuple[GroupSamples, GroupSamples]:
    color = numpy.asarray(components, dtype=numpy.float64)
    complete = numpy.asarray(alpha, dtype=numpy.float64)
    backdrop = numpy.asarray(backdrop_components, dtype=numpy.float64)
    initial = numpy.asarray(backdrop_alpha, dtype=numpy.float64)
    accumulated = numpy.asarray(group_alpha, dtype=numpy.float64)
    if (
        color.ndim < 1
        or color.shape[-1] == 0
        or backdrop.shape != color.shape
        or complete.shape != color.shape[:-1]
        or initial.shape != complete.shape
        or accumulated.shape != complete.shape
        or (validate and not unit_range(color, complete, backdrop, initial, accumulated))
    ):
        raise ValueError("invalid transparency group samples")
    backdrop_weight = (1.0 - accumulated) * initial
    premultiplied = color * complete[..., None] - backdrop * backdrop_weight[..., None]
    source = numpy.zeros_like(color)
    numpy.divide(
        premultiplied,
        accumulated[..., None],
        out=source,
        where=accumulated[..., None] != 0.0,
    )
    return source, accumulated.copy()


__all__ = ("GroupSamples", "remove_group_backdrop")
