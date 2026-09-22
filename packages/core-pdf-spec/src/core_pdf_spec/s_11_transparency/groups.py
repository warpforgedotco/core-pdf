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
    validate: bool = True,
) -> tuple[GroupSamples, GroupSamples, GroupSamples]:
    color = numpy.asarray(components, dtype=numpy.float64)
    complete = numpy.asarray(alpha, dtype=numpy.float64)
    backdrop = numpy.asarray(backdrop_components, dtype=numpy.float64)
    initial = numpy.asarray(backdrop_alpha, dtype=numpy.float64)
    element = numpy.asarray(element_components, dtype=numpy.float64)
    element_complete = numpy.asarray(element_alpha, dtype=numpy.float64)
    coverage = numpy.asarray(shape, dtype=numpy.float64)
    accumulated = numpy.asarray(group_alpha, dtype=numpy.float64)
    element_accumulated = numpy.asarray(element_group_alpha, dtype=numpy.float64)
    if (
        color.ndim < 1
        or color.shape[-1] == 0
        or backdrop.shape != color.shape
        or element.shape != color.shape
        or complete.shape != color.shape[:-1]
        or any(
            values.shape != complete.shape
            for values in (initial, element_complete, coverage, accumulated, element_accumulated)
        )
        or (
            validate
            and (
                not unit_range(
                    color,
                    complete,
                    backdrop,
                    initial,
                    element,
                    element_complete,
                    coverage,
                    accumulated,
                    element_accumulated,
                )
                or bool(numpy.any(element_accumulated > coverage))
            )
        )
    ):
        raise ValueError("invalid knockout group samples")
    remaining = 1.0 - coverage
    result_group_alpha = numpy.asarray(
        element_accumulated + remaining * accumulated, dtype=numpy.float64
    )
    result_alpha = numpy.asarray(
        initial + (1.0 - initial) * result_group_alpha, dtype=numpy.float64
    )
    premultiplied = element * element_complete[..., None] + remaining[..., None] * (
        color * complete[..., None] - backdrop * initial[..., None]
    )
    result = numpy.zeros_like(color)
    numpy.divide(
        premultiplied,
        result_alpha[..., None],
        out=result,
        where=result_alpha[..., None] != 0.0,
    )
    return result, result_alpha, result_group_alpha


__all__ = ("GroupSamples", "composite_knockout_element", "remove_group_backdrop")
