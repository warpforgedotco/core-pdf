# SPDX-License-Identifier: AGPL-3.0-only
"""Transparency-group calculations in the group's blending colour space."""

from __future__ import annotations

from typing import Any

import numpy

GroupSamples = numpy.ndarray[Any, numpy.dtype[numpy.float64]]


def remove_group_backdrop(
    components: numpy.ndarray[Any, Any],
    alpha: numpy.ndarray[Any, Any],
    backdrop_components: numpy.ndarray[Any, Any],
    backdrop_alpha: numpy.ndarray[Any, Any],
    group_alpha: numpy.ndarray[Any, Any],
) -> tuple[GroupSamples, GroupSamples]:
    """Return the group's source colour and alpha, ISO 32000-1/2 11.4.4/11.4.8.

    ``components`` and ``alpha`` describe the accumulated result including its
    initial backdrop. ``group_alpha`` contains only the group elements' alpha,
    accumulated independently of that backdrop. For a non-knockout group it is
    the union of its source alphas; it cannot be recovered from complete alpha
    when the initial backdrop is opaque. Outer group opacity, mask and blend
    mode apply subsequently, when these returned samples are composited.

    Colours have shape ``(..., channels)`` and alphas have shape ``(...)``.
    All inputs must be finite, unit-range samples with matching dimensions.
    The caller supplies complete alpha satisfying ``Union(backdrop_alpha,
    group_alpha)``; independent raster rounding of these inputs is permitted.
    The returned float64 arrays are new, unquantized, and unclipped. Inexact
    inputs may produce colours outside the colour space's component range.

    Backdrop removal reverses Normal compositing of the initial backdrop:
    ``Cg = (alpha * Cn - (1 - group_alpha) * backdrop_alpha * C0) / group_alpha``.
    Where group alpha is zero the source colour has no effect; return zero
    without dividing. An isolated group's initial backdrop alpha is zero.
    """
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
        or any(
            not numpy.all((values >= 0.0) & (values <= 1.0))
            for values in (color, complete, backdrop, initial, accumulated)
        )
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
) -> tuple[GroupSamples, GroupSamples, GroupSamples]:
    """Insert an element into a knockout group, ISO 32000-1/2 11.4.6/11.4.8.

    ``components`` and ``alpha`` are the previous accumulated group result,
    including its initial backdrop. The element's colour and complete alpha
    are its ordinary compositing result against that same initial backdrop,
    already incorporating the element's shape, opacity, mask and blend mode.
    A nested non-isolated group also uses that initial backdrop, rather than
    the preceding group's result (11.4.6 Note 6).

    ``shape`` is the element's effective source shape. ``group_alpha`` and
    ``element_group_alpha`` exclude the initial backdrop; the latter includes
    the effective shape and opacity, so it must not exceed ``shape``. Shape
    remains relevant when element alpha is zero: a transparent object still
    knocks out earlier elements wherever its shape is nonzero.

    Colours have shape ``(..., channels)`` and alphas and shape have shape
    ``(...)``. Inputs must be finite, unit-range samples with matching
    dimensions. Independent rounding of complete alpha is permitted, as in
    ``remove_group_backdrop``. All returned arrays are new float64 samples,
    ordered colour, complete alpha, and group alpha. Colours are neither
    clipped nor quantized; inexact raster inputs may put them out of range.

    With premultiplied colours P, the result is
    ``P = P_element + (1 - shape) * (P_previous - P_initial)`` and its group
    alpha is ``element_group_alpha + (1 - shape) * group_alpha``. Complete
    alpha is the union of that group alpha and initial backdrop alpha.
    At zero complete alpha the colour is immaterial; return zero safely.
    """
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
        or any(
            not numpy.all((values >= 0.0) & (values <= 1.0))
            for values in (
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
        )
        or numpy.any(element_accumulated > coverage)
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
