# SPDX-License-Identifier: AGPL-3.0-only
"""Transparency-group backdrop removal in the group's blending colour space."""

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


__all__ = ("GroupSamples", "remove_group_backdrop")
