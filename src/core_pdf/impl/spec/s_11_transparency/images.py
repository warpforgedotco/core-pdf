# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy


def unblend_matte_components(
    values: numpy.ndarray[Any, Any],
    alpha: numpy.ndarray[Any, Any],
    matte: tuple[float, ...],
) -> numpy.ndarray[Any, Any]:
    components = numpy.asarray(values, dtype=numpy.float64)
    opacity = numpy.asarray(alpha, dtype=numpy.float64)
    background = numpy.asarray(matte, dtype=numpy.float64)
    if (
        components.ndim != 2
        or opacity.shape != (components.shape[0],)
        or background.shape != (components.shape[1],)
        or not numpy.isfinite(components).all()
        or not numpy.isfinite(opacity).all()
        or not numpy.isfinite(background).all()
        or numpy.any((opacity < 0) | (opacity > 1))
    ):
        raise ValueError("invalid image matte components")
    difference = numpy.zeros_like(components)
    numpy.divide(
        components - background,
        opacity[:, None],
        out=difference,
        where=opacity[:, None] != 0,
    )
    return background + difference


__all__ = ("unblend_matte_components",)
