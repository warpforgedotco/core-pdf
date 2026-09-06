# SPDX-License-Identifier: AGPL-3.0-only
"""Device-independent PDF color equations."""

from __future__ import annotations

from typing import Any

import numpy

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


def lab_to_xyz(
    values: ColorSamples,
    white_point: tuple[float, float, float],
) -> ColorSamples:
    l_star = values[:, 0] * 100.0
    a_star = values[:, 1] * 255.0 - 128.0
    b_star = values[:, 2] * 255.0 - 128.0
    fy = (l_star + 16.0) / 116.0
    fx = a_star / 500.0 + fy
    fz = fy - b_star / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = numpy.where(fx**3 > eps, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = numpy.where(l_star > kappa * eps, fy**3, l_star / kappa)
    zr = numpy.where(fz**3 > eps, fz**3, (116.0 * fz - 16.0) / kappa)
    return numpy.column_stack((xr, yr, zr)).astype(numpy.float32) * numpy.asarray(
        white_point,
        dtype=numpy.float32,
    )
