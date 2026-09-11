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
    """Convert normalized Lab values using the original scaling convention.

    Each row encodes L* as L*/100 and a*/b* as (component + 128)/255.
    Use ``lab_components_to_xyz`` for actual PDF Lab components.
    """
    components = numpy.column_stack(
        (values[:, 0] * 100.0, values[:, 1] * 255.0 - 128.0, values[:, 2] * 255.0 - 128.0)
    )
    return lab_components_to_xyz(components, white_point)


def lab_components_to_xyz(
    values: ColorSamples,
    white_point: tuple[float, float, float],
) -> ColorSamples:
    """Convert rows of actual L*, a*, b* components to CIE XYZ.

    ISO 32000-1, 8.6.5.4 defines the Lab transformation relative to WhitePoint.
    ``values`` has shape (n, 3); L* spans 0 to 100 and a*/b* use the color
    space's component ranges. Callers decode samples and enforce those ranges
    before conversion. The input is unchanged and the result is float32.
    """
    l_star = values[:, 0]
    a_star = values[:, 1]
    b_star = values[:, 2]
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


__all__ = (
    "ColorSamples",
    "lab_components_to_xyz",
    "lab_to_xyz",
)
