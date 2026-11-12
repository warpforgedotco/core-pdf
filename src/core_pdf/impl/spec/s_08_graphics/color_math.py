# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


def lab_components_to_xyz(
    values: ColorSamples,
    white_point: tuple[float, float, float],
) -> ColorSamples:
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


def xyz_to_lab_components(
    values: numpy.ndarray[Any, Any], white_point: tuple[float, float, float]
) -> numpy.ndarray[Any, Any]:
    white = numpy.asarray(white_point, dtype=numpy.float64)
    xyz = numpy.asarray(values, dtype=numpy.float64)
    if white.shape != (3,) or numpy.any(white <= 0) or not numpy.isfinite(white).all():
        raise ValueError("invalid Lab white point")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not numpy.isfinite(xyz).all():
        raise ValueError("invalid XYZ components")
    normalized = xyz / white
    delta = 6.0 / 29.0
    f = numpy.where(
        normalized > delta**3, numpy.cbrt(normalized), normalized / (3 * delta**2) + 4.0 / 29.0
    )
    return numpy.column_stack(
        (116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2]))
    )


def compensate_black_point_xyz(
    values: numpy.ndarray[Any, Any],
    white_point: tuple[float, float, float],
    source_black: tuple[float, float, float],
    destination_black: tuple[float, float, float],
) -> numpy.ndarray[Any, Any]:
    white = numpy.asarray(white_point, dtype=numpy.float64)
    source = numpy.asarray(source_black, dtype=numpy.float64)
    destination = numpy.asarray(destination_black, dtype=numpy.float64)
    xyz = numpy.asarray(values, dtype=numpy.float64)
    if any(endpoint.shape != (3,) for endpoint in (white, source, destination)):
        raise ValueError("invalid black point endpoints")
    if (
        not numpy.isfinite([white, source, destination]).all()
        or numpy.any(white <= 0)
        or numpy.any(source < 0)
        or numpy.any(destination < 0)
        or numpy.any(source >= white)
        or numpy.any(destination >= white)
    ):
        raise ValueError("invalid black point endpoints")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not numpy.isfinite(xyz).all():
        raise ValueError("invalid XYZ components")
    scale = (white - destination) / (white - source)
    return xyz * scale + destination - source * scale


__all__ = (
    "compensate_black_point_xyz",
    "xyz_to_lab_components",
    "ColorSamples",
    "lab_components_to_xyz",
)
