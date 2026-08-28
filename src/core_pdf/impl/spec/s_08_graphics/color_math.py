# SPDX-License-Identifier: AGPL-3.0-only
"""Vectorized color-space conversion math."""

from __future__ import annotations

from typing import Any

import numpy

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]

SRGB_MATRIX = numpy.asarray(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    ),
    dtype=numpy.float32,
)
D50_TO_D65_MATRIX = numpy.asarray(
    (
        (0.955473, -0.023098, 0.063259),
        (-0.028369, 1.009995, 0.021300),
        (0.012314, -0.020507, 1.330365),
    ),
    dtype=numpy.float32,
)


# One matmul for the ubiquitous xyz_to_srgb(adapt_d50_to_d65(x)) tail:
# x @ D50.T @ SRGB.T == x @ (SRGB @ D50).T.
D50_XYZ_TO_SRGB_MATRIX = (SRGB_MATRIX @ D50_TO_D65_MATRIX).astype(numpy.float32)


def linear_to_srgb(values: ColorSamples) -> ColorSamples:
    clipped = numpy.clip(values, 0.0, None)
    return numpy.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * numpy.power(clipped, 1.0 / 2.4) - 0.055,
    ).astype(numpy.float32, copy=False)


def xyz_to_srgb(values: ColorSamples) -> ColorSamples:
    return linear_to_srgb(values @ SRGB_MATRIX.T)


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


def adapt_d50_to_d65(values: ColorSamples) -> ColorSamples:
    return (values @ D50_TO_D65_MATRIX.T).astype(numpy.float32, copy=False)


def d50_xyz_to_srgb(values: ColorSamples) -> ColorSamples:
    """Convert D50 XYZ straight to normalized sRGB with a single matmul."""
    return linear_to_srgb(values @ D50_XYZ_TO_SRGB_MATRIX.T)
