# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055


def xyz_to_srgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl)


def lab_to_xyz(
    l_star: float, a: float, b: float, wp: list[float] | tuple[float, float, float]
) -> tuple[float, float, float]:
    fy = (l_star + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = fx**3 if fx**3 > eps else (116.0 * fx - 16.0) / kappa
    yr = ((l_star + 16.0) / 116.0) ** 3 if l_star > kappa * eps else l_star / kappa
    zr = fz**3 if fz**3 > eps else (116.0 * fz - 16.0) / kappa
    return xr * wp[0], yr * wp[1], zr * wp[2]


def adapt_d50_to_d65(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax = 0.955473 * x - 0.023098 * y + 0.063259 * z
    ay = -0.028369 * x + 1.009995 * y + 0.021300 * z
    az = 0.012314 * x - 0.020507 * y + 1.330365 * z
    return ax, ay, az
