# SPDX-License-Identifier: AGPL-3.0-only
"""Gradient-pattern geometry and colour evaluation."""

from __future__ import annotations

import math
from typing import Any

from core_pdf.impl.render.blend import internal_clamp01, internal_color_component
from core_pdf.impl.spec.s_08_graphics.device_profiles import cmyk_floats_to_srgb


def axial_shading_t(coords: list[float] | tuple[float, ...], px: float, py: float) -> float | None:
    x0, y0, x1, y1 = coords[:4]
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return None
    return ((px - x0) * dx + (py - y0) * dy) / denom


def radial_shading_t(coords: list[float] | tuple[float, ...], px: float, py: float) -> float | None:
    x0, y0, r0, x1, y1, r1 = coords[:6]
    dx = x1 - x0
    dy = y1 - y0
    dr = r1 - r0
    qx = px - x0
    qy = py - y0
    a = dx * dx + dy * dy - dr * dr
    b = -2.0 * (qx * dx + qy * dy + r0 * dr)
    c = qx * qx + qy * qy - r0 * r0
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return None
        return -c / b
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = disc**0.5
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    valid = [t for t in (t0, t1) if math.isfinite(t)]
    if not valid:
        return None
    in_range = [t for t in valid if 0.0 <= t <= 1.0]
    return max(in_range) if in_range else min(valid, key=lambda t: abs(t - 0.5))


def internal_shading_color_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    opacity: Any,
) -> tuple[int, int, int, int]:
    alpha = internal_color_component(opacity, 255) if type(opacity) in {int, float} else 255
    name = color_model or "DeviceRGB"
    if name.endswith("DeviceGray") or len(components) == 1:
        gray = internal_color_component(components[0] if components else 0.0)
        return gray, gray, gray, alpha
    if name.endswith("DeviceCMYK") and len(components) >= 4:
        c, m, y, k = (internal_clamp01(v) for v in components[:4])
        red, green, blue = cmyk_floats_to_srgb(c, m, y, k)
        return red, green, blue, alpha
    rgb = [internal_color_component(c) for c in components[:3]]
    while len(rgb) < 3:
        rgb.append(rgb[-1] if rgb else 0)
    return rgb[0], rgb[1], rgb[2], alpha
