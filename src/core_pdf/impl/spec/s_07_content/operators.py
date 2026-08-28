# SPDX-License-Identifier: AGPL-3.0-only
"""Compiled content operator helpers."""

from __future__ import annotations

from functools import lru_cache
from math import hypot

MATRIX_TOLERANCE = 0.1


@lru_cache(maxsize=128)
def detect_rotation_from_linear(
    A: float, B: float, C: float, D: float, tolerance: float = MATRIX_TOLERANCE
) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    if (
        abs(na - 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd - 1.0) < tolerance
    ):
        return 0
    if (
        abs(na) < tolerance
        and abs(nb - 1.0) < tolerance
        and abs(nc + 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 90
    if (
        abs(na + 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd + 1.0) < tolerance
    ):
        return 180
    if (
        abs(na) < tolerance
        and abs(nb + 1.0) < tolerance
        and abs(nc - 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 270
    return 0


__all__ = ("detect_rotation_from_linear",)
