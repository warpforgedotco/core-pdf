# SPDX-License-Identifier: AGPL-3.0-only

"""Adobe TN 5177: points on the cubic curves charstring path operators draw."""

from __future__ import annotations


def cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    mt = 1.0 - t
    mt3 = mt**3
    t3 = t**3
    mt2t = 3.0 * mt * mt * t
    mtt2 = 3.0 * mt * t * t
    return (
        mt3 * p0[0] + mt2t * p1[0] + mtt2 * p2[0] + t3 * p3[0],
        mt3 * p0[1] + mt2t * p1[1] + mtt2 * p2[1] + t3 * p3[1],
    )


__all__ = ("cubic_point",)
