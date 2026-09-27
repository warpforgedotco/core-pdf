# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.caches import BoundedDict


class GlyphOutlineArrays:
    __slots__ = ("linear", "spans", "xs", "ys")

    def __init__(
        self,
        xs: numpy.ndarray[Any, Any],
        ys: numpy.ndarray[Any, Any],
        spans: tuple[tuple[int, int], ...],
    ) -> None:
        self.xs = xs
        self.ys = ys
        self.spans = spans
        self.linear: BoundedDict[
            tuple[float, float, float, float],
            tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]],
        ] = BoundedDict(LINEAR_CACHE_LIMIT)

    def linear_columns(
        self, a: float, b: float, c: float, d: float
    ) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
        key = (a, b, c, d)
        columns = self.linear.get(key)
        if columns is None:
            xs = self.xs
            ys = self.ys
            columns = self.linear.put(key, (xs * a + ys * c, xs * b + ys * d))
        return columns


LINEAR_CACHE_LIMIT = 256


def outline_arrays(
    contours: tuple[tuple[tuple[float, float], ...], ...],
) -> GlyphOutlineArrays | None:
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int]] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        start = len(xs)
        for x, y in contour:
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs)))
    if not spans:
        return None
    return GlyphOutlineArrays(
        numpy.asarray(xs, dtype=numpy.float64), numpy.asarray(ys, dtype=numpy.float64), tuple(spans)
    )
