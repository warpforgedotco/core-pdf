# SPDX-License-Identifier: AGPL-3.0-only
"""Packed stroke-atlas rasterization for vector-text recognition."""

from __future__ import annotations

import math

import numpy

from core_pdf.impl.render.model import DisplayItem, PathPaintItem, PathPaintKind, RasterImage
from core_pdf.impl.runtime.array_views import uint8_view
from core_pdf.impl.spec.s_07_content.capture import CapturedPath


def rasterize_packed_stroked_paths(
    items: tuple[DisplayItem, ...],
    width: float,
    height: float,
    scale: float,
) -> RasterImage:
    """Rasterize the opaque packed OCR atlas with a lightweight line kernel.

    Packed vector text is a deliberately narrow rendering mode: every item is a
    solid black stroke, there are no clips or blend modes, and OCR benefits more
    from a clean one-pixel antialiased skeleton than from the general renderer's
    full 4x supersampling.  Xiaolin-Wu coverage keeps diagonal glyph strokes
    legible while avoiding one Python pixel loop per 4x4 sample.
    """
    raster_scale = max(0.01, float(scale))
    raster_width = max(1, int(round(width * raster_scale)))
    raster_height = max(1, int(round(height * raster_scale)))
    # Flat bytearray while plotting: per-pixel numpy scalar reads/writes are far
    # slower than plain byte indexing; the buffer becomes an array once at the end.
    gray = bytearray(b"\xff" * (raster_height * raster_width))
    page_height = float(height)

    def plot(x: int, y: int, coverage: float) -> None:
        if coverage <= 0.0 or not (0 <= x < raster_width and 0 <= y < raster_height):
            return
        index = y * raster_width + x
        value = gray[index]
        gray[index] = max(0, min(255, round(value * (1.0 - coverage))))

    def draw_line(x0: float, y0: float, x1: float, y1: float) -> None:
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0, x1, y1 = y0, x0, y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        delta_x = x1 - x0
        delta_y = y1 - y0
        if abs(delta_x) <= 1e-12:
            return
        gradient = delta_y / delta_x

        first_x = round(x0)
        first_y = y0 + gradient * (first_x - x0)
        first_gap = 1.0 - ((x0 + 0.5) - math.floor(x0 + 0.5))
        first_y_integer = math.floor(first_y)
        plot(
            first_y_integer if steep else first_x,
            first_x if steep else first_y_integer,
            (1.0 - (first_y - first_y_integer)) * first_gap,
        )
        plot(
            first_y_integer + 1 if steep else first_x,
            first_x if steep else first_y_integer + 1,
            (first_y - first_y_integer) * first_gap,
        )
        intersect_y = first_y + gradient

        last_x = round(x1)
        last_y = y1 + gradient * (last_x - x1)
        last_gap = (x1 + 0.5) - math.floor(x1 + 0.5)
        last_y_integer = math.floor(last_y)
        plot(
            last_y_integer if steep else last_x,
            last_x if steep else last_y_integer,
            (1.0 - (last_y - last_y_integer)) * last_gap,
        )
        plot(
            last_y_integer + 1 if steep else last_x,
            last_x if steep else last_y_integer + 1,
            (last_y - last_y_integer) * last_gap,
        )

        for pixel_x in range(first_x + 1, last_x):
            pixel_y = math.floor(intersect_y)
            plot(
                pixel_y if steep else pixel_x,
                pixel_x if steep else pixel_y,
                1.0 - (intersect_y - pixel_y),
            )
            plot(
                pixel_y + 1 if steep else pixel_x,
                pixel_x if steep else pixel_y + 1,
                intersect_y - pixel_y,
            )
            intersect_y += gradient

    for item in items:
        if type(item) is not PathPaintItem or item.paint_kind is not PathPaintKind.STROKE:
            continue
        path = item.path
        if type(path) is not CapturedPath:
            continue
        line_width = float(item.line_width or 1.0)
        thickness = max(1, round(line_width * raster_scale * 0.5))
        offset_start = -(thickness - 1) * 0.5
        for subpath in path.subpaths:
            points = subpath.points
            if len(points) < 2:
                continue
            segments = list(zip(points, points[1:], strict=False))
            if subpath.closed and points[0] != points[-1]:
                segments.append((points[-1], points[0]))
            for (x0, y0), (x1, y1) in segments:
                x0 *= raster_scale
                x1 *= raster_scale
                y0 = (page_height - y0) * raster_scale
                y1 = (page_height - y1) * raster_scale
                delta_x = x1 - x0
                delta_y = y1 - y0
                segment_length = math.hypot(delta_x, delta_y)
                if segment_length <= 1e-12:
                    continue
                normal_x = -delta_y / segment_length
                normal_y = delta_x / segment_length
                for offset_index in range(thickness):
                    offset = offset_start + offset_index
                    draw_line(
                        x0 + normal_x * offset,
                        y0 + normal_y * offset,
                        x1 + normal_x * offset,
                        y1 + normal_y * offset,
                    )

    rgba = numpy.empty((raster_height, raster_width, 4), dtype=numpy.uint8)
    rgba[:, :, :3] = uint8_view(gray).reshape(raster_height, raster_width)[:, :, None]
    rgba[:, :, 3] = 255
    return RasterImage(rgba, raster_width, raster_height, 4)
