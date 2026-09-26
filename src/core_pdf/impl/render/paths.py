# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array
from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.geometry import intersect_bbox

RASTER_KERNEL_MIN_PIXEL_AREA = 64
RASTER_CIRCLE_MIN_PIXEL_AREA = 16
CIRCLE_VERTICES = tuple(
    (math.cos(index * math.tau / 32), math.sin(index * math.tau / 32)) for index in range(32)
)


def circle_path(cx: float, cy: float, radius: float) -> CapturedPath:
    return CapturedPath(
        [
            CapturedSubpath(
                [(cx + radius * x, cy + radius * y) for x, y in CIRCLE_VERTICES],
                closed=True,
            )
        ]
    )


def dash_subpath(
    subpath: CapturedSubpath, dash_pattern: tuple[list[float], float]
) -> list[CapturedSubpath]:
    lengths = [max(0.0, float(value)) for value in dash_pattern[0]]
    if len(lengths) % 2:
        lengths *= 2
    total = sum(lengths)
    if not lengths or total <= 0.0:
        return [subpath]
    points = subpath.points
    if len(points) < 2:
        return []
    vertices = [*points, points[0]] if subpath.closed and points[-1] != points[0] else points
    phase = float(dash_pattern[1]) % total
    index = 0
    while phase > 0.0 and phase >= lengths[index]:
        phase -= lengths[index]
        index = (index + 1) % len(lengths)
    remaining = lengths[index] - phase
    pieces: list[CapturedSubpath] = []
    current: list[tuple[float, float]] = []
    for start, end in zip(vertices, vertices[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= 0.0:
            continue
        position = 0.0
        while position < distance or remaining <= 0.0:
            point = (
                start[0] + dx * (position / distance),
                start[1] + dy * (position / distance),
            )
            if remaining <= 0.0:
                if index % 2 == 0 and lengths[index] == 0.0:
                    pieces.append(CapturedSubpath([point, point]))
                index = (index + 1) % len(lengths)
                remaining = lengths[index]
                continue
            if position >= distance:
                break
            step = min(remaining, distance - position)
            endpoint = (
                start[0] + dx * ((position + step) / distance),
                start[1] + dy * ((position + step) / distance),
            )
            if index % 2 == 0:
                if not current:
                    current.append(point)
                if endpoint != current[-1]:
                    current.append(endpoint)
            elif current:
                pieces.append(CapturedSubpath(current))
                current = []
            position += step
            remaining -= step
    if current:
        pieces.append(CapturedSubpath(current))
    if subpath.closed and pieces:
        first, last = pieces[0], pieces[-1]
        if first.points[0] == vertices[0] and last.points[-1] == vertices[-1]:
            if first is last:
                first.closed = True
            else:
                pieces[0] = CapturedSubpath([*last.points, *first.points[1:]])
                pieces.pop()
    return pieces


def rasterize_unclipped_line_normal(
    target_pixels: numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]],
    crop_x0: float,
    crop_y1: float,
    scale: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_width: float,
    rgba: tuple[int, int, int, int],
    pixel_box: tuple[int, int, int, int],
    *,
    return_source_alpha: bool = False,
    source_shape: UInt8Array | None = None,
) -> UInt8Array | None:
    """A butt-capped segment's 4x4 coverage over its pixel box, blended normally.

    fill_line's route for boxes of more than 64 pixels under no clip or a
    rectangular one. The cap extension is zero, and still multiplied
    through as it was, so a NaN or infinite segment samples as before.
    """
    x_delta = x1 - x0
    y_delta = y1 - y0
    segment_length_squared = x_delta * x_delta + y_delta * y_delta
    if segment_length_squared <= 1e-12:
        return None

    segment_length = segment_length_squared**0.5
    half = max(0.5 / scale, line_width * 0.5)
    cap_extension = 0.0
    ix0, iy0, ix1, iy1 = pixel_box
    samples = 4
    sample_total = samples * samples
    red, green, blue, source_alpha = rgba
    x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
    y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
    if x_coords.size == 0 or y_coords.size == 0:
        return None

    covered = numpy.zeros((y_coords.size, x_coords.size), dtype=numpy.int16)
    x_page = crop_x0 + (x_coords + 0.5 / samples) / scale
    y_page = crop_y1 - (y_coords + 0.5 / samples) / scale
    x_offset = x_page - x0
    y_offset = y_page - y0
    projection_base = numpy.add.outer(y_offset * y_delta, x_offset * x_delta)
    cross_base = numpy.add.outer(-y_offset * x_delta, x_offset * y_delta)
    sample_step = 1.0 / (samples * scale)
    cross_limit = half * segment_length
    projection_extension = cap_extension * segment_length
    mask = numpy.empty_like(projection_base, dtype=bool)
    condition = numpy.empty_like(projection_base, dtype=bool)
    for sy in range(samples):
        for sx in range(samples):
            projection_shift = sample_step * (sx * x_delta - sy * y_delta)
            numpy.greater_equal(projection_base, -projection_extension - projection_shift, out=mask)
            numpy.less_equal(
                projection_base,
                segment_length_squared + projection_extension - projection_shift,
                out=condition,
            )
            numpy.logical_and(mask, condition, out=mask)
            cross_shift = sample_step * (sx * y_delta + sy * x_delta)
            numpy.greater_equal(cross_base, -cross_limit - cross_shift, out=condition)
            numpy.logical_and(mask, condition, out=mask)
            numpy.less_equal(cross_base, cross_limit - cross_shift, out=condition)
            numpy.logical_and(mask, condition, out=mask)
            numpy.add(covered, mask, out=covered)

    if not numpy.any(covered):
        return None

    if source_shape is not None:
        source_shape[...] = numpy.rint(255 * covered / sample_total).astype(numpy.uint8)

    alpha = numpy.rint(source_alpha * covered / sample_total).astype(numpy.int16)
    alpha = numpy.clip(alpha, 0, 255)
    mask = alpha > 0
    if not numpy.any(mask):
        return None

    target = target_pixels[iy0:iy1, ix0:ix1, :]

    opaque = alpha >= 255
    if numpy.any(opaque):
        target[opaque, 0] = red
        target[opaque, 1] = green
        target[opaque, 2] = blue
        target[opaque, 3] = 255

    partial = mask & (~opaque)
    if numpy.any(partial):
        source_alpha_fraction = alpha[partial].astype(numpy.float32) / 255.0
        destination = target[partial].astype(numpy.float32)
        destination_alpha_fraction = destination[:, 3] / 255.0
        output_alpha = source_alpha_fraction + destination_alpha_fraction * (
            1.0 - source_alpha_fraction
        )
        safe_output_alpha = numpy.where(output_alpha > 0.0, output_alpha, 1.0)
        output_red = (
            red * source_alpha_fraction
            + destination[:, 0] * destination_alpha_fraction * (1.0 - source_alpha_fraction)
        ) / safe_output_alpha
        output_green = (
            green * source_alpha_fraction
            + destination[:, 1] * destination_alpha_fraction * (1.0 - source_alpha_fraction)
        ) / safe_output_alpha
        output_blue = (
            blue * source_alpha_fraction
            + destination[:, 2] * destination_alpha_fraction * (1.0 - source_alpha_fraction)
        ) / safe_output_alpha
        destination[:, 0] = numpy.clip(numpy.rint(output_red), 0, 255)
        destination[:, 1] = numpy.clip(numpy.rint(output_green), 0, 255)
        destination[:, 2] = numpy.clip(numpy.rint(output_blue), 0, 255)
        destination[:, 3] = numpy.clip(numpy.rint(output_alpha * 255.0), 0, 255)
        target[partial] = destination.astype(numpy.uint8)
    return alpha.astype(numpy.uint8) if return_source_alpha else None


def intersect_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """`intersect_bbox` for two known rectangles, with empty results as None."""
    box = intersect_bbox(a, b)
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def translate_rect(rect: Any, tx: float, ty: float) -> Any:
    rect_type = type(rect)
    if (rect_type is list or rect_type is tuple) and len(rect) == 4:
        return (
            float(rect[0]) + tx,
            float(rect[1]) + ty,
            float(rect[2]) + tx,
            float(rect[3]) + ty,
        )
    return rect


def fill_path_crossing_spans(
    crossings: list[tuple[float, int]],
    fill_rule: str,
) -> list[tuple[float, float]]:
    count = len(crossings)
    if not count:
        return []
    if count == 2:
        first_x = crossings[0][0]
        second_x = crossings[1][0]
        if second_x < first_x:
            first_x, second_x = second_x, first_x
        return [(first_x, second_x)] if second_x > first_x else []
    if fill_rule == "evenodd":
        xs = sorted(crossing[0] for crossing in crossings)
        return [(start, end) for start, end in zip(xs[0::2], xs[1::2], strict=False) if end > start]
    crossings.sort(key=lambda crossing: crossing[0])
    spans: list[tuple[float, float]] = []
    winding = 0
    previous_x: float | None = None
    index = 0
    while index < len(crossings):
        x = crossings[index][0]
        if previous_x is not None and winding != 0 and x > previous_x:
            spans.append((previous_x, x))
        delta = 0
        while index < len(crossings) and crossings[index][0] == x:
            delta += crossings[index][1]
            index += 1
        winding += delta
        previous_x = x
    return spans
