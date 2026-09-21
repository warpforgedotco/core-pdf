# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from typing import Any

import numpy

from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl._impl.runtime.array_views import UInt8Array, uint8_view

RASTER_KERNEL_MIN_PIXEL_AREA = 64
RASTER_CIRCLE_MIN_PIXEL_AREA = 16
RASTER_SAMPLE_OFFSETS = (0.125, 0.375, 0.625, 0.875)
INTERNAL_CROSSING_MASK_CELL_LIMIT = 1 << 24
internal_CIRCLE_VERTICES = tuple(
    (math.cos(index * math.tau / 32), math.sin(index * math.tau / 32)) for index in range(32)
)


def internal_circle_path(cx: float, cy: float, radius: float) -> CapturedPath:
    return CapturedPath(
        [
            CapturedSubpath(
                [(cx + radius * x, cy + radius * y) for x, y in internal_CIRCLE_VERTICES],
                closed=True,
            )
        ]
    )


def internal_dash_subpath(
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
    pixels: bytearray,
    width: int,
    crop_x0: float,
    crop_y1: float,
    scale: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_width: float,
    rgba: tuple[int, int, int, int],
    line_cap: int,
    pixel_box: tuple[int, int, int, int],
    *,
    target_pixels: numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]] | None = None,
    x_coords: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]] | None = None,
    y_coords: numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]] | None = None,
    return_source_alpha: bool = False,
    source_shape: UInt8Array | None = None,
) -> UInt8Array | None:
    x_delta = x1 - x0
    y_delta = y1 - y0
    segment_length_squared = x_delta * x_delta + y_delta * y_delta
    if segment_length_squared <= 1e-12:
        return None

    segment_length = segment_length_squared**0.5
    half = max(0.5 / scale, line_width * 0.5)
    half_squared = half * half
    cap_extension = half if line_cap == 2 else 0.0
    inv_segment_length_squared = 1.0 / segment_length_squared
    ix0, iy0, ix1, iy1 = pixel_box
    samples = 4
    sample_total = samples * samples
    red, green, blue, source_alpha = rgba
    if x_coords is None:
        x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
    if y_coords is None:
        y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
    if x_coords.size == 0 or y_coords.size == 0:
        return None

    covered = numpy.zeros((y_coords.size, x_coords.size), dtype=numpy.int16)
    if line_cap in {0, 2}:
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
                numpy.greater_equal(
                    projection_base, -projection_extension - projection_shift, out=mask
                )
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
    else:
        base_page_x = crop_x0 + (x_coords + 0.5 / samples) / scale
        base_page_y = crop_y1 - (y_coords + 0.5 / samples) / scale
        t_base = (
            numpy.add.outer(
                (base_page_y - y0) * y_delta,
                (base_page_x - x0) * x_delta,
            )
            * inv_segment_length_squared
        )
        sample_step = 1.0 / (samples * scale)
        for sy in range(samples):
            for sx in range(samples):
                t = (
                    t_base
                    + sample_step * (sx * x_delta - sy * y_delta) * inv_segment_length_squared
                )
                closest_t = numpy.clip(t, 0.0, 1.0)
                distance_x = base_page_x + sx * sample_step - (x0 + x_delta * closest_t)
                distance_y = base_page_y[:, None] - sy * sample_step - (y0 + y_delta * closest_t)
                inside = distance_x * distance_x + distance_y * distance_y <= half_squared
                covered += inside

    if not numpy.any(covered):
        return None

    if source_shape is not None:
        source_shape[...] = numpy.rint(255 * covered / sample_total).astype(numpy.uint8)

    alpha = numpy.rint(source_alpha * covered / sample_total).astype(numpy.int16)
    alpha = numpy.clip(alpha, 0, 255)
    mask = alpha > 0
    if not numpy.any(mask):
        return None

    if target_pixels is None:
        target_pixels = uint8_view(pixels).reshape(-1, width, 4)
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


def internal_group_offsets(
    counts: numpy.ndarray[Any, Any],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    total = int(counts.sum())
    if total == 0:
        empty = numpy.empty(0, numpy.int64)
        return empty, empty
    group = numpy.repeat(numpy.arange(counts.size, dtype=numpy.int64), counts)
    starts = numpy.zeros(counts.size, dtype=numpy.int64)
    numpy.cumsum(counts[:-1], out=starts[1:])
    return group, numpy.arange(total, dtype=numpy.int64) - starts[group]


def internal_signed_area_coverage(
    edges: numpy.ndarray[Any, Any],
    width: int,
    height: int,
) -> numpy.ndarray[Any, Any]:
    if height <= 0 or width <= 0 or edges.size == 0:
        return numpy.zeros((max(height, 0), max(width, 0)), numpy.float64)
    start_y = edges[:, 1]
    end_y = edges[:, 3]
    sloped = start_y != end_y
    if not sloped.any():
        return numpy.zeros((height, width), numpy.float64)
    start_x = edges[sloped, 0]
    end_x = edges[sloped, 2]
    start_y = start_y[sloped]
    end_y = end_y[sloped]

    downward = end_y > start_y
    direction = numpy.where(downward, 1.0, -1.0)
    top_x = numpy.where(downward, start_x, end_x)
    top_y = numpy.where(downward, start_y, end_y)
    bottom_x = numpy.where(downward, end_x, start_x)
    bottom_y = numpy.where(downward, end_y, start_y)
    x_per_y = (bottom_x - top_x) / (bottom_y - top_y)

    first_row = numpy.maximum(0.0, numpy.floor(top_y)).astype(numpy.int64)
    last_row = numpy.minimum(float(height), numpy.ceil(bottom_y)).astype(numpy.int64)
    edge_index, row_offset = internal_group_offsets(numpy.maximum(last_row - first_row, 0))
    if edge_index.size == 0:
        return numpy.zeros((height, width), numpy.float64)

    row = first_row[edge_index] + row_offset
    row_top = numpy.maximum(top_y[edge_index], row)
    row_bottom = numpy.minimum(bottom_y[edge_index], row + 1.0)
    row_height = row_bottom - row_top
    inside = row_height > 0.0
    if not inside.all():
        edge_index = edge_index[inside]
        row = row[inside]
        row_top = row_top[inside]
        row_bottom = row_bottom[inside]
        row_height = row_height[inside]
        if edge_index.size == 0:
            return numpy.zeros((height, width), numpy.float64)

    slope = x_per_y[edge_index]
    origin_x = top_x[edge_index]
    origin_y = top_y[edge_index]
    entry_x = origin_x + (row_top - origin_y) * slope
    exit_x = origin_x + (row_bottom - origin_y) * slope
    left_x = numpy.minimum(entry_x, exit_x)
    right_x = numpy.maximum(entry_x, exit_x)

    walk_left = numpy.clip(left_x, -1.0, width + 1.0)
    walk_right = numpy.clip(right_x, -1.0, width + 1.0)
    column_count = (numpy.floor(walk_right) - numpy.floor(walk_left) + 1.0).astype(numpy.int64)
    piece_index, column_offset = internal_group_offsets(column_count)
    column_start = numpy.floor(walk_left)[piece_index] + column_offset
    fragment_left = numpy.where(column_offset == 0, left_x[piece_index], column_start)
    fragment_right = numpy.where(
        column_offset == column_count[piece_index] - 1,
        right_x[piece_index],
        column_start + 1.0,
    )
    piece_span = (right_x - left_x)[piece_index]
    vertical = piece_span <= 0.0
    share = numpy.where(
        vertical,
        1.0,
        (fragment_right - fragment_left) / numpy.where(vertical, 1.0, piece_span),
    )
    fragment_height = row_height[piece_index] * share
    keep = vertical | (fragment_right > fragment_left)
    if not keep.all():
        piece_index = piece_index[keep]
        fragment_left = fragment_left[keep]
        fragment_right = fragment_right[keep]
        fragment_height = fragment_height[keep]
        if piece_index.size == 0:
            return numpy.zeros((height, width), numpy.float64)

    midpoint = (fragment_left + fragment_right) * 0.5
    column = numpy.clip(numpy.floor(midpoint), 0, width).astype(numpy.int64)
    offset_in_cell = numpy.clip(midpoint - column, 0.0, 1.0)
    signed_height = direction[edge_index][piece_index] * fragment_height

    stride = width + 2
    flat_row = row[piece_index] * stride
    accumulator = numpy.bincount(
        numpy.concatenate([flat_row + column, flat_row + column + 1]),
        weights=numpy.concatenate(
            [signed_height * (1.0 - offset_in_cell), signed_height * offset_in_cell]
        ),
        minlength=height * stride,
    )[: height * stride].reshape(height, stride)
    return numpy.minimum(numpy.abs(numpy.cumsum(accumulator, axis=1)[:, :width]), 1.0)


def internal_intersect_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def internal_translate_rect(rect: Any, tx: float, ty: float) -> Any:
    rect_type = type(rect)
    if (rect_type is list or rect_type is tuple) and len(rect) == 4:
        return (
            float(rect[0]) + tx,
            float(rect[1]) + ty,
            float(rect[2]) + tx,
            float(rect[3]) + ty,
        )
    return rect


def internal_fill_path_sample_crossings(
    edge_segments: list[tuple[float, float, float, float, float, float]],
    page_y: float,
) -> list[tuple[float, int]]:
    crossings: list[tuple[float, int]] = []
    for ex0, ey0, ex1, ey1, low, high in edge_segments:
        if not (low <= page_y < high):
            continue
        t = (page_y - ey0) / (ey1 - ey0)
        x_intersection = ex0 + t * (ex1 - ex0)
        crossings.append((x_intersection, 1 if ey1 > ey0 else -1))
    return crossings


def internal_fill_path_sample_crossings_numpy(
    edge_segments: numpy.ndarray[Any, Any],
    page_ys: numpy.ndarray[Any, Any],
) -> list[list[tuple[float, int]]]:
    row_count = len(page_ys)
    if row_count == 0:
        return []
    edge_count = len(edge_segments)
    if edge_count == 0:
        return [[] for _ in range(row_count)]
    if row_count * edge_count > INTERNAL_CROSSING_MASK_CELL_LIMIT:
        return [
            internal_fill_path_sample_crossings_row(edge_segments, float(page_y))
            for page_y in page_ys
        ]

    ys = page_ys.reshape(-1, 1)
    active = (edge_segments[:, 4].reshape(1, -1) <= ys) & (ys < edge_segments[:, 5].reshape(1, -1))
    row_indexes, edge_indexes = numpy.nonzero(active)
    if row_indexes.size == 0:
        return [[] for _ in range(row_count)]

    edge_x0 = edge_segments[edge_indexes, 0]
    edge_y0 = edge_segments[edge_indexes, 1]
    delta_y = edge_segments[edge_indexes, 3] - edge_y0
    intersections = edge_x0 + (
        (page_ys[row_indexes] - edge_y0) / delta_y * (edge_segments[edge_indexes, 2] - edge_x0)
    )
    directions = numpy.where(delta_y > 0.0, 1, -1)

    xs = intersections.tolist()
    ds = directions.tolist()
    counts = numpy.bincount(row_indexes, minlength=row_count).tolist()
    crossings_rows: list[list[tuple[float, int]]] = []
    start = 0
    for count in counts:
        if count:
            stop = start + count
            crossings_rows.append(list(zip(xs[start:stop], ds[start:stop], strict=True)))
            start = stop
        else:
            crossings_rows.append([])
    return crossings_rows


def internal_fill_path_sample_crossings_row(
    edge_segments: numpy.ndarray[Any, Any],
    page_y: float,
) -> list[tuple[float, int]]:
    active = edge_segments[(edge_segments[:, 4] <= page_y) & (page_y < edge_segments[:, 5])]
    if not len(active):
        return []
    delta_y = active[:, 3] - active[:, 1]
    intersections = active[:, 0] + (
        (page_y - active[:, 1]) / delta_y * (active[:, 2] - active[:, 0])
    )
    directions = numpy.where(delta_y > 0.0, 1, -1)
    return list(zip(intersections.tolist(), directions.tolist(), strict=True))


def internal_fill_path_crossing_spans(
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
