# SPDX-License-Identifier: AGPL-3.0-only
"""Pure and allocation-bounded raster kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import numpy

from core_pdf.impl.model.geometry import RectBox
from core_pdf.impl.render.display import (
    DisplayItem,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.render.raster_image import RasterImage
from core_pdf.impl.runtime.array_views import uint8_view
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_08_graphics.device_profiles import (
    cmyk_floats_to_srgb,
)

# ===== raster_kernel =====


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
) -> None:
    """Rasterize one antialiased line into an unclipped normal RGBA bitmap.

    This intentionally mirrors the renderer's general diagonal-line path.  Keeping the
    kernel free of renderer state makes it suitable for isolated performance testing
    while preserving
    the existing renderer for clipped, dashed, and non-normal blend-mode paths.
    """
    x_delta = x1 - x0
    y_delta = y1 - y0
    segment_length_squared = x_delta * x_delta + y_delta * y_delta
    if segment_length_squared <= 1e-12:
        return

    segment_length = segment_length_squared**0.5
    half = max(0.5 / scale, line_width * 0.5)
    half_squared = half * half
    cap_extension = half if line_cap == 2 else 0.0
    inv_segment_length_squared = 1.0 / segment_length_squared
    extension_t = cap_extension / segment_length
    ix0, iy0, ix1, iy1 = pixel_box
    samples = 4
    sample_total = samples * samples
    red, green, blue, source_alpha = rgba
    if x_coords is None:
        x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
    if y_coords is None:
        y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
    if x_coords.size == 0 or y_coords.size == 0:
        return

    covered = numpy.zeros((y_coords.size, x_coords.size), dtype=numpy.int16)
    if line_cap == 0:
        x_page = crop_x0 + (x_coords + 0.5 / samples) / scale
        y_page = crop_y1 - (y_coords + 0.5 / samples) / scale
        x_offset = x_page - x0
        y_offset = y_page - y0
        projection_base = numpy.add.outer(y_offset * y_delta, x_offset * x_delta)
        cross_base = numpy.add.outer(-y_offset * x_delta, x_offset * y_delta)
        sample_step = 1.0 / (samples * scale)
        cross_limit = half * segment_length
        mask = numpy.empty_like(projection_base, dtype=bool)
        condition = numpy.empty_like(projection_base, dtype=bool)
        for sy in range(samples):
            for sx in range(samples):
                projection_shift = sample_step * (sx * x_delta - sy * y_delta)
                numpy.greater_equal(projection_base, -projection_shift, out=mask)
                numpy.less_equal(
                    projection_base,
                    segment_length_squared - projection_shift,
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
                if line_cap == 2:
                    inside &= (t >= -extension_t) & (t <= 1.0 + extension_t)
                covered += inside

    if not numpy.any(covered):
        return

    alpha = numpy.rint(source_alpha * covered / sample_total).astype(numpy.int16)
    alpha = numpy.clip(alpha, 0, 255)
    mask = alpha > 0
    if not numpy.any(mask):
        return

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


# ===== raster =====


RASTER_KERNEL_MIN_PIXEL_AREA = 64
RASTER_COORDINATE_CACHE_MAX_ENTRIES = 256
# NumPy's coordinate mask remains cheaper than Python pixel loops for modest
# circles, while very small caps are faster to paint directly.
RASTER_CIRCLE_MIN_PIXEL_AREA = 16
RASTER_NUMPY_SPAN_MIN_PIXELS = 32
AFFINE_BLIT_SCRATCH_BYTES = 1 << 20
RASTER_SAMPLE_OFFSETS = (0.125, 0.375, 0.625, 0.875)


def internal_blend_normal_solid_span_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    row: int,
    start: int,
    end: int,
    rgba: tuple[int, int, int, int],
) -> None:
    """Blend one contiguous RGBA span through a zero-copy page view.

    The target is an existing ``(height, width, 4)`` view over the active
    raster buffer.  Only the temporary floating-point working arrays are
    allocated; the result is written directly back into the page view.
    """
    sa = rgba[3]
    if sa <= 0 or end <= start:
        return
    internal_blend_normal_solid_array_numpy(target[row, start:end], rgba)


def internal_blend_normal_solid_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
) -> None:
    """Blend a solid normal-alpha source into an existing RGBA view.

    Indexes ``target`` directly (``target[..., i]`` / ``target[...]``) rather
    than ``target.reshape(-1, 4)``: callers pass both flat ``(n, 4)`` spans
    and 2D ``(rows, cols, 4)`` boxes (e.g. ``fill_rect``'s whole-rectangle
    fast path), and a box slice narrower than the full row stride is not
    C-contiguous, so reshaping it silently returns a disconnected copy and
    every write below would be lost -- confirmed reproducible: a semi-
    transparent, non-full-width, multi-row fill through that path painted
    nothing.
    """
    sr, sg, sb, sa = rgba
    if sa <= 0 or target.size == 0:
        return
    if sa >= 255:
        target[...] = rgba
        return
    if not numpy.any(target[..., 3]):
        target[..., :3] = (sr, sg, sb)
        target[..., 3] = sa
        return
    source_alpha = sa / 255.0
    inverse_source_alpha = 1.0 - source_alpha
    if numpy.all(target[..., 3] == 255):
        destination_rgb = target[..., :3].astype(numpy.float32)
        destination_rgb[..., 0] = numpy.rint(
            sr * source_alpha + destination_rgb[..., 0] * inverse_source_alpha
        )
        destination_rgb[..., 1] = numpy.rint(
            sg * source_alpha + destination_rgb[..., 1] * inverse_source_alpha
        )
        destination_rgb[..., 2] = numpy.rint(
            sb * source_alpha + destination_rgb[..., 2] * inverse_source_alpha
        )
        target[..., :3] = numpy.clip(destination_rgb, 0.0, 255.0).astype(numpy.uint8)
        return

    destination_float = target.astype(numpy.float32)
    destination_alpha = destination_float[..., 3] / 255.0
    output_alpha = source_alpha + destination_alpha * inverse_source_alpha
    destination_rgb = destination_float[..., :3]
    destination_rgb[..., 0] = (
        sr * source_alpha + destination_rgb[..., 0] * destination_alpha * inverse_source_alpha
    ) / output_alpha
    destination_rgb[..., 1] = (
        sg * source_alpha + destination_rgb[..., 1] * destination_alpha * inverse_source_alpha
    ) / output_alpha
    destination_rgb[..., 2] = (
        sb * source_alpha + destination_rgb[..., 2] * destination_alpha * inverse_source_alpha
    ) / output_alpha
    destination_float[..., 3] = numpy.rint(output_alpha * 255.0)
    numpy.rint(destination_float, out=destination_float)
    numpy.clip(destination_float, 0.0, 255.0, out=destination_float)
    target[...] = destination_float.astype(numpy.uint8)


def internal_blend_solid_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> None:
    """Blend a solid source into an RGBA view, replicating ``blend_px``'s
    per-pixel math (Multiply/Screen premultiply, generic alpha compositing)
    across every destination pixel in one pass.

    Callers fold any transparency-group alpha into ``rgba``'s alpha before
    calling -- ``blend_px`` reads that scale from state that is invariant
    across the whole span, so it is computed once here instead of per pixel.
    Uses float64 throughout (not float32, unlike the Normal-blend fast paths
    above) so every intermediate matches ``blend_px``'s plain-Python-float
    arithmetic bit for bit; the golden-raster digests depend on it.

    Indexes ``target``'s last axis directly (``target[..., 0]`` etc.) rather
    than ``target.reshape(-1, 4)``: callers pass both flat ``(n, 4)`` spans
    and 2D ``(rows, cols, 4)`` boxes, and a box slice narrower than the full
    row stride is not C-contiguous, so reshaping it silently returns a
    disconnected copy and every write below would be lost.
    """
    sr, sg, sb, sa = rgba
    if sa <= 0 or target.size == 0:
        return
    mode = blend_mode.lower() if isinstance(blend_mode, str) else None
    if sa >= 255 and mode is None:
        target[..., 0] = sr
        target[..., 1] = sg
        target[..., 2] = sb
        target[..., 3] = 255
        return
    dr = target[..., 0].astype(numpy.float64)
    dg = target[..., 1].astype(numpy.float64)
    db = target[..., 2].astype(numpy.float64)
    da = target[..., 3].astype(numpy.float64)
    src_a = sa / 255.0
    one_minus_src_a = 1.0 - src_a
    dst_a = da / 255.0
    src_r: numpy.ndarray | float = sr / 255.0
    src_g: numpy.ndarray | float = sg / 255.0
    src_b: numpy.ndarray | float = sb / 255.0
    if mode == "multiply":
        src_r = src_r * (dr / 255.0)
        src_g = src_g * (dg / 255.0)
        src_b = src_b * (db / 255.0)
    elif mode == "screen":
        src_r = 1.0 - (1.0 - src_r) * (1.0 - dr / 255.0)
        src_g = 1.0 - (1.0 - src_g) * (1.0 - dg / 255.0)
        src_b = 1.0 - (1.0 - src_b) * (1.0 - db / 255.0)
    out_a = src_a + dst_a * one_minus_src_a
    safe_out_a = numpy.where(out_a > 0.0, out_a, 1.0)
    out_r = numpy.round(((src_r * 255.0) * src_a + dr * dst_a * one_minus_src_a) / safe_out_a)
    out_g = numpy.round(((src_g * 255.0) * src_a + dg * dst_a * one_minus_src_a) / safe_out_a)
    out_b = numpy.round(((src_b * 255.0) * src_a + db * dst_a * one_minus_src_a) / safe_out_a)
    out_a_i = numpy.round(out_a * 255.0)
    transparent = out_a <= 0.0
    out_r = numpy.where(transparent, 0.0, out_r)
    out_g = numpy.where(transparent, 0.0, out_g)
    out_b = numpy.where(transparent, 0.0, out_b)
    out_a_i = numpy.where(transparent, 0.0, out_a_i)
    target[..., 0] = numpy.clip(out_r, 0.0, 255.0).astype(numpy.uint8)
    target[..., 1] = numpy.clip(out_g, 0.0, 255.0).astype(numpy.uint8)
    target[..., 2] = numpy.clip(out_b, 0.0, 255.0).astype(numpy.uint8)
    target[..., 3] = numpy.clip(out_a_i, 0.0, 255.0).astype(numpy.uint8)


def internal_group_offsets(
    counts: numpy.ndarray[Any, Any],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    """Expand per-group counts into (group index, index within group) pairs."""
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
    """Exact analytic coverage for a nonzero-wound polygon, in one vectorized pass.

    Accumulates signed area per pixel the way font-rs and stb_truetype do, then
    prefix-sums along x. Two properties matter here:

    * Cost is O(sum of edge extents), not O(rows x edges). The sampling
      rasterizer this replaces tested every edge against every sample row, so a
      30-pixel glyph with 84 edges did ~2,000 intersections; here each edge
      touches only the cells it actually crosses.
    * Coverage is exact rather than quantized to the 17 levels a 4x4 sample grid
      can express, so edges are smoother, not just cheaper.

    ``edges`` is (n, 4) of x0, y0, x1, y1 in device pixels with y increasing
    downward, already translated so the box origin is (0, 0). Returns an
    (height, width) float array in [0, 1]. Nonzero winding only -- the
    abs-and-clamp at the end is what makes it nonzero, and even-odd needs the
    span-based path.
    """
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

    # Split each row piece again at column boundaries so every fragment lies in
    # one cell; the midpoint split below is exact only within a single cell.
    #
    # The walk is bounded to one cell either side of the box. A fill clipped to
    # a small box still arrives with the whole path's edges, so a near-
    # horizontal edge can cross millions of columns that the box never shows --
    # one fragment each, gigabytes of them. Everything left of the box already
    # lands on column 0 and everything right of it past the visible slice, so
    # collapsing those runs into the first and last fragment changes nothing.
    # Those two keep the piece's original extents, which is what holds the
    # coverage weights identical to the unbounded walk (and bit-identical when
    # the piece lies inside the box, where the clamp does not bite).
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


def internal_box_downsample(
    samples: numpy.ndarray[Any, Any],
    source_width: int,
    source_height: int,
    channels: int,
    target_width: int,
    target_height: int,
) -> tuple[numpy.ndarray[Any, Any], int, int]:
    """Area-average an image down to about (target_width, target_height).

    Sampling a shrunk image with nearest-neighbour throws away most of it: a
    2544x3296 scan placed on a 612x792 page keeps roughly one source pixel in
    eighteen, so the thin rules and letter stems of a scanned form fall between
    samples and the page renders visibly faint. Averaging the block each output
    pixel covers keeps that ink.

    Bin edges are ``i * source // target`` so the blocks tile the source exactly
    even when the ratio is not integral; ``add.reduceat`` then sums each block in
    one pass per axis. Returns the reduced samples with their new dimensions, so
    the caller's existing nearest-neighbour map resamples an already-averaged
    image.
    """
    if target_width <= 0 or target_height <= 0:
        return samples, source_width, source_height
    if source_width <= target_width and source_height <= target_height:
        return samples, source_width, source_height
    target_width = min(target_width, source_width)
    target_height = min(target_height, source_height)
    grid = samples.reshape(source_height, source_width, channels)
    row_edges = (numpy.arange(target_height + 1, dtype=numpy.int64) * source_height) // (
        target_height
    )
    column_edges = (numpy.arange(target_width + 1, dtype=numpy.int64) * source_width) // (
        target_width
    )
    totals = numpy.add.reduceat(grid.astype(numpy.uint32), row_edges[:-1], axis=0)
    totals = numpy.add.reduceat(totals, column_edges[:-1], axis=1)
    counts = numpy.diff(row_edges)[:, None, None] * numpy.diff(column_edges)[None, :, None]
    reduced = (totals // numpy.maximum(counts, 1)).astype(numpy.uint8)
    return reduced.reshape(-1), target_width, target_height


def internal_blend_normal_alpha_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
    alpha: numpy.ndarray[Any, Any],
) -> None:
    """Blend a normal source with one coverage alpha per target pixel.

    Indexes ``target`` directly rather than ``target.reshape(-1, 4)`` -- see
    ``internal_blend_normal_solid_array_numpy`` for why a reshaped box slice
    can silently discard every write below.
    """
    if target.size == 0 or not numpy.any(alpha):
        return
    source_alpha = numpy.minimum(alpha, rgba[3]).astype(numpy.float32) / 255.0
    destination_float = target.astype(numpy.float32)
    destination_alpha = destination_float[..., 3] / 255.0
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    safe_output_alpha = numpy.where(output_alpha > 0.0, output_alpha, 1.0)
    inverse_source_alpha = 1.0 - source_alpha
    dst_weight = destination_alpha * inverse_source_alpha
    destination_float[..., 0] = (
        rgba[0] * source_alpha + destination_float[..., 0] * dst_weight
    ) / safe_output_alpha
    destination_float[..., 1] = (
        rgba[1] * source_alpha + destination_float[..., 1] * dst_weight
    ) / safe_output_alpha
    destination_float[..., 2] = (
        rgba[2] * source_alpha + destination_float[..., 2] * dst_weight
    ) / safe_output_alpha
    destination_float[..., 3] = output_alpha * 255.0
    numpy.rint(destination_float, out=destination_float)
    numpy.clip(destination_float, 0.0, 255.0, out=destination_float)
    # Only touch pixels the source actually covers. Blending zero coverage is
    # meant to be a no-op, but the arithmetic above drives RGB to zero when the
    # destination is fully transparent, which discards colour a later blend
    # would need -- and it is what stopped a whole box being blended in one call.
    numpy.copyto(target, destination_float.astype(numpy.uint8), where=(alpha > 0)[..., None])


def internal_composite_normal_group_numpy(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_alpha_scale: float,
    target_alpha_scale: float = 1.0,
) -> None:
    """Composite a straight-alpha normal group through two RGBA views.

    Indexes both views directly (``[..., i]``) instead of reshaping to
    ``(-1, 4)`` -- see ``internal_blend_normal_solid_array_numpy`` for why a
    reshaped non-contiguous ``destination`` box slice silently discards
    writes. ``source`` is read-only here, so reshaping it would be safe on
    its own, but two of the early-return branches below wrote into a
    reshaped *destination* view and returned without ever assigning back
    into the real ``destination`` array -- kept both views in the same
    (unflattened) shape so every write, including boolean-masked ones,
    targets `destination` (or a scratch copy explicitly written back to it)
    rather than a same-shaped-but-disconnected copy.
    """
    if destination.size == 0 or source_alpha_scale <= 0.0:
        return
    source_alpha_u8 = source[..., 3]
    if not numpy.any(source_alpha_u8):
        return
    if (
        source_alpha_scale == 1.0
        and target_alpha_scale == 1.0
        and numpy.all(source_alpha_u8 == 255)
    ):
        destination[...] = source
        return
    effective_alpha = numpy.rint(source_alpha_u8 * source_alpha_scale)
    if target_alpha_scale != 1.0:
        effective_alpha = numpy.rint(effective_alpha * target_alpha_scale)
    effective_alpha = numpy.clip(effective_alpha, 0.0, 255.0)
    if (
        source_alpha_scale <= 1.0
        and target_alpha_scale <= 1.0
        and numpy.all(destination[..., 3] == 255)
    ):
        source_alpha = effective_alpha / 255.0
        source_rgb = source[..., :3].astype(numpy.float32)
        destination_rgb = destination[..., :3].astype(numpy.float32)
        destination_rgb[...] = numpy.rint(
            source_rgb * source_alpha[..., None] + destination_rgb * (1.0 - source_alpha[..., None])
        )
        destination[..., :3] = numpy.clip(destination_rgb, 0.0, 255.0).astype(numpy.uint8)
        return
    if not numpy.any(destination[..., 3]):
        visible = effective_alpha > 0.0
        if numpy.any(visible):
            destination[..., :3][visible] = source[..., :3][visible]
            destination[..., 3][visible] = effective_alpha[visible]
        return
    source_float = source.astype(numpy.float32)
    destination_float = destination.astype(numpy.float32)
    source_alpha = numpy.rint(source_float[..., 3] * source_alpha_scale)
    if target_alpha_scale != 1.0:
        source_alpha = numpy.rint(source_alpha * target_alpha_scale)
    source_alpha = source_alpha / 255.0
    visible = source_alpha > 0.0
    if not numpy.any(visible):
        return
    source_a = source_alpha[visible]
    destination_a = destination_float[..., 3][visible] / 255.0
    output_alpha = source_a + destination_a * (1.0 - source_a)
    destination_rgb = destination_float[..., :3][visible]
    output_rgb = (
        source_float[..., :3][visible] * source_a[:, None]
        + destination_rgb * destination_a[:, None] * (1.0 - source_a)[:, None]
    ) / output_alpha[:, None]
    destination_float[..., :3][visible] = numpy.rint(output_rgb)
    destination_float[..., 3][visible] = numpy.rint(output_alpha * 255.0)
    numpy.rint(destination_float, out=destination_float)
    numpy.clip(destination_float, 0.0, 255.0, out=destination_float)
    destination[...] = destination_float.astype(numpy.uint8)


def internal_blend_normal_masked_array_numpy(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_rgb: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_mask: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    alpha: int,
) -> None:
    """Blend masked RGB samples into an RGBA destination view.

    ``source_rgb``/``source_mask`` must already be shaped to match
    ``destination``'s leading (non-channel) dimensions -- callers no longer
    flatten them before calling. ``destination`` is indexed directly
    (``[..., i]``) rather than reshaped to ``(-1, 4)``: see
    ``internal_blend_normal_solid_array_numpy`` for why reshaping a
    non-contiguous box slice silently discards writes. ``source_rgb`` and
    ``source_mask`` are read-only here, so their own shape is otherwise
    unconstrained as long as it matches.
    """
    if destination.size == 0:
        return
    if alpha <= 0 or not numpy.any(source_mask):
        return
    if alpha == 255 and numpy.all(source_mask == 255):
        destination[..., :3] = source_rgb
        destination[..., 3] = 255
        return
    if alpha == 255:
        partial_mask = (source_mask > 0) & (source_mask < 255)
        if not numpy.any(partial_mask):
            opaque = source_mask == 255
            destination[..., :3][opaque] = source_rgb[opaque]
            destination[..., 3][opaque] = 255
            return
    source_alpha = source_mask.astype(numpy.float32)
    if alpha != 255:
        source_alpha = numpy.rint(source_alpha * (alpha / 255.0))
    if not numpy.any(destination[..., 3]):
        visible = source_alpha > 0.0
        if numpy.any(visible):
            destination[..., :3][visible] = source_rgb[visible]
            destination[..., 3][visible] = source_alpha[visible]
        return
    if numpy.all(destination[..., 3] == 255):
        source_alpha = source_alpha / 255.0
        destination_rgb = destination[..., :3].astype(numpy.float32)
        destination_rgb[...] = numpy.rint(
            source_rgb * source_alpha[..., None] + destination_rgb * (1.0 - source_alpha[..., None])
        )
        destination[..., :3] = numpy.clip(destination_rgb, 0.0, 255.0).astype(numpy.uint8)
        return
    destination_float = destination.astype(numpy.float32)
    opaque = source_alpha >= 255.0
    if numpy.any(opaque):
        destination_float[..., :3][opaque] = source_rgb[opaque]
        destination_float[..., 3][opaque] = 255.0
    partial = (~opaque) & (source_alpha > 0.0)
    if numpy.any(partial):
        source_a = source_alpha[partial] / 255.0
        destination_a = destination_float[..., 3][partial] / 255.0
        output_a = source_a + destination_a * (1.0 - source_a)
        destination_rgb = destination_float[..., :3][partial]
        output_rgb = (
            source_rgb[partial] * source_a[:, None]
            + destination_rgb * destination_a[:, None] * (1.0 - source_a)[:, None]
        ) / output_a[:, None]
        destination_float[..., :3][partial] = numpy.rint(output_rgb)
        destination_float[..., 3][partial] = numpy.rint(output_a * 255.0)
    numpy.rint(destination_float, out=destination_float)
    numpy.clip(destination_float, 0, 255, out=destination_float)
    destination[...] = destination_float.astype(numpy.uint8)


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


# Pure helpers lifted out of RenderedPage.rasterize so they can be tested
# directly. They capture nothing from the rasterizer's frame.
def internal_cached_raster_coordinates(
    cache: dict[tuple[int, int], numpy.ndarray[Any, Any]],
    start: int,
    stop: int,
) -> numpy.ndarray[Any, Any]:
    key = (start, stop)
    coordinates = cache.get(key)
    if coordinates is None:
        coordinates = numpy.arange(start, stop, dtype=numpy.float64)
        if len(cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
            cache[key] = coordinates
    return coordinates


def internal_blit_reshaped_channels(
    target_region: numpy.ndarray[Any, Any],
    sampled: numpy.ndarray[Any, Any],
    valid: numpy.ndarray[Any, Any],
    comps: int,
) -> None:
    """Copy gray/RGB(A) channels from a pre-reshaped source into ``target_region``."""
    if comps == 1:
        target_region[valid, 0:3] = sampled[valid, 0][:, None]
    else:
        target_region[valid, 0:3] = sampled[valid, :3]
    target_region[..., 3][valid] = 255


def internal_blit_indexed_channels(
    target_region: numpy.ndarray[Any, Any],
    source_bytes: numpy.ndarray[Any, Any],
    safe_index: numpy.ndarray[Any, Any],
    valid: numpy.ndarray[Any, Any],
    comps: int,
) -> None:
    """Copy gray/RGB(A) channels looked up via ``safe_index`` into ``target_region``."""
    if comps == 1:
        gray_samples = source_bytes[safe_index]
        target_region[:, :, 0][valid] = gray_samples[valid]
        target_region[:, :, 1][valid] = gray_samples[valid]
        target_region[:, :, 2][valid] = gray_samples[valid]
    else:
        target_region[:, :, 0][valid] = source_bytes[safe_index][valid]
        target_region[:, :, 1][valid] = source_bytes[safe_index + 1][valid]
        target_region[:, :, 2][valid] = source_bytes[safe_index + 2][valid]
    target_region[:, :, 3][valid] = 255


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
    if type(rect) is RectBox:
        return RectBox(
            rect.x0 + tx,
            rect.y0 + ty,
            rect.x1 + tx,
            rect.y1 + ty,
            seqno=rect.seqno,
            fill=rect.fill,
            fill_opacity=rect.fill_opacity,
        )
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


# Above this many (row, edge) pairs the activity mask below costs more memory
# than the per-row loop costs time, so the loop stays the fallback.
INTERNAL_CROSSING_MASK_CELL_LIMIT = 1 << 24


def internal_fill_path_sample_crossings_numpy(
    edge_segments: numpy.ndarray[Any, Any],
    page_ys: numpy.ndarray[Any, Any],
) -> list[list[tuple[float, int]]]:
    """Intersect every edge with every scanline, one row per returned list.

    Solves all rows in a single pass rather than looping in Python: this runs
    once per scanline per filled path, and at a handful of numpy calls on a
    few-element array per row it was pure call overhead.

    The per-element arithmetic is spelled exactly as the row-at-a-time form
    below it -- ``ex0 + ((y - ey0) / dy * (ex1 - ex0))`` -- because these are
    elementwise IEEE double operations that numpy does not reassociate, so
    batching them cannot move a bit. ``nonzero`` walks the mask in C order, so
    each row keeps its edges in edge_segments order, matching the loop.
    """
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


internal_first_item = itemgetter(0)


def internal_fill_path_crossing_spans(
    crossings: list[tuple[float, int]],
    fill_rule: str,
) -> list[tuple[float, float]]:
    count = len(crossings)
    if not count:
        return []
    if count == 2:
        # 44% of calls on a text-heavy page: a single edge pair. Both rules
        # collapse to one comparison -- even-odd pairs the two sorted crossings,
        # and nonzero's sweep opens a span iff the winding after the first is
        # non-zero, which it always is because directions are only ever +/-1.
        first_x = crossings[0][0]
        second_x = crossings[1][0]
        if second_x < first_x:
            first_x, second_x = second_x, first_x
        return [(first_x, second_x)] if second_x > first_x else []
    if fill_rule == "evenodd":
        xs = sorted(map(internal_first_item, crossings))
        return [(start, end) for start, end in zip(xs[0::2], xs[1::2], strict=False) if end > start]
    crossings.sort(key=internal_first_item)
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


def internal_color_component(value: Any, default: int = 0) -> int:
    if type(value) is bool:
        return default
    try:
        return max(0, min(255, int(round(float(value) * 255.0))))
    except (TypeError, ValueError):
        return default


def internal_clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def internal_soft_mask_alpha_at(
    mask: numpy.ndarray[Any, Any] | None,
    u: float,
    v: float,
) -> int:
    if mask is None:
        return 255
    if mask.ndim != 2 or mask.size == 0:
        return 255
    mask_height, mask_width = mask.shape
    src_x = min(mask_width - 1, max(0, int(u * mask_width)))
    src_y = min(mask_height - 1, max(0, int((1.0 - v) * mask_height)))
    return int(mask[src_y, src_x])


# Colour and soft-mask helpers lifted out of RenderedPage.rasterize.
def internal_color_rgba(color: Any, opacity: Any) -> tuple[int, int, int, int]:
    alpha = 255
    if type(opacity) in {int, float}:
        alpha = internal_color_component(opacity, 255)
    if isinstance(color, (list, tuple)) and color:
        if len(color) == 1:
            gray = internal_color_component(color[0])
            return gray, gray, gray, alpha
        if len(color) == 4:
            # DeviceCMYK. The component count is the colour space here --
            # normalize_colors clamps and preserves arity, and folds in no alpha
            # -- so four components is CMYK and nothing else. Without this the
            # tuple fell through to the RGB branch below, which read the first
            # three components as red/green/blue: `1 1 1 1 k` (rich black)
            # painted white and `0 0 0 0 k` (white) painted black.
            try:
                cyan, magenta, yellow, black = (internal_clamp01(float(c)) for c in color)
            except (TypeError, ValueError):
                return 0, 0, 0, alpha
            red, green, blue = cmyk_floats_to_srgb(cyan, magenta, yellow, black)
            return red, green, blue, alpha
        rgb = [internal_color_component(c) for c in color[:3]]
        while len(rgb) < 3:
            rgb.append(rgb[-1] if rgb else 0)
        return rgb[0], rgb[1], rgb[2], alpha
    return 0, 0, 0, alpha


def internal_scale_rgba_alpha(
    rgba: tuple[int, int, int, int],
    alpha_scale: Any,
) -> tuple[int, int, int, int]:
    """Scale a colour's alpha by a soft-mask factor, clamped to a byte."""
    return (
        rgba[0],
        rgba[1],
        rgba[2],
        max(0, min(255, int(round(rgba[3] * float(alpha_scale))))),
    )


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


def internal_make_page_geometry(
    crop_x0: float, crop_y1: float, scale: float, width: int, height: int
) -> tuple[
    Callable[[float, float, float, float], tuple[int, int, int, int] | None],
    Callable[[float, float], tuple[int, int] | None],
]:
    """Build the page-to-pixel converters, closed over a fixed page geometry.

    These run ~1.8M times over the corpus. Binding the geometry into a closure
    once keeps every read a `LOAD_DEREF`; holding them as instance attributes
    would add an attribute load per access on the hottest path in the rasterizer.
    """

    def page_box_to_pixels(
        x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, int, int, int] | None:
        ix0 = max(0, min(width, math.floor((x0 - crop_x0) * scale)))
        ix1 = max(0, min(width, math.ceil((x1 - crop_x0) * scale)))
        iy0 = max(0, min(height, math.floor((crop_y1 - y1) * scale)))
        iy1 = max(0, min(height, math.ceil((crop_y1 - y0) * scale)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return ix0, iy0, ix1, iy1

    def page_x_to_pixel_span(start_x: float, end_x: float) -> tuple[int, int] | None:
        if end_x <= start_x:
            return None
        start = math.ceil((start_x - crop_x0) * scale - 0.5)
        end = math.ceil((end_x - crop_x0) * scale - 0.5)
        start = max(0, min(width, start))
        end = max(0, min(width, end))
        if end <= start:
            return None
        return start, end

    return page_box_to_pixels, page_x_to_pixel_span
