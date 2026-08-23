# SPDX-License-Identifier: AGPL-3.0-only
"""Pure and allocation-bounded raster kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy

from core_pdf.impl.engine.array_views import (
    ByteBuffer,
    contiguous_bytes,
    uint8_view,
)
from core_pdf.impl.engine.layout.geometry import RectBox
from core_pdf.impl.engine.render.display import (
    DisplayItem,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.engine.render.raster_image import RasterImage
from core_pdf.impl.engine.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.engine.spec.s_07_filters.models import DecodedImage
from core_pdf.impl.engine.spec.s_07_filters.pipeline import decode_stream_data
from core_pdf.impl.engine.spec.s_07_objects.coercion import parse_float
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.color import ImageColorManager
from core_pdf.impl.engine.spec.s_08_graphics.color_kernels import (
    evaluate_sampled_tint_function,
)
from core_pdf.impl.engine.spec.s_08_graphics.image_decode import (
    decode_pdf_image_samples,
)
from core_pdf.impl.engine.spec.s_08_graphics.image_metadata import (
    image_color_space_name,
    pdf_float,
    pdf_int,
    pdf_number,
)
from core_pdf.impl.objects import PdfStream

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
BIT_IMAGE_MASK_ALPHA = tuple(
    bytes(255 if byte & (0x80 >> bit) else 0 for bit in range(8)) for byte in range(256)
)
BIT_IMAGE_MASK_ALPHA_ARRAY = numpy.frombuffer(
    b"".join(BIT_IMAGE_MASK_ALPHA), dtype=numpy.uint8
).reshape(256, 8)
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
    target[...] = destination_float.astype(numpy.uint8)


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
    gray = numpy.full((raster_height, raster_width), 255, dtype=numpy.uint8)
    page_height = float(height)

    def plot(x: int, y: int, coverage: float) -> None:
        if coverage <= 0.0 or not (0 <= x < raster_width and 0 <= y < raster_height):
            return
        value = int(gray[y, x])
        gray[y, x] = max(0, min(255, round(value * (1.0 - coverage))))

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
    rgba[:, :, :3] = gray[:, :, None]
    rgba[:, :, 3] = 255
    return RasterImage(rgba, raster_width, raster_height, 4)


def number_array(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = parse_float(item, None)
        if parsed is None:
            return []
        out.append(parsed)
    return out


def evaluate_pdf_function(function: Any, value: float) -> list[float]:
    if isinstance(function, PdfStream):
        function_type = pdf_int(lookup_dict_key(function.dictionary, "FunctionType"), -1)
        if function_type == 0:
            try:
                return evaluate_sampled_tint_function(function, value)
            except Exception:
                return [value]
        dictionary = function.dictionary
    elif isinstance(function, dict):
        function_type = pdf_int(lookup_dict_key(function, "FunctionType"), -1)
        dictionary = function
    else:
        return [value]

    if function_type == 2:
        exponent = pdf_float(lookup_dict_key(dictionary, "N"), 1.0)
        c0 = number_array(lookup_dict_key(dictionary, "C0")) or [0.0]
        c1 = number_array(lookup_dict_key(dictionary, "C1")) or [1.0]
        count = max(len(c0), len(c1))
        if len(c0) < count:
            c0.extend([c0[-1] if c0 else 0.0] * (count - len(c0)))
        if len(c1) < count:
            c1.extend([c1[-1] if c1 else 1.0] * (count - len(c1)))
        factor = value**exponent
        return [c0[i] + factor * (c1[i] - c0[i]) for i in range(count)]

    if function_type == 3:
        functions = lookup_dict_key(dictionary, "Functions")
        if not isinstance(functions, (list, tuple)) or not functions:
            return [value]
        bounds = number_array(lookup_dict_key(dictionary, "Bounds"))
        encode = number_array(lookup_dict_key(dictionary, "Encode"))
        index = 0
        while index < len(bounds) and value >= bounds[index]:
            index += 1
        low = bounds[index - 1] if index > 0 else 0.0
        high = bounds[index] if index < len(bounds) else 1.0
        enc0 = encode[index * 2] if index * 2 < len(encode) else 0.0
        enc1 = encode[index * 2 + 1] if index * 2 + 1 < len(encode) else 1.0
        if high == low:
            encoded = enc0
        else:
            encoded = enc0 + (value - low) * (enc1 - enc0) / (high - low)
        return evaluate_pdf_function(functions[min(index, len(functions) - 1)], encoded)

    return [value]


def axial_shading_t(coords: list[float], px: float, py: float) -> float | None:
    x0, y0, x1, y1 = coords[:4]
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return None
    return ((px - x0) * dx + (py - y0) * dy) / denom


def radial_shading_t(coords: list[float], px: float, py: float) -> float | None:
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


def internal_image_raw_bytes(raw: bytes | bytearray | memoryview) -> bytes | memoryview:
    """Return image source storage without copying it."""
    if type(raw) is bytes or type(raw) is memoryview:
        return raw
    return memoryview(raw).cast("B")


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


def internal_fill_path_sample_crossings_numpy(
    edge_segments: numpy.ndarray[Any, Any],
    page_ys: numpy.ndarray[Any, Any],
) -> list[list[tuple[float, int]]]:
    crossings_rows: list[list[tuple[float, int]]] = []
    for page_y in page_ys:
        active = edge_segments[(edge_segments[:, 4] <= page_y) & (page_y < edge_segments[:, 5])]
        if not len(active):
            crossings_rows.append([])
            continue
        delta_y = active[:, 3] - active[:, 1]
        intersections = active[:, 0] + (
            (page_y - active[:, 1]) / delta_y * (active[:, 2] - active[:, 0])
        )
        directions = numpy.where(delta_y > 0.0, 1, -1)
        crossings_rows.append(list(zip(intersections.tolist(), directions.tolist(), strict=True)))
    return crossings_rows


def internal_fill_path_crossings_contain_point(
    crossings: list[tuple[float, int]],
    page_x: float,
    fill_rule: str,
) -> bool:
    if fill_rule == "evenodd":
        odd = False
        for x_intersection, internal_delta in crossings:
            if x_intersection > page_x:
                odd = not odd
        return odd
    winding = 0
    for x_intersection, delta in crossings:
        if x_intersection > page_x:
            winding += delta
    return winding != 0


def internal_fill_path_crossing_spans(
    crossings: list[tuple[float, int]],
    fill_rule: str,
) -> list[tuple[float, float]]:
    if not crossings:
        return []
    if fill_rule == "evenodd":
        xs = sorted(x for x, internal_delta in crossings)
        return [(start, end) for start, end in zip(xs[0::2], xs[1::2], strict=False) if end > start]
    crossings.sort(key=lambda item: item[0])
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


def internal_image_samples(
    raw: bytes | memoryview, dictionary: dict[Any, Any]
) -> tuple[bytes | DecodedImage | memoryview, dict[Any, Any]] | None:
    return decode_pdf_image_samples(raw, dictionary)


def internal_image_mask_samples(
    raw: bytes | memoryview, dictionary: dict[Any, Any], width_px: int, height_px: int
) -> bytes | memoryview:
    if width_px <= 0 or height_px <= 0:
        return b""
    try:
        decoded = decode_stream_data(raw, dictionary)
    except Exception:
        decoded = raw
    row_bytes = (width_px + 7) >> 3
    if len(decoded) < row_bytes * height_px:
        return b""
    packed = uint8_view(decoded, count=row_bytes * height_px)
    expanded = BIT_IMAGE_MASK_ALPHA_ARRAY[packed].reshape(height_px, row_bytes * 8)
    return contiguous_bytes(expanded[:, :width_px])


def internal_image_mask_decode_inverts(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        return float(value[0]) > float(value[1])
    except (TypeError, ValueError):
        return False


def internal_soft_mask_alpha_at(
    mask: tuple[bytes | memoryview, int, int] | None, u: float, v: float
) -> int:
    if mask is None:
        return 255
    samples, mask_width, mask_height = mask
    src_x = min(mask_width - 1, max(0, int(u * mask_width)))
    src_y = min(mask_height - 1, max(0, int((1.0 - v) * mask_height)))
    idx = src_y * mask_width + src_x
    return samples[idx] if idx < len(samples) else 255


def internal_image_quad(data: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    quad = data.get("quad")
    if isinstance(quad, (list, tuple)) and len(quad) >= 3:
        try:
            return tuple((float(point[0]), float(point[1])) for point in quad)
        except (TypeError, ValueError, IndexError):
            return None
    items = data.get("items")
    if not isinstance(items, list):
        return None
    for kind, value in items:
        if kind != "quad":
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return tuple((float(point[0]), float(point[1])) for point in value)
        except (TypeError, ValueError, IndexError):
            return None
    return None


# Colour and soft-mask helpers lifted out of RenderedPage.rasterize.
def internal_color_rgba(color: Any, opacity: Any) -> tuple[int, int, int, int]:
    alpha = 255
    if pdf_number(opacity):
        alpha = internal_color_component(opacity, 255)
    if isinstance(color, (list, tuple)) and color:
        if len(color) == 1:
            gray = internal_color_component(color[0])
            return gray, gray, gray, alpha
        rgb = [internal_color_component(c) for c in color[:3]]
        while len(rgb) < 3:
            rgb.append(rgb[-1] if rgb else 0)
        return rgb[0], rgb[1], rgb[2], alpha
    return 0, 0, 0, alpha


def internal_shading_color_rgba(
    color_space: Any, components: list[float], opacity: Any
) -> tuple[int, int, int, int]:
    alpha = internal_color_component(opacity, 255) if pdf_number(opacity) else 255
    name = image_color_space_name(color_space) or "DeviceRGB"
    if name.endswith("DeviceGray") or len(components) == 1:
        gray = internal_color_component(components[0] if components else 0.0)
        return gray, gray, gray, alpha
    if name.endswith("DeviceCMYK") and len(components) >= 4:
        c, m, y, k = (internal_clamp01(v) for v in components[:4])
        return (
            max(0, min(255, int(round(255.0 * (1.0 - c) * (1.0 - k))))),
            max(0, min(255, int(round(255.0 * (1.0 - m) * (1.0 - k))))),
            max(0, min(255, int(round(255.0 * (1.0 - y) * (1.0 - k))))),
            alpha,
        )
    rgb = [internal_color_component(c) for c in components[:3]]
    while len(rgb) < 3:
        rgb.append(rgb[-1] if rgb else 0)
    return rgb[0], rgb[1], rgb[2], alpha


def internal_soft_mask_samples(data: dict[str, Any]) -> tuple[bytes | memoryview, int, int] | None:
    dictionary = data.get("dictionary")
    if not isinstance(dictionary, dict):
        return None
    raw = dictionary.get("__soft_mask_raw_data__")
    mask_dict = dictionary.get("__soft_mask_dictionary__")
    if not isinstance(raw, (bytes, bytearray, memoryview)) or not isinstance(mask_dict, dict):
        return None
    mask_width = pdf_int(lookup_dict_key(mask_dict, "Width"), 0)
    mask_height = pdf_int(lookup_dict_key(mask_dict, "Height"), 0)
    if mask_width <= 0 or mask_height <= 0:
        return None
    sample_dict = dict(mask_dict)
    sample_dict.setdefault("ColorSpace", "DeviceGray")
    sample_dict.setdefault("BitsPerComponent", 8)
    raw_bytes = internal_image_raw_bytes(raw)
    sample_result = internal_image_samples(raw_bytes, sample_dict)
    samples: bytes | memoryview | DecodedImage
    if sample_result is None:
        samples = raw_bytes
    else:
        samples, sample_dict = sample_result
    converted: ByteBuffer
    if isinstance(samples, DecodedImage):
        converted = samples.array.reshape(-1)
    else:
        converted_or_none: ByteBuffer | None
        try:
            converted_or_none = ImageColorManager.convert_image_data(samples, sample_dict)
        except Exception:
            converted_or_none = None
        converted = converted_or_none if converted_or_none is not None else samples
    pixel_count = mask_width * mask_height
    converted_array = uint8_view(converted)
    if len(converted_array) >= pixel_count * 3:
        converted_array = numpy.ascontiguousarray(converted_array[0 : pixel_count * 3 : 3])
    elif len(converted_array) < pixel_count:
        return None
    return (
        contiguous_bytes(converted_array[:pixel_count]),
        mask_width,
        mask_height,
    )


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
