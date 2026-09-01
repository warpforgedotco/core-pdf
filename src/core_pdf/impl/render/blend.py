# SPDX-License-Identifier: AGPL-3.0-only
"""Colour conversion and straight-alpha compositing kernels."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.spec.s_08_graphics.device_profiles import cmyk_floats_to_srgb


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


def internal_blend_channels_f64(
    src_r: numpy.ndarray | float,
    src_g: numpy.ndarray | float,
    src_b: numpy.ndarray | float,
    src_a: numpy.ndarray | float,
    dr: numpy.ndarray,
    dg: numpy.ndarray,
    db: numpy.ndarray,
    da: numpy.ndarray,
    mode: str | None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Multiply/screen/normal source-over in float64, shared by the solid and group paths.

    Inputs are unit-scaled source channels (scalar or array) and 0-255 float64
    destination channels; the result is rounded 0-255 channels with fully
    transparent pixels zeroed, exactly as ``blend_px`` computes them. Operation
    order is part of the contract: the golden-raster digests depend on it.
    """
    one_minus_src_a = 1.0 - src_a
    dst_a = da / 255.0
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
    return out_r, out_g, out_b, out_a_i


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
    out_r, out_g, out_b, out_a_i = internal_blend_channels_f64(
        sr / 255.0, sg / 255.0, sb / 255.0, sa / 255.0, dr, dg, db, da, mode
    )
    target[..., 0] = numpy.clip(out_r, 0.0, 255.0).astype(numpy.uint8)
    target[..., 1] = numpy.clip(out_g, 0.0, 255.0).astype(numpy.uint8)
    target[..., 2] = numpy.clip(out_b, 0.0, 255.0).astype(numpy.uint8)
    target[..., 3] = numpy.clip(out_a_i, 0.0, 255.0).astype(numpy.uint8)


def internal_composite_blended_group_numpy(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_alpha_scale: float | None,
    target_alpha_scale: float | None,
    blend_mode: str | None,
) -> None:
    """Composite a straight-alpha group that carries a blend mode, in one pass.

    The per-pixel twin of this ran ``blend_px`` once per pixel for every group
    whose blend mode was not Normal, re-resolving the enclosing group's alpha
    and re-lowercasing ``blend_mode`` on each of them. This is the same math
    hoisted out of that loop and vectorized: a source *array* where
    ``internal_blend_solid_array_numpy`` takes a single colour.

    ``source_alpha_scale`` is the group's own alpha and ``target_alpha_scale``
    the enclosing group's. ``None`` means "absent", i.e. not applied. They are
    rounded and clamped separately, in that order, exactly as the two scalar
    steps did -- folding them into one multiply would round once and drift.

    Uses float64 throughout (not the float32 of the Normal-blend fast paths) so
    every intermediate matches ``blend_px``'s plain-Python-float arithmetic bit
    for bit; the golden-raster digests depend on it.

    Indexes both views' last axis directly rather than reshaping to ``(-1, 4)``
    -- see ``internal_blend_normal_solid_array_numpy`` for why a reshaped
    non-contiguous destination slice silently discards every write.
    """
    if destination.size == 0:
        return
    source_alpha = source[..., 3].astype(numpy.float64)
    if source_alpha_scale is not None:
        source_alpha = numpy.clip(numpy.rint(source_alpha * source_alpha_scale), 0.0, 255.0)
    if target_alpha_scale is not None:
        source_alpha = numpy.clip(numpy.rint(source_alpha * target_alpha_scale), 0.0, 255.0)
    # A pixel invisible at any stage stays invisible: scaling is monotonic and
    # clamped at zero, so this one test covers all three scalar early-returns.
    visible = source_alpha > 0.0
    if not numpy.any(visible):
        return
    # Everything below runs on the visible pixels only. A group is usually a
    # shaped region inside a full-page buffer, so this is most of the buffer.
    dr = destination[..., 0][visible].astype(numpy.float64)
    dg = destination[..., 1][visible].astype(numpy.float64)
    db = destination[..., 2][visible].astype(numpy.float64)
    da = destination[..., 3][visible].astype(numpy.float64)
    mode = blend_mode.lower() if isinstance(blend_mode, str) else None
    out_r, out_g, out_b, out_a_i = internal_blend_channels_f64(
        source[..., 0][visible].astype(numpy.float64) / 255.0,
        source[..., 1][visible].astype(numpy.float64) / 255.0,
        source[..., 2][visible].astype(numpy.float64) / 255.0,
        source_alpha[visible] / 255.0,
        dr,
        dg,
        db,
        da,
        mode,
    )
    for channel, values in ((0, out_r), (1, out_g), (2, out_b), (3, out_a_i)):
        # `destination[..., channel]` is a basic-indexing view, so the masked
        # assignment writes through to `destination` itself.
        destination[..., channel][visible] = numpy.clip(values, 0.0, 255.0).astype(numpy.uint8)


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


def internal_color_component(value: Any, default: int = 0) -> int:
    if type(value) is bool:
        return default
    try:
        return max(0, min(255, int(round(float(value) * 255.0))))
    except (TypeError, ValueError):
        return default


def internal_clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
