# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.graphics.device_profiles import cmyk_floats_to_srgb, component_byte
from core_pdf.impl.scalars import clamp01
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_11_transparency.blend import BlendMode, blend_components
from core_pdf_spec.standards import PdfVersion, SemanticContext

RASTER_NUMPY_SPAN_MIN_PIXELS = 32
FALLBACK_BLEND_CONTEXT = SemanticContext(PdfVersion(2, 0))


def blend_context(context: SemanticContext | None) -> SemanticContext:
    if context is None or context.version is None or not context.version.recognized:
        return FALLBACK_BLEND_CONTEXT
    return context


def blend_normal_solid_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
) -> None:
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


def blend_channels_f64(
    src_r: numpy.ndarray | float,
    src_g: numpy.ndarray | float,
    src_b: numpy.ndarray | float,
    src_a: numpy.ndarray | float,
    dr: numpy.ndarray,
    dg: numpy.ndarray,
    db: numpy.ndarray,
    da: numpy.ndarray,
    mode: str | None,
    *,
    semantic_context: SemanticContext | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    one_minus_src_a = 1.0 - src_a
    dst_a = da / 255.0
    if mode == "multiply":
        src_r = src_r * (1.0 - dst_a) + dst_a * (src_r * (dr / 255.0))
        src_g = src_g * (1.0 - dst_a) + dst_a * (src_g * (dg / 255.0))
        src_b = src_b * (1.0 - dst_a) + dst_a * (src_b * (db / 255.0))
    elif mode == "screen":
        src_r = src_r * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_r) * (1.0 - dr / 255.0))
        src_g = src_g * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_g) * (1.0 - dg / 255.0))
        src_b = src_b * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_b) * (1.0 - db / 255.0))
    elif mode in {"colordodge", "colorburn"}:
        component_mode: BlendMode = "ColorDodge" if mode == "colordodge" else "ColorBurn"
        context = blend_context(semantic_context)
        src_r = src_r * (1.0 - dst_a) + dst_a * blend_components(
            dr / 255.0, src_r, component_mode, context=context
        )
        src_g = src_g * (1.0 - dst_a) + dst_a * blend_components(
            dg / 255.0, src_g, component_mode, context=context
        )
        src_b = src_b * (1.0 - dst_a) + dst_a * blend_components(
            db / 255.0, src_b, component_mode, context=context
        )
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


def blend_solid_array_numpy(
    target: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext | None = None,
) -> None:
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
    out_r, out_g, out_b, out_a_i = blend_channels_f64(
        sr / 255.0,
        sg / 255.0,
        sb / 255.0,
        sa / 255.0,
        dr,
        dg,
        db,
        da,
        mode,
        semantic_context=semantic_context,
    )
    target[..., 0] = numpy.clip(out_r, 0.0, 255.0).astype(numpy.uint8)
    target[..., 1] = numpy.clip(out_g, 0.0, 255.0).astype(numpy.uint8)
    target[..., 2] = numpy.clip(out_b, 0.0, 255.0).astype(numpy.uint8)
    target[..., 3] = numpy.clip(out_a_i, 0.0, 255.0).astype(numpy.uint8)


def composite_blended_group_numpy(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_alpha_scale: float | None,
    target_alpha_scale: float | None,
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext | None = None,
) -> None:
    if destination.size == 0:
        return
    source_alpha = source[..., 3].astype(numpy.float64)
    if source_alpha_scale is not None:
        source_alpha = numpy.clip(numpy.rint(source_alpha * source_alpha_scale), 0.0, 255.0)
    if target_alpha_scale is not None:
        source_alpha = numpy.clip(numpy.rint(source_alpha * target_alpha_scale), 0.0, 255.0)
    visible = source_alpha > 0.0
    if not numpy.any(visible):
        return
    dr = destination[..., 0][visible].astype(numpy.float64)
    dg = destination[..., 1][visible].astype(numpy.float64)
    db = destination[..., 2][visible].astype(numpy.float64)
    da = destination[..., 3][visible].astype(numpy.float64)
    mode = blend_mode.lower() if isinstance(blend_mode, str) else None
    out_r, out_g, out_b, out_a_i = blend_channels_f64(
        source[..., 0][visible].astype(numpy.float64) / 255.0,
        source[..., 1][visible].astype(numpy.float64) / 255.0,
        source[..., 2][visible].astype(numpy.float64) / 255.0,
        source_alpha[visible] / 255.0,
        dr,
        dg,
        db,
        da,
        mode,
        semantic_context=semantic_context,
    )
    for channel, values in ((0, out_r), (1, out_g), (2, out_b), (3, out_a_i)):
        destination[..., channel][visible] = numpy.clip(values, 0.0, 255.0).astype(numpy.uint8)


def composite_normal_group_numpy(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    source_alpha_scale: float,
    target_alpha_scale: float = 1.0,
) -> None:
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


def color_component(value: Any, default: int = 0) -> int:
    """The tolerant form of `component_byte`, for values off a content stream."""
    if type(value) is bool:
        return default
    try:
        return component_byte(float(value))
    except TypeError, ValueError:
        return default


def resolve_constant_alpha(opacity: object, soft_mask_alpha: object) -> float:
    return (float(opacity) if is_pdf_number(opacity) else 1.0) * (
        float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else 1.0
    )


def blend_visible_pixels(
    destination: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]],
    red: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    green: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    blue: float | numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    alpha: numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext | None = None,
) -> None:
    backdrop = destination[visible].astype(numpy.float64)
    channels = blend_channels_f64(
        red,
        green,
        blue,
        alpha,
        backdrop[:, 0],
        backdrop[:, 1],
        backdrop[:, 2],
        backdrop[:, 3],
        blend_mode,
        semantic_context=semantic_context,
    )
    destination[visible] = numpy.clip(numpy.column_stack(channels), 0, 255).astype(numpy.uint8)


def color_rgba(color: Any, opacity: Any) -> tuple[int, int, int, int]:
    alpha = 255
    if type(opacity) in {int, float}:
        alpha = color_component(opacity, 255)
    if isinstance(color, (list, tuple)) and color:
        if len(color) == 1:
            gray = color_component(color[0])
            return gray, gray, gray, alpha
        if len(color) == 4:
            try:
                cyan, magenta, yellow, black = (clamp01(float(c)) for c in color)
            except TypeError, ValueError:
                return 0, 0, 0, alpha
            red, green, blue = cmyk_floats_to_srgb(cyan, magenta, yellow, black)
            return red, green, blue, alpha
        rgb = [color_component(c) for c in color[:3]]
        while len(rgb) < 3:
            rgb.append(rgb[-1] if rgb else 0)
        return rgb[0], rgb[1], rgb[2], alpha
    return 0, 0, 0, alpha


def scale_rgba_alpha(
    rgba: tuple[int, int, int, int],
    alpha_scale: Any,
) -> tuple[int, int, int, int]:
    return (
        rgba[0],
        rgba[1],
        rgba[2],
        max(0, min(255, int(round(rgba[3] * float(alpha_scale))))),
    )
