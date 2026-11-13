# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: blending source pixels into the target, and the clip mask they
pass through.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.base.arrays import UInt8Array
from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    BlendOp,
    blend_normal_solid_array_numpy,
    blend_op,
    kernel_blend_code,
)
from core_pdf.impl.spec.s_11_transparency.blend import BlendMode, blend_component
lazy from core_pdf_cythonized import (
    blend_coverage_counts,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


def resolved_blend(self: RasterTarget, blend_mode: str | None) -> BlendOp | None:
    return blend_op(blend_mode)


def blend_px(
    self: RasterTarget,
    idx: int,
    rgba: tuple[int, int, int, int],
    mode: BlendOp | None,
    *,
    shape: int = 255,
) -> None:
    pixels = self.pixels
    sr, sg, sb, sa = rgba
    alpha_plane = self.group_source_alpha
    if self.group_source_shape is not None:
        row, column = divmod(idx // 4, self.width)
        self.record_source_shape(row, column, shape)
    elif alpha_plane is None and self.paint_window is not None:
        row, column = divmod(idx // 4, self.width)
        self.extend_paint_window(row, column)
    if sa <= 0:
        return
    if alpha_plane is not None:
        row, column = divmod(idx // 4, self.width)
        self.record_source_alpha(row, column, sa)
    if sa >= 255 and mode is None:
        pixels[idx] = sr
        pixels[idx + 1] = sg
        pixels[idx + 2] = sb
        pixels[idx + 3] = 255
        return
    dr = pixels[idx]
    dg = pixels[idx + 1]
    db = pixels[idx + 2]
    da = pixels[idx + 3]
    src_a = sa / 255.0
    dst_a = da / 255.0
    src_r = sr / 255.0
    src_g = sg / 255.0
    src_b = sb / 255.0
    dst_r = dr / 255.0
    dst_g = dg / 255.0
    dst_b = db / 255.0
    if mode is BlendOp.MULTIPLY:
        src_r = src_r * (1.0 - dst_a) + dst_a * (src_r * dst_r)
        src_g = src_g * (1.0 - dst_a) + dst_a * (src_g * dst_g)
        src_b = src_b * (1.0 - dst_a) + dst_a * (src_b * dst_b)
    elif mode is BlendOp.SCREEN:
        src_r = src_r * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_r) * (1.0 - dst_r))
        src_g = src_g * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_g) * (1.0 - dst_g))
        src_b = src_b * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_b) * (1.0 - dst_b))
    elif mode is BlendOp.COLOR_DODGE or mode is BlendOp.COLOR_BURN:
        component_mode: BlendMode = "ColorDodge" if mode is BlendOp.COLOR_DODGE else "ColorBurn"
        src_r = src_r * (1.0 - dst_a) + dst_a * blend_component(
            dst_r, src_r, component_mode, context=self.semantic_context
        )
        src_g = src_g * (1.0 - dst_a) + dst_a * blend_component(
            dst_g, src_g, component_mode, context=self.semantic_context
        )
        src_b = src_b * (1.0 - dst_a) + dst_a * blend_component(
            dst_b, src_b, component_mode, context=self.semantic_context
        )
    out_a = src_a + dst_a * (1.0 - src_a)
    out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
    out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
    out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
    out_a_i = int(round(out_a * 255.0))
    pixels[idx] = max(0, min(255, out_r))
    pixels[idx + 1] = max(0, min(255, out_g))
    pixels[idx + 2] = max(0, min(255, out_b))
    pixels[idx + 3] = max(0, min(255, out_a_i))


def blend_normal_solid_span(
    self: RasterTarget, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
) -> None:
    sr, sg, sb, sa = rgba
    if end <= start:
        return
    self.record_source_shape(row // (self.width * 4), slice(start, end), 255)
    if sa <= 0:
        return
    self.record_source_alpha(row // (self.width * 4), slice(start, end), sa)
    pixels = self.pixels
    width = self.width
    if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
        target = self.pixel_array
        blend_normal_solid_array_numpy(target[row // (width * 4), start:end], rgba)
        return
    start_offset = row + start * 4
    stop_offset = row + end * 4
    if sa >= 255:
        self.pixel_array[row // (width * 4), start:end] = (sr, sg, sb, 255)
        return
    src_a = sa / 255.0
    one_minus_src_a = 1.0 - src_a
    for idx in range(start_offset, stop_offset, 4):
        dr = pixels[idx]
        dg = pixels[idx + 1]
        db = pixels[idx + 2]
        da = pixels[idx + 3]
        dst_a = da / 255.0
        out_a = src_a + dst_a * one_minus_src_a
        out_r = int(round((sr * src_a + dr * dst_a * one_minus_src_a) / out_a))
        out_g = int(round((sg * src_a + dg * dst_a * one_minus_src_a) / out_a))
        out_b = int(round((sb * src_a + db * dst_a * one_minus_src_a) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))


def clip_pixel_mask(self: RasterTarget, ix0: int, iy0: int, ix1: int, iy1: int) -> bytearray:
    box_width = ix1 - ix0
    allowed = bytearray(box_width * (iy1 - iy0))
    ones = b"\x01" * box_width
    clip_row_visible_spans = self.clip.clip_row_visible_spans
    for py in range(iy0, iy1):
        row_start = (py - iy0) * box_width - ix0
        for span_start, span_end in clip_row_visible_spans(py):
            start = max(ix0, span_start)
            end = min(ix1, span_end)
            if end > start:
                allowed[row_start + start : row_start + end] = ones[: end - start]
    return allowed


def blend_rules(self: RasterTarget, mode: int) -> tuple[bool, Exception | None]:
    if mode not in {BlendOp.COLOR_DODGE, BlendOp.COLOR_BURN}:
        return True, None
    try:
        revised = blend_component(0.0, 1.0, "ColorDodge", context=self.semantic_context)
    except Exception as error:
        return True, error
    return revised == 0.0, None


def blend_counts(
    self: RasterTarget,
    counts: UInt8Array,
    left: int,
    top: int,
    allowed: bytearray | None,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> None:
    mode = kernel_blend_code(self.resolved_blend(blend_mode))
    revised, blend_error = self.blend_rules(mode)
    shape_plane = self.group_source_shape
    window, stopped = blend_coverage_counts(
        self.pixel_array,
        counts,
        left,
        top,
        allowed,
        *rgba,
        self.group_source_alpha,
        shape_plane,
        shape_plane is not None,
        self.shape_alpha,
        mode,
        revised,
        blend_error is not None,
    )
    if window is not None:
        self.extend_paint_window(slice(window[1], window[3]), slice(window[0], window[2]))
    if stopped:
        assert blend_error is not None
        raise blend_error
