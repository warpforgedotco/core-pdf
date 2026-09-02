# SPDX-License-Identifier: AGPL-3.0-only
"""Stateful compositing operations for the mutable raster target."""

from __future__ import annotations

from typing import Any

from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    internal_blend_normal_solid_span_numpy,
    internal_composite_blended_group_numpy,
    internal_composite_normal_group_numpy,
)
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_BlendTargetMixin:
    """Stateful compositing operations shared by the mutable raster target."""

    __slots__ = ()

    def internal_resolved_blend(
        self: Any, blend_mode: str | None
    ) -> tuple[float | None, str | None]:
        """Resolve the half of `blend_px`'s arguments that is span-invariant.

        The enclosing group's alpha comes from `buffer_stack`, which none of the
        paint methods push or pop, and the lowercased blend mode is fixed for a
        whole call. Callers resolve both once and pass them down rather than
        making `blend_px` re-derive them on every pixel.
        """
        buffer_stack = self.buffer_stack
        target_alpha = buffer_stack[-1][1] if buffer_stack else None
        return (
            float(target_alpha) if pdf_number(target_alpha) else None,
            blend_mode.lower() if isinstance(blend_mode, str) else None,
        )

    def blend_px(
        self: Any,
        idx: int,
        rgba: tuple[int, int, int, int],
        target_alpha_scale: float | None,
        mode: str | None,
    ) -> None:
        """Blend one pixel, with the span-invariant arguments already resolved.

        `target_alpha_scale` is the enclosing group's alpha (`None` when absent)
        and `mode` the lowercased blend mode, both from `internal_resolved_blend`.
        This is the rasterizer's hottest method -- `fill_rect` alone reaches it
        ~1.8M times over the corpus -- so it takes them pre-resolved instead of
        re-reading `buffer_stack` and re-lowercasing `mode` on each call.
        """
        pixels = self.pixels
        sr, sg, sb, sa = rgba
        if sa <= 0:
            return
        if sa >= 255 and target_alpha_scale is None and mode is None:
            pixels[idx] = sr
            pixels[idx + 1] = sg
            pixels[idx + 2] = sb
            pixels[idx + 3] = 255
            return
        if target_alpha_scale is not None:
            sa = max(0, min(255, int(round(sa * target_alpha_scale))))
            if sa <= 0:
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
        if mode == "multiply":
            src_r *= dst_r
            src_g *= dst_g
            src_b *= dst_b
        elif mode == "screen":
            src_r = 1.0 - (1.0 - src_r) * (1.0 - dst_r)
            src_g = 1.0 - (1.0 - src_g) * (1.0 - dst_g)
            src_b = 1.0 - (1.0 - src_b) * (1.0 - dst_b)
        out_a = src_a + dst_a * (1.0 - src_a)
        if out_a <= 0:
            pixels[idx] = 0
            pixels[idx + 1] = 0
            pixels[idx + 2] = 0
            pixels[idx + 3] = 0
            return
        out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))

    def can_blend_normal_fast(self: Any, blend_mode: str | None) -> bool:
        return blend_mode is None and self.buffer_stack[-1][1] is None

    def blend_normal_pixel(self: Any, idx: int, sr: int, sg: int, sb: int, sa: int) -> None:
        if sa <= 0:
            return
        pixels = self.pixels
        if sa >= 255:
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
        out_a = src_a + dst_a * (1.0 - src_a)
        if out_a <= 0:
            pixels[idx] = 0
            pixels[idx + 1] = 0
            pixels[idx + 2] = 0
            pixels[idx + 3] = 0
            return
        out_r = int(round((sr * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round((sg * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round((sb * src_a + db * dst_a * (1.0 - src_a)) / out_a))
        out_a_i = int(round(out_a * 255.0))
        pixels[idx] = max(0, min(255, out_r))
        pixels[idx + 1] = max(0, min(255, out_g))
        pixels[idx + 2] = max(0, min(255, out_b))
        pixels[idx + 3] = max(0, min(255, out_a_i))

    def blend_normal_solid_span(
        self: Any, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
    ) -> None:
        sr, sg, sb, sa = rgba
        if sa <= 0 or end <= start:
            return
        pixels = self.pixels
        width = self.width
        if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
            target = self.pixel_view(pixels)
            internal_blend_normal_solid_span_numpy(target, row // (width * 4), start, end, rgba)
            return
        start_offset = row + start * 4
        stop_offset = row + end * 4
        if sa >= 255:
            # Keep the destination as a NumPy view instead of allocating a
            # repeated RGBA byte string for every short span.
            self.pixel_view(pixels)[row // (width * 4), start:end] = (sr, sg, sb, 255)
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
            if out_a <= 0:
                pixels[idx] = 0
                pixels[idx + 1] = 0
                pixels[idx + 2] = 0
                pixels[idx + 3] = 0
                continue
            out_r = int(round((sr * src_a + dr * dst_a * one_minus_src_a) / out_a))
            out_g = int(round((sg * src_a + dg * dst_a * one_minus_src_a) / out_a))
            out_b = int(round((sb * src_a + db * dst_a * one_minus_src_a) / out_a))
            out_a_i = int(round(out_a * 255.0))
            pixels[idx] = max(0, min(255, out_r))
            pixels[idx + 1] = max(0, min(255, out_g))
            pixels[idx + 2] = max(0, min(255, out_b))
            pixels[idx + 3] = max(0, min(255, out_a_i))

    def composite_group(
        self: Any, child: bytearray, group_alpha: float | None, group_blend_mode: str | None
    ) -> None:
        buffer_stack = self.buffer_stack
        normalized_blend_mode = (
            group_blend_mode.casefold() if isinstance(group_blend_mode, str) else None
        )
        parent_alpha = buffer_stack[-1][1] if buffer_stack else None
        if normalized_blend_mode in {None, "normal"} and len(child) >= 4_096:
            source_pixels = self.pixel_view(child)
            target_pixels = self.pixel_view(self.pixels)
            source_scale = float(group_alpha) if pdf_number(group_alpha) else 1.0
            target_scale = float(parent_alpha) if pdf_number(parent_alpha) else 1.0
            internal_composite_normal_group_numpy(
                target_pixels,
                source_pixels,
                source_scale,
                target_scale,
            )
            return
        # Both group alphas and the blend mode are invariant across the whole
        # buffer, so they are resolved once here rather than re-derived on every
        # pixel the way a `blend_px` loop would; the blend then runs as a single
        # vectorized pass instead of one Python call per pixel.
        internal_composite_blended_group_numpy(
            self.pixel_view(self.pixels),
            self.pixel_view(child),
            float(group_alpha) if pdf_number(group_alpha) else None,
            float(parent_alpha) if pdf_number(parent_alpha) else None,
            group_blend_mode,
        )
