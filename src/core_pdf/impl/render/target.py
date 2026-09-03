# SPDX-License-Identifier: AGPL-3.0-only
"""Mutable raster target and clipping."""

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    internal_blend_normal_solid_span_numpy,
    internal_color_rgba,
    internal_composite_blended_group_numpy,
    internal_composite_normal_group_numpy,
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.clipping import internal_ClipState
from core_pdf.impl.render.image_affine_target import internal_ImageAffineTargetMixin
from core_pdf.impl.render.image_axis_target import internal_ImageAxisTargetMixin
from core_pdf.impl.render.kernels import RASTER_COORDINATE_CACHE_MAX_ENTRIES
from core_pdf.impl.render.model import PathPaintItem, PathPaintKind
from core_pdf.impl.render.path_fill_target import internal_PathFillTargetMixin
from core_pdf.impl.render.path_shape_target import internal_PathShapeTargetMixin
from core_pdf.impl.render.path_stroke_target import internal_PathStrokeTargetMixin
from core_pdf.impl.render.patterns import internal_PatternTargetMixin
from core_pdf.impl.runtime.array_views import uint8_image_view
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_RasterTarget(
    internal_ImageAffineTargetMixin,
    internal_ImageAxisTargetMixin,
    internal_PathShapeTargetMixin,
    internal_PathFillTargetMixin,
    internal_PathStrokeTargetMixin,
    internal_PatternTargetMixin,
):
    """The RGBA byte buffer being painted, plus the transparency-group stack.

    Lifted out of ``RenderedPage.rasterize``. ``pixels`` is *rebound*, not just
    mutated: a ``group-begin`` pushes a fresh buffer that subsequent painting
    goes to, and ``group-end`` pops it and composites it back down. That is why
    this is an object with explicit push/pop rather than a plain buffer.

    Every method hoists ``self.pixels`` into a local before touching it — these
    run per pixel and per span, where a repeated attribute load is not free.
    """

    __slots__ = (
        "pixels",
        "buffer_stack",
        "pixel_views",
        "clip",
        "width",
        "height",
        "scale",
        "crop_x0",
        "crop_y1",
        "page_pixels",
        "page_buffer",
        "raster_x_coordinate_cache",
        "raster_y_coordinate_cache",
        "raster_x_sample_cache",
        "raster_y_sample_cache",
        # Bound clip methods cached at construction. Reading them back is a plain
        # attribute load; going through `self.clip.<name>` would allocate a fresh
        # bound method on every call, and fill_rect alone makes ~1.8M of them.
        "page_box_to_pixels",
        "clipped_pixel_box",
        "current_clip",
        "clip_paths_are_axis_aligned_rects",
        "clip_row_visible_spans",
        "pixel_in_clip",
        "page_x_coordinate_cache",
        "page_y_coordinate_cache",
        "crop_y0",
        "color_cache",
        "mark_clip_metadata_dirty",
        "path_bbox",
        "clip_path_stack",
    )

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: internal_ClipState,
        width: int,
        height: int,
        scale: float,
        crop_x0: float,
        crop_y0: float,
        crop_y1: float,
        page_view: numpy.ndarray[Any, Any],
    ) -> None:
        self.pixels = pixels
        self.buffer_stack: list[tuple[bytearray, float | None, str | None]] = [
            (pixels, group_alpha, None)
        ]
        self.pixel_views: dict[int, numpy.ndarray[Any, Any]] = {id(pixels): page_view}
        self.clip = clip
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.raster_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.raster_x_sample_cache: dict[int, tuple[float, ...]] = {}
        self.raster_y_sample_cache: dict[int, tuple[float, ...]] = {}
        self.page_box_to_pixels = clip.page_box_to_pixels
        self.clipped_pixel_box = clip.clipped_pixel_box
        self.current_clip = clip.current_clip
        self.clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        self.clip_row_visible_spans = clip.clip_row_visible_spans
        self.pixel_in_clip = clip.pixel_in_clip
        self.page_x_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.page_y_coordinate_cache: dict[tuple[int, int], numpy.ndarray[Any, Any]] = {}
        self.crop_y0 = crop_y0
        self.color_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.mark_clip_metadata_dirty = clip.mark_clip_metadata_dirty
        self.path_bbox = clip.path_bbox
        self.clip_path_stack = clip.clip_path_stack

    def push_group(
        self, buffer: bytearray, group_alpha: float | None, blend_mode: str | None
    ) -> None:
        self.buffer_stack.append((buffer, group_alpha, blend_mode))
        self.pixels = buffer

    def pop_group(self) -> tuple[bytearray, float | None, str | None]:
        child = self.buffer_stack.pop()
        self.pixels = self.buffer_stack[-1][0]
        return child

    def pixel_view(self, buffer: bytearray | bytes) -> numpy.ndarray[Any, Any]:
        """Return a reusable array view for an active RGBA byte buffer."""
        pixel_views = self.pixel_views
        key = id(buffer)
        view = pixel_views.get(key)
        if view is None:
            view = uint8_image_view(buffer, (self.height, self.width, 4))
            pixel_views[key] = view
        return view

    def internal_resolved_blend(self, blend_mode: str | None) -> tuple[float | None, str | None]:
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
        self,
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

    def can_blend_normal_fast(self, blend_mode: str | None) -> bool:
        return blend_mode is None and self.buffer_stack[-1][1] is None

    def blend_normal_pixel(self, idx: int, sr: int, sg: int, sb: int, sa: int) -> None:
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
        self, row: int, start: int, end: int, rgba: tuple[int, int, int, int]
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
        self, child: bytearray, group_alpha: float | None, group_blend_mode: str | None
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

    def paint_typed_path(self, item: PathPaintItem) -> None:
        color_cache = self.color_cache
        fill_path = self.fill_path
        stroke_path = self.stroke_path
        path = item.path
        if type(path) is not CapturedPath:
            return
        item_bbox = item.bbox
        if item_bbox is not None:
            # compose_page already computed ``path.bbox()`` for the item.
            self.clip.path_bbox_cache.setdefault(id(path), item_bbox)
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        soft_mask_alpha = item.soft_mask_alpha
        paint_kind = item.paint_kind
        if paint_kind is not PathPaintKind.STROKE:
            color_key = (id(item.fill), id(item.fill_opacity))
            rgba = color_cache.get(color_key)
            if rgba is None:
                rgba = internal_color_rgba(item.fill, item.fill_opacity)
                if len(color_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    color_cache[color_key] = rgba
            if pdf_number(soft_mask_alpha):
                rgba = internal_scale_rgba_alpha(rgba, soft_mask_alpha)
            fill_path(
                path,
                rgba,
                blend_mode,
                item.fill_rule or "nonzero",
            )
        if paint_kind is not PathPaintKind.FILL:
            color_key = (id(item.stroke_color), id(item.stroke_opacity))
            stroke_rgba = color_cache.get(color_key)
            if stroke_rgba is None:
                stroke_rgba = internal_color_rgba(item.stroke_color, item.stroke_opacity)
                if len(color_cache) < RASTER_COORDINATE_CACHE_MAX_ENTRIES:
                    color_cache[color_key] = stroke_rgba
            if pdf_number(soft_mask_alpha):
                stroke_rgba = internal_scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
            stroke_path(
                path,
                float(item.line_width or 1.0),
                stroke_rgba,
                item.dash_pattern,
                blend_mode,
                int(item.line_cap or 0),
                int(item.line_join or 0),
            )
