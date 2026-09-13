# SPDX-License-Identifier: AGPL-3.0-only
"""Mutable raster target and clipping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy

from core_pdf.impl._impl.capture.records import CapturedPath, CapturedSoftMask
from core_pdf.impl._impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    internal_blend_context,
    internal_blend_normal_solid_array_numpy,
    internal_color_rgba,
    internal_composite_blended_group_numpy,
    internal_composite_normal_group_numpy,
    internal_scale_rgba_alpha,
)
from core_pdf.impl._impl.render.clipping import internal_ClipState
from core_pdf.impl._impl.render.commands import translated_command
from core_pdf.impl._impl.render.groups import (
    internal_composite_knockout_group,
    internal_composite_masked_group,
    internal_composite_nonisolated_group,
)
from core_pdf.impl._impl.render.image_affine_target import internal_ImageAffineTargetMixin
from core_pdf.impl._impl.render.image_axis_target import internal_ImageAxisTargetMixin
from core_pdf.impl._impl.render.model import (
    DisplayItem,
    ImagePaintItem,
    PathPaintItem,
    PathPaintKind,
    internal_RasterGroup,
)
from core_pdf.impl._impl.render.path_fill_target import internal_PathFillTargetMixin
from core_pdf.impl._impl.render.path_shape_target import internal_PathShapeTargetMixin
from core_pdf.impl._impl.render.path_stroke_target import internal_PathStrokeTargetMixin
from core_pdf.impl._impl.render.patterns import internal_PatternTargetMixin
from core_pdf.impl._impl.render.soft_masks import (
    SoftMaskCache,
    SoftMaskKey,
    SoftMaskPlane,
    internal_graphics_soft_mask,
    internal_resolve_soft_mask,
)
from core_pdf.impl._impl.render.stroke_paint import internal_paint_stroke_once
from core_pdf.impl._impl.runtime.array_views import UInt8Array, uint8_image_view
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_11_transparency.blend import BlendMode, blend_component
from core_pdf_spec.standards import SemanticContext


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

    Painting helpers operate against this target instead of capturing duplicate
    renderer state in closures.
    """

    __slots__ = (
        "pixels",
        "semantic_context",
        "buffer_stack",
        "group_source_alpha",
        "group_source_shape",
        "paint_alpha_is_shape",
        "shape_alpha",
        "clip",
        "width",
        "height",
        "scale",
        "crop_x0",
        "crop_y1",
        "page_pixels",
        "page_buffer",
        "crop_y0",
        "clip_stack",
        "clip_floor",
        "group_floor",
        "scope_stack",
        "soft_mask_cache",
        "active_soft_masks",
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
        page_view: UInt8Array,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.pixels = pixels
        self.semantic_context = internal_blend_context(semantic_context)
        self.buffer_stack = [internal_RasterGroup(pixels)]
        self.group_source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None
        self.group_source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None
        self.paint_alpha_is_shape = False
        self.shape_alpha = 1.0
        self.clip = clip
        self.width = width
        self.height = height
        self.scale = scale
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.crop_y0 = crop_y0
        self.clip_stack: list[int] = []
        self.clip_floor = 0
        self.group_floor = 1
        self.scope_stack: list[tuple[int, list[int], int, int, int]] = []
        self.soft_mask_cache: SoftMaskCache = {}
        self.active_soft_masks: set[SoftMaskKey] = set()
        if group_alpha is not None:
            self.push_group(bytearray(len(pixels)), group_alpha, None)
            self.group_floor = len(self.buffer_stack)

    def push_scope(self, clip_path: CapturedPath | None = None) -> None:
        """Isolate a source scope from malformed q/Q or group boundaries."""
        self.scope_stack.append(
            (
                self.clip.depth,
                self.clip_stack,
                self.clip_floor,
                len(self.buffer_stack),
                self.group_floor,
            )
        )
        self.clip_stack = []
        if clip_path is not None:
            self.clip.push(clip_path, "nonzero")
        self.clip_floor = self.clip.depth
        self.group_floor = len(self.buffer_stack)

    def pop_scope(self) -> None:
        if not self.scope_stack:
            return
        clip_depth, clip_stack, clip_floor, buffer_depth, group_floor = self.scope_stack.pop()
        while len(self.buffer_stack) > buffer_depth:
            self.composite_group(self.pop_group())
        self.clip.restore(clip_depth)
        self.clip_stack = clip_stack
        self.clip_floor = clip_floor
        self.group_floor = group_floor

    def paint_items(
        self,
        items: Iterable[DisplayItem],
        *,
        translation: tuple[float, float] | None = None,
        parent_blend_mode: str | None = None,
        clip_path: CapturedPath | None = None,
    ) -> None:
        """Replay canonical commands; repeated cells get an isolated clip scope."""
        if translation is None:
            for item in items:
                self.paint_item(item)
            return
        self.push_scope(clip_path.translated(*translation) if clip_path is not None else None)
        try:
            for item in items:
                self.paint_item(translated_command(item, *translation, parent_blend_mode))
        finally:
            self.pop_scope()

    def paint_item(self, item: DisplayItem) -> None:
        """One paint/scope dispatcher, shared by pages and pattern cells."""
        if isinstance(item, (PathPaintItem, ImagePaintItem)) or item.kind in {"glyph", "shading"}:
            previous_shape_state = self.paint_alpha_is_shape, self.shape_alpha
            self.paint_alpha_is_shape = (
                item.alpha_is_shape
                if isinstance(item, (PathPaintItem, ImagePaintItem))
                else item.data.get("alpha_is_shape") is True
            )
            self.shape_alpha = 1.0
            knockout = self.buffer_stack[-1].knockout
            mask = internal_graphics_soft_mask(item)
            mask_alpha = internal_resolve_soft_mask(self, mask) if mask is not None else None
            fillstroke = (
                isinstance(item, PathPaintItem) and item.paint_kind is PathPaintKind.FILL_STROKE
            )
            if knockout or mask_alpha is not None:
                # An elementary object blends with the initial backdrop. Its
                # completed shape then replaces preceding group contributions.
                self.push_group(
                    bytearray(len(self.pixels)),
                    None,
                    None,
                    isolated=False,
                    track_shape=mask_alpha is not None,
                    # Combined fill/stroke inherits the mask on its children
                    # inside its implicit knockout group, including AIS shape.
                    mask_alpha=None if fillstroke else mask_alpha,
                    alpha_is_shape=self.paint_alpha_is_shape,
                )
            try:
                self.internal_paint_item(item)
            finally:
                if knockout or mask_alpha is not None:
                    self.composite_group(self.pop_group())
                self.paint_alpha_is_shape, self.shape_alpha = previous_shape_state
            return
        self.internal_paint_item(item)

    def internal_paint_item(self, item: DisplayItem) -> None:
        if isinstance(item, PathPaintItem):
            self.paint_typed_path(item)
            return
        if isinstance(item, ImagePaintItem):
            self.blit_image(item)
            return
        data = item.data
        blend_mode = data.get("blend_mode")
        if blend_mode == "Normal":
            blend_mode = None
        if item.kind == "scope-begin":
            path = data.get("path")
            self.push_scope(path if isinstance(path, CapturedPath) else None)
        elif item.kind == "scope-end":
            self.pop_scope()
        elif item.kind == "state-push":
            self.clip_stack.append(self.clip.depth)
        elif item.kind == "state-pop":
            if self.clip_stack:
                self.clip.restore(self.clip_stack.pop())
            else:
                self.clip.restore(self.clip_floor)
        elif item.kind == "clip":
            path = data.get("path")
            if isinstance(path, CapturedPath) and path.has_segments():
                self.clip.push(path, data.get("fill_rule") or "nonzero")
        elif item.kind == "group-begin":
            opacity = data.get("fill_opacity")
            mask = data.get("soft_mask_alpha")
            if is_pdf_number(mask):
                opacity = (float(opacity) if is_pdf_number(opacity) else 1.0) * float(mask)
            self.push_group(
                bytearray(self.width * self.height * 4),
                opacity,
                data.get("blend_mode"),
                isolated=data.get("group_isolated", True),
                knockout=data.get("group_knockout", False),
                alpha_is_shape=data.get("alpha_is_shape", False),
                track_shape=data.get("group_track_shape", False),
                mask_alpha=internal_resolve_soft_mask(self, graphics_mask)
                if isinstance(graphics_mask := data.get("graphics_soft_mask"), CapturedSoftMask)
                else None,
            )
        elif item.kind == "group-end" and len(self.buffer_stack) > self.group_floor:
            self.composite_group(self.pop_group())
        elif item.kind == "glyph" and data.get("visible") is not False:
            rgba = internal_color_rgba(data.get("fill_color"), data.get("fill_opacity"))
            if is_pdf_number(mask := data.get("soft_mask_alpha")):
                rgba = internal_scale_rgba_alpha(rgba, mask)
            self.shape_alpha = rgba[3] / 255.0 if self.paint_alpha_is_shape else 1.0
            self.draw_glyph_bitmap(
                data.get("bbox"),
                data.get("bitmap"),
                rgba,
                blend_mode,
                data.get("bitmap_width"),
                data.get("bitmap_height"),
            )
        elif item.kind == "shading":
            if self.paint_alpha_is_shape:
                opacity = data.get("fill_opacity")
                mask = data.get("soft_mask_alpha")
                self.shape_alpha = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (float(opacity) if is_pdf_number(opacity) else 1.0)
                            * (float(mask) if is_pdf_number(mask) else 1.0)
                        ),
                    ),
                )
            self.paint_shading(data, blend_mode)

    def push_group(
        self,
        buffer: bytearray,
        group_alpha: float | None,
        blend_mode: str | None,
        *,
        isolated: bool = True,
        knockout: bool = False,
        alpha_is_shape: bool = False,
        track_shape: bool = False,
        mask_alpha: SoftMaskPlane | None = None,
    ) -> None:
        parent = self.buffer_stack[-1]
        # ISO 32000-2 11.4.6: a non-isolated child of a knockout group
        # inherits the parent's initial backdrop, not its preceding elements.
        backdrop = None if isolated else parent.backdrop if parent.knockout else self.pixels
        source_alpha = None
        if backdrop is not None:
            buffer[:] = backdrop
        if backdrop is not None or knockout:
            source_alpha = numpy.zeros((self.height, self.width), dtype=numpy.float32)
        source_shape = (
            numpy.zeros((self.height, self.width), dtype=numpy.float32)
            if knockout or track_shape or parent.source_shape is not None
            else None
        )
        self.buffer_stack.append(
            internal_RasterGroup(
                buffer,
                group_alpha,
                blend_mode,
                backdrop=backdrop,
                source_alpha=source_alpha,
                source_shape=source_shape,
                knockout=knockout,
                alpha_is_shape=alpha_is_shape,
                mask_alpha=mask_alpha,
            )
        )
        self.pixels = buffer
        self.group_source_alpha = source_alpha
        self.group_source_shape = source_shape

    def pop_group(self) -> internal_RasterGroup:
        child = self.buffer_stack.pop()
        self.pixels = self.buffer_stack[-1].pixels
        self.group_source_alpha = self.buffer_stack[-1].source_alpha
        self.group_source_shape = self.buffer_stack[-1].source_shape
        return child

    def record_source_alpha(
        self,
        rows: int | slice,
        columns: int | slice,
        alpha: int | UInt8Array,
        *,
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
    ) -> None:
        """Accumulate only paint alpha, excluding the group's initial backdrop."""
        plane = self.group_source_alpha
        if plane is None:
            return
        previous = plane[rows, columns]
        source = alpha / 255.0
        updated = previous + (1.0 - previous) * source
        plane[rows, columns] = (
            updated if visible is None else numpy.where(visible, updated, previous)
        )

    def pixel_view(self, buffer: bytearray | bytes) -> UInt8Array:
        """Return an array view for an RGBA byte buffer."""
        return uint8_image_view(buffer, (self.height, self.width, 4))

    def record_source_shape(
        self,
        rows: int | slice,
        columns: int | slice,
        shape: int | UInt8Array,
        *,
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
    ) -> None:
        """Union geometric coverage independently of opacity, including zero alpha."""
        plane = self.group_source_shape
        if plane is None:
            return
        previous = plane[rows, columns]
        source = shape / 255.0 * self.shape_alpha
        updated = previous + (1.0 - previous) * source
        plane[rows, columns] = (
            updated if visible is None else numpy.where(visible, updated, previous)
        )

    def internal_resolved_blend(self, blend_mode: str | None) -> str | None:
        """Normalize object blend state without consulting group composite state."""
        return blend_mode.lower() if isinstance(blend_mode, str) else None

    def blend_px(
        self,
        idx: int,
        rgba: tuple[int, int, int, int],
        mode: str | None,
        *,
        shape: int = 255,
    ) -> None:
        """Blend object paint into the current buffer at its own opacity."""
        pixels = self.pixels
        sr, sg, sb, sa = rgba
        if self.group_source_shape is not None:
            row, column = divmod(idx // 4, self.width)
            self.record_source_shape(row, column, shape)
        if sa <= 0:
            return
        if self.group_source_alpha is not None:
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
        if mode == "multiply":
            src_r = src_r * (1.0 - dst_a) + dst_a * (src_r * dst_r)
            src_g = src_g * (1.0 - dst_a) + dst_a * (src_g * dst_g)
            src_b = src_b * (1.0 - dst_a) + dst_a * (src_b * dst_b)
        elif mode == "screen":
            src_r = src_r * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_r) * (1.0 - dst_r))
            src_g = src_g * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_g) * (1.0 - dst_g))
            src_b = src_b * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - src_b) * (1.0 - dst_b))
        elif mode in {"colordodge", "colorburn"}:
            component_mode: BlendMode = "ColorDodge" if mode == "colordodge" else "ColorBurn"
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
        return blend_mode is None

    def blend_normal_pixel(
        self, idx: int, sr: int, sg: int, sb: int, sa: int, *, shape: int = 255
    ) -> None:
        if self.group_source_shape is not None:
            row, column = divmod(idx // 4, self.width)
            self.record_source_shape(row, column, shape)
        if sa <= 0:
            return
        if self.group_source_alpha is not None:
            row, column = divmod(idx // 4, self.width)
            self.record_source_alpha(row, column, sa)
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
        self, row: int, start: int, end: int, rgba: tuple[int, int, int, int], *, shape: int = 255
    ) -> None:
        sr, sg, sb, sa = rgba
        if end <= start:
            return
        self.record_source_shape(row // (self.width * 4), slice(start, end), shape)
        if sa <= 0:
            return
        self.record_source_alpha(row // (self.width * 4), slice(start, end), sa)
        pixels = self.pixels
        width = self.width
        if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
            target = self.pixel_view(pixels)
            internal_blend_normal_solid_array_numpy(target[row // (width * 4), start:end], rgba)
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

    def composite_group(self, group: internal_RasterGroup) -> None:
        parent = self.buffer_stack[-1]
        source_scale = (
            max(0.0, min(1.0, float(group.composite_alpha)))
            if is_pdf_number(group.composite_alpha)
            else 1.0
        )
        shape = group.source_shape
        if shape is not None and group.alpha_is_shape:
            shape = shape * source_scale
            if group.mask_alpha is not None:
                shape = shape * group.mask_alpha
        if parent.knockout:
            assert parent.source_alpha is not None
            assert shape is not None
            initial = parent.backdrop or bytearray(len(parent.pixels))
            element = bytearray(initial)
            effective_alpha = self.internal_composite_group_into(group, element)
            internal_composite_knockout_group(
                self.pixel_view(parent.pixels),
                self.pixel_view(initial),
                self.pixel_view(element),
                parent.source_alpha,
                effective_alpha,
                shape,
            )
        else:
            effective_alpha = self.internal_composite_group_into(group, self.pixels)
            self.record_source_alpha(slice(None), slice(None), effective_alpha)
        if shape is not None and parent.source_shape is not None:
            # Child shape is complete, including its own AIS constant. Do not
            # apply the currently executing outer paint's shape constant twice.
            parent.source_shape[:] += (1.0 - parent.source_shape) * shape

    def internal_composite_group_into(
        self, group: internal_RasterGroup, destination: bytearray
    ) -> UInt8Array:
        """Apply one group's outer state, returning its effective source alpha."""
        child = group.pixels
        group_alpha = group.composite_alpha
        group_blend_mode = group.blend_mode
        normalized_blend_mode = (
            group_blend_mode.casefold() if isinstance(group_blend_mode, str) else None
        )
        source_scale = max(0.0, min(1.0, float(group_alpha))) if is_pdf_number(group_alpha) else 1.0
        if group.mask_alpha is not None:
            return internal_composite_masked_group(
                self.pixel_view(destination),
                self.pixel_view(child),
                group.source_alpha if group.backdrop is not None else None,
                source_scale,
                group_blend_mode,
                group.mask_alpha,
                semantic_context=self.semantic_context,
            )
        if group.backdrop is not None:
            assert group.source_alpha is not None
            return internal_composite_nonisolated_group(
                self.pixel_view(destination),
                self.pixel_view(child),
                group.source_alpha,
                source_scale,
                group_blend_mode,
                semantic_context=self.semantic_context,
            )
        effective_alpha = numpy.clip(
            numpy.rint(self.pixel_view(child)[..., 3].astype(numpy.float64) * source_scale),
            0.0,
            255.0,
        ).astype(numpy.uint8)
        if normalized_blend_mode in {None, "normal"} and len(child) >= 4_096:
            source_pixels = self.pixel_view(child)
            target_pixels = self.pixel_view(destination)
            internal_composite_normal_group_numpy(
                target_pixels,
                source_pixels,
                source_scale,
            )
            return effective_alpha
        # The parent retains its own composite opacity until it is closed.
        internal_composite_blended_group_numpy(
            self.pixel_view(destination),
            self.pixel_view(child),
            float(group_alpha) if is_pdf_number(group_alpha) else None,
            None,
            group_blend_mode,
            semantic_context=self.semantic_context,
        )
        return effective_alpha

    def paint_typed_path(self, item: PathPaintItem) -> None:
        path = item.path
        if type(path) is not CapturedPath:
            return
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        soft_mask_alpha = item.soft_mask_alpha
        paint_kind = item.paint_kind
        if paint_kind is PathPaintKind.FILL_STROKE and self.group_source_shape is not None:
            # ISO 32000-2 11.7.4.4: the fill and stroke are one outer object,
            # with an implicit knockout group preventing a doubled border.
            self.push_group(bytearray(len(self.pixels)), None, None, isolated=False, knockout=True)
            try:
                self.paint_item(replace(item, paint_kind=PathPaintKind.FILL))
                self.paint_item(replace(item, paint_kind=PathPaintKind.STROKE))
            finally:
                self.composite_group(self.pop_group())
            return
        if paint_kind is not PathPaintKind.STROKE:
            rgba = internal_color_rgba(item.fill, item.fill_opacity)
            if is_pdf_number(soft_mask_alpha):
                rgba = internal_scale_rgba_alpha(rgba, soft_mask_alpha)
            self.shape_alpha = rgba[3] / 255.0 if self.paint_alpha_is_shape else 1.0
            if item.fill_pattern is None or not self.paint_fill_pattern(item, blend_mode):
                self.fill_path(path, rgba, blend_mode, item.fill_rule)
        if paint_kind is not PathPaintKind.FILL:
            stroke_rgba = internal_color_rgba(item.stroke_color, item.stroke_opacity)
            if is_pdf_number(soft_mask_alpha):
                stroke_rgba = internal_scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
            self.shape_alpha = stroke_rgba[3] / 255.0 if self.paint_alpha_is_shape else 1.0
            if self.group_source_shape is not None:
                internal_paint_stroke_once(
                    self,
                    path,
                    item.line_width,
                    stroke_rgba,
                    item.dash_pattern,
                    blend_mode,
                    item.line_cap,
                    item.line_join,
                )
                return
            self.stroke_path(
                path,
                item.line_width,
                stroke_rgba,
                item.dash_pattern,
                blend_mode,
                item.line_cap,
                item.line_join,
            )
