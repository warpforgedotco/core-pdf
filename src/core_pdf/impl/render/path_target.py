# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_pdf.impl.render.blend import (
    internal_color_rgba,
    internal_scale_rgba_alpha,
)
from core_pdf.impl.render.kernels import RASTER_COORDINATE_CACHE_MAX_ENTRIES
from core_pdf.impl.render.model import PathPaintItem, PathPaintKind
from core_pdf.impl.render.path_fill_target import internal_PathFillTargetMixin
from core_pdf.impl.render.path_shape_target import internal_PathShapeTargetMixin
from core_pdf.impl.render.path_stroke_target import internal_PathStrokeTargetMixin
from core_pdf.impl.spec.s_07_content.capture import CapturedPath
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_number


class internal_PathTargetMixin(
    internal_PathShapeTargetMixin,
    internal_PathFillTargetMixin,
    internal_PathStrokeTargetMixin,
):
    """Path-painting facade mixed into the mutable raster target."""

    __slots__ = ()

    def paint_typed_path(self: Any, item: PathPaintItem) -> None:
        # Captured frame values hoisted into locals so the body below runs on
        # LOAD_FAST exactly as it did when this was a closure.
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
