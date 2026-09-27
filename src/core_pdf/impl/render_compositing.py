# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.array_views import UInt8Array
from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_blend import (
    BlendOp,
    blend_op,
    blend_visible_pixels,
    composite_blended_group_numpy,
    declared_blend,
)
from core_pdf.impl.render_coverage import RasterCoverage
from core_pdf.impl.render_model import (
    DisplayItem,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
    SoftMaskPlane,
    is_plain_fill,
)
from core_pdf.impl.render_raster_state import EMPTY_PIXEL_BOX, PixelBox
from core_pdf.impl.scalars import clamp01
from core_pdf_cythonized import (
    composite_elementary_knockout,
    composite_elementary_normal,
    composite_knockout_group,
    composite_masked_normal,
    composite_normal_group,
    fill_glyph_knockout,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_11_transparency.groups import remove_group_backdrop
from core_pdf_spec.standards import SemanticContext


def composite_nonisolated_group(
    destination: UInt8Array,
    rendered: UInt8Array,
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    opacity: float,
    blend_mode: str | None,
    *,
    semantic_context: SemanticContext,
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None = None,
) -> UInt8Array:
    opacity = clamp01(opacity)
    mode = blend_op(blend_mode, casefold=True)
    normal = mode is None or mode is BlendOp.NORMAL
    if opacity == 1.0 and normal and mask_alpha is None:
        return composite_elementary_normal(destination, rendered, source_alpha)
    scaled_alpha = source_alpha.astype(numpy.float64) * opacity * 255.0
    if mask_alpha is not None:
        scaled_alpha *= mask_alpha
    effective_alpha = numpy.rint(scaled_alpha).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    if opacity == 1.0 and normal:
        unchanged_alpha = visible & (mask_alpha == 1.0)
        destination[unchanged_alpha] = rendered[unchanged_alpha]
        visible &= ~unchanged_alpha
        if not numpy.any(visible):
            return effective_alpha
    backdrop = destination[visible].astype(numpy.float64)
    result = rendered[visible].astype(numpy.float64) / 255.0
    colors, _ = remove_group_backdrop(
        result[..., :3],
        result[..., 3],
        backdrop[..., :3] / 255.0,
        backdrop[..., 3] / 255.0,
        source_alpha[visible],
        validate=False,
    )
    colors = numpy.clip(colors, 0.0, 1.0)
    blend_visible_pixels(
        destination,
        visible,
        colors[..., 0],
        colors[..., 1],
        colors[..., 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        mode,
        semantic_context=semantic_context,
    )
    return effective_alpha


def composite_masked_group(
    destination: UInt8Array,
    rendered: UInt8Array,
    source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None,
    opacity: float,
    blend_mode: str | None,
    mask: SoftMaskPlane,
    window: tuple[slice, slice],
    *,
    semantic_context: SemanticContext,
) -> UInt8Array:
    mode = blend_op(blend_mode, casefold=True)
    if source_alpha is None and (mode is None or mode is BlendOp.NORMAL):
        return composite_masked_normal(
            destination, rendered, opacity, mask.alpha[window], mask.values()
        )
    mask_alpha = mask[window]
    if source_alpha is not None:
        return composite_nonisolated_group(
            destination,
            rendered,
            source_alpha,
            opacity,
            blend_mode,
            semantic_context=semantic_context,
            mask_alpha=mask_alpha,
        )
    effective_alpha = numpy.clip(
        numpy.rint(rendered[..., 3].astype(numpy.float64) * opacity * mask_alpha), 0, 255
    ).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    colors = rendered[visible, :3].astype(numpy.float64) / 255.0
    blend_visible_pixels(
        destination,
        visible,
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
        effective_alpha[visible].astype(numpy.float64) / 255.0,
        mode,
        semantic_context=semantic_context,
    )
    return effective_alpha


class RasterCompositing(RasterCoverage):
    __slots__ = ()

    KNOCKOUT_DISJOINT_LIMIT = 96

    def knockout_glyph_fill(self, item: DisplayItem) -> bool:
        if type(item) is not PathPaintItem or item.paint_kind is not PathPaintKind.FILL:
            return False
        edge_array = item.edge_array
        bbox = item.bbox
        if (
            type(item.path) is not CapturedPath
            or edge_array is None
            or bbox is None
            or item.fill_pattern is not None
            or declared_blend(item.blend_mode) is not None
            or item.fill_rule != "nonzero"
        ):
            return False
        parent = self.buffer_stack[-1]
        if parent.backdrop is None or parent.source_alpha is None or parent.source_shape is None:
            return False
        if item.path.axis_aligned_rect() is not None:
            return False
        rgba = item.fill_rgba()
        if len(edge_array) == 0:
            return True
        clipped = self.clip.clipped_pixel_box(bbox)
        if clipped is None:
            return True
        ix0, iy0, ix1, iy1 = clipped[1]
        if not (
            (ix1 - ix0) * (iy1 - iy0) < 10_000 and self.clip.clip_paths_are_axis_aligned_rects()
        ):
            return False
        self.set_shape_alpha(rgba[3] / 255.0)
        rows = slice(iy0, iy1)
        columns = slice(ix0, ix1)
        drawn = fill_glyph_knockout(
            edge_array,
            self.crop_x0,
            self.crop_y1,
            self.scale,
            ix0,
            iy0,
            ix1 - ix0,
            iy1 - iy0,
            rgba,
            parent.view[rows, columns],
            self.pixel_view(parent.backdrop)[rows, columns],
            parent.source_alpha[rows, columns],
            parent.source_shape[rows, columns],
            self.shape_alpha,
        )
        if drawn is None:
            return True
        if parent.painted_boxes is not None:
            self.record_knockout_paint((ix0, iy0, ix1, iy1))
        self.extend_paint_window(rows, columns)
        return True

    def knockout_group_region(self, item: DisplayItem) -> PixelBox | None:
        boxes = self.group_member_boxes
        if boxes is None:
            return None
        key = id(item)
        if key not in boxes:
            return None
        bbox = boxes[key]
        clipped = self.clip.clipped_pixel_box(bbox) if bbox is not None else None
        return EMPTY_PIXEL_BOX if clipped is None else clipped[1]

    def knockout_paint_box(self, item: DisplayItem) -> PixelBox | None:
        if not is_plain_fill(item):
            return None
        clipped = self.clip.clipped_pixel_box(item.bbox)
        if clipped is None:
            return EMPTY_PIXEL_BOX
        return clipped[1]

    def record_knockout_paint(self, box: PixelBox) -> None:
        boxes = self.buffer_stack[-1].painted_boxes
        if boxes is None or box == EMPTY_PIXEL_BOX:
            return
        if len(boxes) >= self.KNOCKOUT_DISJOINT_LIMIT:
            boxes.clear()
            boxes.append((0, 0, self.width, self.height))
        else:
            boxes.append(box)

    def knockout_skip_box(self, item: DisplayItem, boxes: list[PixelBox]) -> PixelBox | None:
        box = self.knockout_paint_box(item)
        if box is None or box == EMPTY_PIXEL_BOX:
            return box
        x0, y0, x1, y1 = box
        for bx0, by0, bx1, by1 in boxes:
            if x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1:
                return None
        return box

    def group_window(self, group: RasterGroup) -> tuple[slice, slice] | None:
        return group.paint_window.slices()

    def composite_group(self, group: RasterGroup) -> None:
        parent = self.buffer_stack[-1]
        window = self.group_window(group)
        if window is None:
            return
        rows, columns = window
        if parent.painted_boxes is not None:
            self.record_knockout_paint((columns.start, rows.start, columns.stop, rows.stop))
        shape = group.source_shape[rows, columns] if group.source_shape is not None else None
        if shape is not None and group.alpha_is_shape:
            shape = shape * group.source_scale
            if group.mask_alpha is not None:
                shape = shape * group.mask_alpha[rows, columns]
        if (
            parent.knockout
            and parent.backdrop is not None
            and group.backdrop is not None
            and group.mask_alpha is None
            and group.source_alpha is not None
            and group.source_scale == 1.0
            and (
                group.blend_mode is None
                or blend_op(group.blend_mode, casefold=True) is BlendOp.NORMAL
            )
        ):
            assert parent.source_alpha is not None
            assert shape is not None
            composite_elementary_knockout(
                parent.view[rows, columns],
                self.pixel_view(parent.backdrop)[rows, columns],
                group.view[rows, columns],
                group.source_alpha[rows, columns],
                parent.source_alpha[rows, columns],
                shape,
                parent.source_shape[rows, columns] if parent.source_shape is not None else None,
            )
            self.extend_paint_window(rows, columns)
            return
        if parent.knockout:
            assert parent.source_alpha is not None
            assert shape is not None
            initial = (
                self.pixel_view(parent.backdrop)[rows, columns]
                if parent.backdrop is not None
                else numpy.zeros((*shape.shape, 4), dtype=numpy.uint8)
            )
            element = initial.copy()
            effective_alpha = self.composite_group_into(group, element, rows, columns)
            composite_knockout_group(
                parent.view[rows, columns],
                initial,
                element,
                parent.source_alpha[rows, columns],
                effective_alpha,
                shape,
            )
        else:
            effective_alpha = self.composite_group_into(
                group, self.pixel_array[rows, columns], rows, columns
            )
            self.record_source_alpha(rows, columns, effective_alpha)
        if shape is not None and parent.source_shape is not None:
            parent_shape = parent.source_shape[rows, columns]
            parent_shape += (1.0 - parent_shape) * shape
        self.extend_paint_window(rows, columns)

    def composite_group_into(
        self,
        group: RasterGroup,
        destination: UInt8Array,
        rows: slice,
        columns: slice,
    ) -> UInt8Array:
        child = group.view[rows, columns]
        group_alpha = group.composite_alpha
        group_blend_mode = group.blend_mode
        normalized_blend_mode = blend_op(group_blend_mode, casefold=True)
        source_scale = group.source_scale
        if group.mask_alpha is not None:
            return composite_masked_group(
                destination,
                child,
                group.source_alpha[rows, columns]
                if group.backdrop is not None and group.source_alpha is not None
                else None,
                source_scale,
                group_blend_mode,
                group.mask_alpha,
                (rows, columns),
                semantic_context=self.semantic_context,
            )
        if group.backdrop is not None:
            assert group.source_alpha is not None
            return composite_nonisolated_group(
                destination,
                child,
                group.source_alpha[rows, columns],
                source_scale,
                group_blend_mode,
                semantic_context=self.semantic_context,
            )
        if (normalized_blend_mode is None or normalized_blend_mode is BlendOp.NORMAL) and len(
            group.pixels
        ) >= 4_096:
            plane = composite_normal_group(destination, child, source_scale, 1.0, True)
            assert plane is not None
            return plane
        effective_alpha = numpy.clip(
            numpy.rint(child[..., 3].astype(numpy.float64) * source_scale), 0.0, 255.0
        ).astype(numpy.uint8)
        composite_blended_group_numpy(
            destination,
            child,
            float(group_alpha) if is_pdf_number(group_alpha) else None,
            None,
            group_blend_mode,
            semantic_context=self.semantic_context,
        )
        return effective_alpha
