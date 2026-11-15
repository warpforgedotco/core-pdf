# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: shadings, and tiling and shading patterns.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import replace
from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.base.geometry import clamp01, normalize_rect, rect_tuple
from core_pdf.impl.capture.records import CapturedPath, ShadingPattern, TilingPattern
from core_pdf.impl.graphics.shading import PreparedShading, SpaceEvaluator
from core_pdf.impl.render.blend import kernel_blend_code
from core_pdf.impl.render.clipping import ClipState
from core_pdf.impl.render.model import (
    DisplayItem,
    PathPaintItem,
    PathPaintKind,
    ShadingItem,
    SoftMaskPlane,
    SpaceMode,
)
from core_pdf.impl.render.paths import intersect_box
from core_pdf.impl.render.raster_support import (
    cell_paints_nothing,
    shading_rgba,
    tiling_cell,
    tiling_pattern_uses_normal_blends,
)
from core_pdf.impl.render.raster_support import prepared_shading as cached_shading
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf.impl.spec.s_08_graphics.color_rendering import ColorRendering
from core_pdf.impl.spec.s_08_graphics.geometry import transform_bbox
from core_pdf.impl.spec.s_08_graphics.matrix import invert_affine
lazy from core_pdf_cythonized import composite_normal_group, shading_blend, shading_values

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

# Cells one pattern paints at most, so a tiny cell over a large area stays bounded.
MAX_TILING_CELLS = 10000
# Cells from which a pattern paints one cell and copies its pixels to the rest.
MIN_STAMPED_CELLS = 16


def shading_box(
    self: RasterTarget, item: ShadingItem, shading: PreparedShading
) -> tuple[float, float, float, float]:
    box = shading.bbox
    if box is not None and item.matrix is not None:
        box = transform_bbox(normalize_rect(box), item.matrix)
    if box is None:
        box = rect_tuple(item.bbox)
    if box is None:
        box = self.grid.page_box()
    return normalize_rect(box)


def prepared_shading(
    self: RasterTarget, dictionary: object, rendering: ColorRendering
) -> PreparedShading | None:
    return cached_shading(self.resources, dictionary, rendering)


@cites(
    "ISO 32000-2",
    "8.7.4.5.3",
    introduced="1.3",
    rule="axial and radial shadings are computed in shading space, clipped by BBox there",
)
def paint_shading(self: RasterTarget, item: ShadingItem, blend_mode: str | None) -> None:
    shading = self.prepared_shading(item.dictionary, item.color_rendering)
    if shading is None:
        return
    # A shading under a transform is computed in its own space: each pixel centre goes back
    # through the inverse, and the BBox clips there (8.7.4.2, 8.7.4.3).
    to_shading = None
    shading_bbox = None
    if item.matrix is not None:
        to_shading = invert_affine(item.matrix)
        if to_shading is None:
            return
        if shading.bbox is not None:
            shading_bbox = normalize_rect(shading.bbox)
    clipped_box = self.clip.clipped_pixel_box(self.shading_box(item, shading))
    if clipped_box is None:
        return
    ix0, iy0, ix1, iy1 = clipped_box[1]
    soft_mask_alpha = item.soft_mask_alpha
    fill_opacity = item.fill_opacity
    shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
    mode = kernel_blend_code(self.resolved_blend(blend_mode))
    domain = shading.domain
    allowed = numpy.frombuffer(self.clip_pixel_mask(ix0, iy0, ix1, iy1), dtype=numpy.uint8).reshape(
        iy1 - iy0, ix1 - ix0
    )
    values, painted = shading_values(
        shading.shading_type,
        shading.coords,
        self.crop_x0,
        self.crop_y1,
        self.scale,
        ix0,
        iy0,
        ix1,
        iy1,
        allowed,
        bool(shading.extend_start),
        bool(shading.extend_end),
        domain[0],
        domain[1] - domain[0],
        0.5,
        to_shading,
        shading_bbox,
    )
    ordered = values[painted.view(numpy.bool_)]
    if not len(ordered):
        return
    _, first, inverse = numpy.unique(ordered, return_index=True, return_inverse=True)
    by_first = numpy.argsort(first, kind="stable")
    colors = numpy.zeros((len(first), 4), dtype=numpy.int32)
    reached = len(ordered)
    color_error: Exception | None = None
    color_model = shading.color_model
    evaluate = shading.evaluator
    rendering = shading.color_rendering
    in_space = self.space_mode is not SpaceMode.RGB
    order = by_first.tolist()
    # A colour space other than a device one converts every colour in one pass.
    batch = (
        evaluate.many(ordered[first[by_first]].tolist())
        if isinstance(evaluate, SpaceEvaluator)
        else None
    )
    for step, unique_index in enumerate(order):
        position = int(first[unique_index])
        try:
            components = batch[step] if batch is not None else evaluate(float(ordered[position]))
            rgba = shading_rgba(color_model, components, fill_opacity, rendering, shading_alpha)
            if in_space:
                rgba = self.shading_space_color(color_model, components, rgba)
            colors[unique_index] = rgba
        except Exception as error:
            color_error = error
            reached = position
            break
    revised, blend_error = self.blend_rules(mode)
    shape_plane = self.group_source_shape
    window, stopped = shading_blend(
        self.pixel_array,
        ix0,
        iy0,
        painted,
        numpy.ascontiguousarray(inverse.reshape(-1)[:reached], dtype=numpy.int64),
        colors,
        mode,
        revised,
        self.group_source_alpha,
        shape_plane,
        255 / 255.0 * self.shape_alpha if shape_plane is not None else 0.0,
        blend_error is not None,
    )
    if window is not None and self.paint_window is not None:
        self.paint_window.extend_box(window)
    if stopped:
        assert blend_error is not None
        raise blend_error
    if color_error is not None:
        raise color_error


def paint_tiling_pattern(
    self: RasterTarget, pattern: TilingPattern, target_data: PathPaintItem, blend_mode: str | None
) -> bool:
    scale = self.scale
    cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
    x_step = abs(pattern.x_step)
    y_step = abs(pattern.y_step)
    if x_step <= 0.0 or y_step <= 0.0:
        return False
    program = pattern.program
    if not program.drawings and not program.glyphs and not program.inline_images:
        return False
    display, cell_clip = tiling_cell(
        self.resources,
        pattern,
        self.width,
        self.height,
        preserve_object_boundaries=self.group_source_shape is not None,
    )
    if cell_paints_nothing(display.items, cell_clip, scale):
        return True
    target_box = target_data.bbox or self.clip.path_bbox(target_data.path)
    if (type(target_box) is list or type(target_box) is tuple) and len(target_box) == 4:
        box = rect_tuple(target_box)
        if box is None:
            return False
    else:
        box = self.grid.page_box()
    x0, y0, x1, y1 = box
    clip_box = self.clip.current_clip()
    if clip_box is not None:
        clipped = intersect_box((x0, y0, x1, y1), clip_box)
        if clipped is None:
            return True
        x0, y0, x1, y1 = clipped
    opacity = target_data.fill_opacity
    alpha = clamp01(opacity) if is_pdf_number(opacity) else 1.0
    if is_pdf_number(target_data.soft_mask_alpha):
        alpha *= clamp01(target_data.soft_mask_alpha)
    matrix = pattern.matrix
    offsets = None
    if matrix is not None:
        offsets = tiling_offsets(pattern, matrix, cell_clip, (x0, y0, x1, y1))
        if offsets is None:
            return True
    self.push_group(
        bytearray(len(self.pixels)),
        alpha,
        blend_mode,
        isolated=tiling_pattern_uses_normal_blends(pattern),
        alpha_is_shape=target_data.alpha_is_shape,
    )
    try:
        if offsets is not None:
            if len(offsets) < MIN_STAMPED_CELLS or not self.stamp_tiling_cells(
                display.items, cell_clip, offsets
            ):
                for translation in offsets:
                    self.paint_items(
                        display.items,
                        translation=translation,
                        parent_blend_mode=None,
                        clip_path=cell_clip,
                    )
            return True
        start_x = cell_x0 + math.floor((x0 - cell_x0) / x_step) * x_step
        start_y = cell_y0 + math.floor((y0 - cell_y0) / y_step) * y_step
        cells = 0
        y = start_y
        while y < y1 + y_step and cells < MAX_TILING_CELLS:
            x = start_x
            while x < x1 + x_step and cells < MAX_TILING_CELLS:
                tx = x - cell_x0
                ty = y - cell_y0
                if x + (cell_x1 - cell_x0) >= x0 and y + (cell_y1 - cell_y0) >= y0:
                    self.paint_items(
                        display.items,
                        translation=(tx, ty),
                        parent_blend_mode=None,
                        clip_path=cell_clip,
                    )
                cells += 1
                x += x_step
            y += y_step
    finally:
        self.composite_group(self.pop_group())
    return True


@cites(
    "ISO 32000-2",
    "8.7.3.1",
    introduced="1.2",
    rule="a tiling's cells may be spaced by whole device pixels, as TilingType 1 to 3 allow",
)
def stamp_tiling_cells(
    self: RasterTarget,
    items: Sequence[DisplayItem],
    cell_clip: CapturedPath,
    offsets: list[tuple[float, float]],
) -> bool:
    """Paint one cell, then copy its pixels to every other cell, each moved by a whole number
    of device pixels: no cell is distorted, and no spacing is off by more than a pixel, which
    every TilingType allows (8.7.3.1, Table 74). False, having painted nothing, when a cell
    cannot be copied: one cell must fit the page whole, and the target must composite in RGB
    with Normal and without tracking shape or alpha."""
    if (
        self.space_mode is not SpaceMode.RGB
        or self.group_source_shape is not None
        or self.group_source_alpha is not None
    ):
        return False
    clip_box = cell_clip.bbox()
    if clip_box is None:
        return False
    scale = self.scale
    crop_x0 = self.crop_x0
    crop_y1 = self.crop_y1
    width = self.width
    height = self.height
    left, bottom, right, top = clip_box
    reference = None
    for tx, ty in offsets:
        # The cell's pixels, with one to spare for antialiasing.
        ix0 = math.floor((left + tx - crop_x0) * scale) - 1
        ix1 = math.ceil((right + tx - crop_x0) * scale) + 1
        iy0 = math.floor((crop_y1 - top - ty) * scale) - 1
        iy1 = math.ceil((crop_y1 - bottom - ty) * scale) + 1
        if ix0 >= 0 and iy0 >= 0 and ix1 <= width and iy1 <= height:
            reference = (tx, ty, ix0, iy0, ix1, iy1)
            break
    if reference is None:
        return False
    ref_x, ref_y, ix0, iy0, ix1, iy1 = reference
    cell, view = self.blank_sibling()
    cell.paint_items(items, translation=(ref_x, ref_y), parent_blend_mode=None, clip_path=cell_clip)
    tile = view[iy0:iy1, ix0:ix1]
    region = self.clip.current_region()
    if region is not None and region.empty:
        return True
    bounds = (0, 0, width, height) if region is None else region.pixel_box
    if bounds is None:
        return True
    bx0, by0, bx1, by1 = bounds
    shaped = region is not None and not region.rectangular
    destination = self.pixel_array
    for tx, ty in offsets:
        dx = round((tx - ref_x) * scale)
        dy = round((ref_y - ty) * scale)
        x0 = max(ix0 + dx, bx0)
        y0 = max(iy0 + dy, by0)
        x1 = min(ix1 + dx, bx1)
        y1 = min(iy1 + dy, by1)
        if x1 <= x0 or y1 <= y0:
            continue
        source = tile[y0 - iy0 - dy : y1 - iy0 - dy, x0 - ix0 - dx : x1 - ix0 - dx]
        if shaped:
            allowed = numpy.frombuffer(self.clip_pixel_mask(x0, y0, x1, y1), dtype=numpy.uint8)
            source = source.copy()
            source[..., 3] *= allowed.reshape(y1 - y0, x1 - x0)
        composite_normal_group(destination[y0:y1, x0:x1], source, 1.0)
        self.extend_paint_window(slice(y0, y1), slice(x0, x1))
    return True


@cites(
    "ISO 32000-2",
    "8.7.3.1",
    introduced="1.2",
    rule="tiling cells sit at multiples of XStep and YStep in pattern space, clipped by BBox there",
)
def tiling_offsets(
    pattern: TilingPattern,
    matrix: tuple[float, float, float, float, float, float],
    cell_clip: CapturedPath,
    box: tuple[float, float, float, float],
) -> list[tuple[float, float]] | None:
    """The page offsets of the cells of a pattern under matrix that reach box, or None when
    the matrix is singular. Cell (i, j) is the key cell moved by (i XStep, j YStep) in pattern
    space, which on the page is the same move through the matrix's linear part."""
    inverse = invert_affine(matrix)
    clip_box = cell_clip.bbox()
    if inverse is None or clip_box is None:
        return None
    a, b, c, d, _, _ = matrix
    x_step = abs(pattern.x_step)
    y_step = abs(pattern.y_step)
    bx0, by0, bx1, by1 = pattern.bbox
    cell_x0, cell_x1 = min(bx0, bx1), max(bx0, bx1)
    cell_y0, cell_y1 = min(by0, by1), max(by0, by1)
    # The box in pattern space bounds the steps whose cell can reach it.
    px0, py0, px1, py1 = transform_bbox(box, inverse)
    if not all(math.isfinite(value) for value in (px0, py0, px1, py1)):
        return None
    first_i = math.ceil((px0 - cell_x1) / x_step)
    last_i = math.floor((px1 - cell_x0) / x_step)
    first_j = math.ceil((py0 - cell_y1) / y_step)
    last_j = math.floor((py1 - cell_y0) / y_step)
    x0, y0, x1, y1 = box
    left, bottom, right, top = clip_box
    offsets: list[tuple[float, float]] = []
    visited = 0
    for j in range(first_j, last_j + 1):
        u = j * y_step
        for i in range(first_i, last_i + 1):
            visited += 1
            if visited > 4 * MAX_TILING_CELLS:
                return offsets
            v = i * x_step
            tx = a * v + c * u
            ty = b * v + d * u
            # A parallelogram cell whose page box misses the box paints nothing there.
            if left + tx > x1 or right + tx < x0 or bottom + ty > y1 or top + ty < y0:
                continue
            offsets.append((tx, ty))
            if len(offsets) >= MAX_TILING_CELLS:
                return offsets
    return offsets


def paint_fill_pattern(self: RasterTarget, data: PathPaintItem, blend_mode: str | None) -> bool:
    clip_state = self.clip
    pattern = data.fill_pattern
    if not isinstance(pattern, (ShadingPattern, TilingPattern)):
        return False
    path = data.path
    pushed_clip = False
    if type(path) is CapturedPath and path.has_segments():
        clip_state.push(path, data.fill_rule or "nonzero")
        pushed_clip = True
    try:
        if isinstance(pattern, ShadingPattern):
            dictionary = pattern.dictionary
            shading_data = {
                "dictionary": dictionary,
                "bbox": data.bbox or clip_state.path_bbox(path),
                "fill_opacity": data.fill_opacity,
                "soft_mask_alpha": data.soft_mask_alpha,
                "color_rendering": pattern.color_rendering,
            }
            if pattern.matrix is not None:
                shading_data["matrix"] = pattern.matrix
            self.paint_shading(ShadingItem.from_data(data.seqno, shading_data), blend_mode)
            return True
        return self.paint_tiling_pattern(pattern, data, blend_mode)
    finally:
        if pushed_clip:
            clip_state.pop()


@cites(
    "ISO 32000-2",
    "8.7.3.1",
    introduced="1.2",
    rule="S with a pattern as its colour paints the pattern through the stroke's outline",
)
def paint_stroke_pattern(self: RasterTarget, data: PathPaintItem, blend_mode: str | None) -> bool:
    """Paint a stroke whose colour is a pattern: the stroke's coverage is the mask of an
    isolated group the pattern fills over the stroke's box (8.7.1, 8.7.3.1)."""
    pattern = data.stroke_pattern
    path = data.path
    if not isinstance(pattern, (ShadingPattern, TilingPattern)) or type(path) is not CapturedPath:
        return False
    box = replace(data, paint_kind=PathPaintKind.STROKE).page_box(self.scale)
    if box is None:
        return True
    clip_box = self.clip.current_clip()
    if clip_box is not None and intersect_box(box, clip_box) is None:
        return True
    # The stroke's coverage, drawn opaque on a blank page of its own.
    coverage, view = self.blank_sibling()
    coverage.stroke_path(
        path,
        data.line_width,
        (255, 255, 255, 255),
        data.dash_pattern,
        None,
        data.line_cap,
        data.line_join,
    )
    self.push_group(
        bytearray(len(self.pixels)),
        None,
        blend_mode,
        isolated=True,
        alpha_is_shape=data.alpha_is_shape,
        mask_alpha=SoftMaskPlane(view[..., 3], None),
    )
    try:
        self.paint_pattern_over(pattern, data, box, data.stroke_opacity)
    finally:
        self.composite_group(self.pop_group())
    return True


def paint_pattern_over(
    self: RasterTarget,
    pattern: ShadingPattern | TilingPattern,
    data: PathPaintItem,
    box: tuple[float, float, float, float],
    opacity: float | None,
) -> None:
    """Paint pattern over box, in the Normal blend mode, at opacity and data's soft mask alpha."""
    if isinstance(pattern, ShadingPattern):
        shading_data: dict[str, Any] = {
            "dictionary": pattern.dictionary,
            "bbox": box,
            "fill_opacity": opacity,
            "soft_mask_alpha": data.soft_mask_alpha,
            "color_rendering": pattern.color_rendering,
        }
        if pattern.matrix is not None:
            shading_data["matrix"] = pattern.matrix
        self.paint_shading(ShadingItem.from_data(data.seqno, shading_data), None)
        return
    self.paint_tiling_pattern(pattern, replace(data, bbox=box, fill_opacity=opacity), None)


def blank_sibling(self: RasterTarget) -> tuple[RasterTarget, UInt8Array]:
    pixels = bytearray(self.width * self.height * 4)
    view = uint8_image_view(pixels, (self.height, self.width, 4))
    sibling = type(self)(
        pixels,
        None,
        clip=ClipState(self.grid),
        page_view=view,
        semantic_context=self.semantic_context,
        resources=self.resources,
    )
    return sibling, view
