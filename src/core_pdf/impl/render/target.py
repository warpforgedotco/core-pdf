# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import replace
from typing import Any

import numpy

from core_pdf.impl.array_views import (
    ByteBuffer,
    UInt8Array,
    uint8_image_view,
    uint8_view,
)
from core_pdf.impl.capture.records import (
    CapturedPath,
    CapturedSoftMask,
    ShadingPattern,
    TilingPattern,
)
from core_pdf.impl.geometry import normalize_rect, points_bbox, rect_tuple
from core_pdf.impl.graphics.images import PreparedImage, prepare_image
from core_pdf.impl.graphics.shading import PreparedShading, prepare_shading
from core_pdf.impl.graphics.soft_masks import image_color_key_mask_is_shape
from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    blend_context,
    blend_normal_solid_array_numpy,
    blend_solid_array_numpy,
    blend_visible_pixels,
    clamp01,
    color_rgba,
    composite_blended_group_numpy,
    composite_normal_group_numpy,
    resolve_constant_alpha,
    scale_rgba_alpha,
)
from core_pdf.impl.render.clipping import ClipState
from core_pdf.impl.render.commands import append_captured_program, translated_command
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import (
    DisplayItem,
    ImagePaintItem,
    LineCap,
    LineJoin,
    PathPaintItem,
    PathPaintKind,
    RasterGroup,
)
from core_pdf.impl.render.paths import (
    RASTER_CIRCLE_MIN_PIXEL_AREA,
    RASTER_KERNEL_MIN_PIXEL_AREA,
    RASTER_SAMPLE_OFFSETS,
    circle_path,
    dash_subpath,
    fill_path_crossing_spans,
    fill_path_sample_crossings,
    fill_path_sample_crossings_numpy,
    intersect_box,
    rasterize_unclipped_line_normal,
)
from core_pdf.impl.render.patterns import (
    TilingCellCache,
    axial_shading_t,
    radial_shading_t,
    shading_color_rgba,
    tiling_cell,
    tiling_pattern_uses_normal_blends,
)
from core_pdf.impl.scalars import parse_int
from core_pdf_cythonized import (
    blend_normal_alpha_array_numpy,
    composite_elementary_normal,
    composite_knockout_group,
    glyph_coverage_plane,
    rect_coverage_plane,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING
from core_pdf_spec.s_08_graphics.image_spec import ImageSource
from core_pdf_spec.s_11_transparency.blend import BlendMode, blend_component
from core_pdf_spec.s_11_transparency.groups import (
    remove_group_backdrop,
)
from core_pdf_spec.standards import SemanticContext


def index_extent(index: int | slice, size: int) -> tuple[int, int]:
    if isinstance(index, slice):
        start, stop, _ = index.indices(size)
        return start, max(start, stop)
    return index, index + 1


class ElementaryScratch:
    __slots__ = ("buffer", "dirty", "source_alpha", "source_shape", "synced_parent", "view")

    def __init__(self, size: int, height: int, width: int) -> None:
        self.buffer = bytearray(size)
        self.view = uint8_image_view(self.buffer, (height, width, 4))
        self.source_alpha = numpy.zeros((height, width), dtype=numpy.float32)
        self.source_shape = numpy.zeros((height, width), dtype=numpy.float32)
        self.synced_parent: RasterGroup | None = None
        self.dirty: list[int] | None = None


PREPARED_IMAGE_CACHE_BYTES = 256 << 20


def prepared_image_bytes(prepared: PreparedImage | None) -> int:
    if prepared is None:
        return 0
    soft_mask = prepared.soft_mask
    return prepared.raster.array.nbytes + (0 if soft_mask is None else soft_mask.array.nbytes)


class ByteBudgetCache[K, V]:
    """Values evicted oldest first once the bytes they hold exceed the budget."""

    __slots__ = ("budget", "entries", "size")

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.entries: dict[K, tuple[V, int]] = {}
        self.size = 0

    def get(self, key: K) -> V | None:
        entry = self.entries.get(key)
        return None if entry is None else entry[0]

    def store(self, key: K, value: V, size: int) -> None:
        entries = self.entries
        previous = entries.pop(key, None)
        if previous is not None:
            self.size -= previous[1]
        if size > self.budget:
            # Too large to keep. Leave the key absent so the value is built
            # again: a stored placeholder would read back as a real result.
            return
        while entries and self.size + size > self.budget:
            self.size -= entries.pop(next(iter(entries)))[1]
        entries[key] = (value, size)
        self.size += size


# Keyed by id(source); the entry holds the source, so the id cannot be reused
# while it is cached.
type PreparedImageCache = ByteBudgetCache[int, tuple[ImageSource, PreparedImage | None]]


def prepared_image(cache: PreparedImageCache, source: ImageSource) -> PreparedImage | None:
    cached = cache.get(id(source))
    if cached is not None and cached[0] is source:
        return cached[1]
    try:
        prepared = prepare_image(source)
    except Exception:
        prepared = None
    cache.store(id(source), (source, prepared), prepared_image_bytes(prepared))
    return prepared


def image_placement(item: ImagePaintItem) -> tuple[tuple[float, float], ...] | None:
    if item.quad is not None:
        return item.quad
    box = rect_tuple(item.bbox)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


def edge_tuples(
    edge_array: numpy.ndarray[Any, Any] | None,
) -> list[tuple[float, float, float, float]]:
    if edge_array is None:
        return []
    return [(x0, y0, x1, y1) for x0, y0, x1, y1 in edge_array.tolist()]


def paint_stroke_once(
    target: RasterTarget,
    path: CapturedPath,
    line_width: float,
    rgba: tuple[int, int, int, int],
    dash_pattern: tuple[list[float], float] | None,
    blend_mode: str | None,
    line_cap: int,
    line_join: int,
) -> None:
    pixels = target.pixels
    coverage_buffer = bytearray(len(pixels))
    with target.detached_buffer(coverage_buffer):
        target.stroke_path(
            path, line_width, (0, 0, 0, 255), dash_pattern, None, line_cap, line_join
        )
    coverage = target.pixel_view(coverage_buffer)[..., 3]
    covered_rows = numpy.flatnonzero(coverage.any(axis=1))
    if covered_rows.size == 0:
        return
    covered_columns = numpy.flatnonzero(coverage.any(axis=0))
    rows = slice(int(covered_rows[0]), int(covered_rows[-1]) + 1)
    columns = slice(int(covered_columns[0]), int(covered_columns[-1]) + 1)
    coverage = coverage[rows, columns]
    alpha = numpy.rint(coverage.astype(numpy.float64) * (rgba[3] / 255.0)).astype(numpy.uint8)
    target.record_source_coverage(rows, columns, alpha, shape=coverage)
    visible = alpha > 0
    if not numpy.any(visible):
        return
    blend_visible_pixels(
        target.pixel_array[rows, columns],
        visible,
        rgba[0] / 255.0,
        rgba[1] / 255.0,
        rgba[2] / 255.0,
        alpha[visible].astype(numpy.float64) / 255.0,
        target.resolved_blend(blend_mode),
        semantic_context=target.semantic_context,
    )


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
    mode = blend_mode.casefold() if isinstance(blend_mode, str) else None
    if opacity == 1.0 and mode in {None, "normal"} and mask_alpha is None:
        # Every elementary group around a glyph lands here: quantizing the
        # coverage and copying the covered pixels across is the whole of it,
        # and the planes are far too small to pay for nine numpy passes.
        return composite_elementary_normal(destination, rendered, source_alpha)
    scaled_alpha = source_alpha.astype(numpy.float64) * opacity * 255.0
    if mask_alpha is not None:
        scaled_alpha *= mask_alpha
    effective_alpha = numpy.rint(scaled_alpha).astype(numpy.uint8)
    visible = effective_alpha > 0
    if not numpy.any(visible):
        return effective_alpha
    if opacity == 1.0 and mode in {None, "normal"}:
        # Reachable only with a soft mask; the unmasked case went to the kernel.
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
    mask_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
    *,
    semantic_context: SemanticContext,
) -> UInt8Array:
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
        blend_mode.casefold() if isinstance(blend_mode, str) else None,
        semantic_context=semantic_context,
    )
    return effective_alpha


SoftMaskPlane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
SoftMaskKey = tuple[int, int, tuple[float, float]]
# A resolved plane covers the whole page in float32, so a page that uses many
# distinct masks keeps far more of them than it can afford: one corpus page,
# PyMuPDF/tests/resources/test_3450.pdf, held 1,598 planes totalling 3,208MB,
# which was 91% of its peak resident set. The budget is generous next to the
# prepared-image one because re-resolving a mask means replaying its content
# stream, which is dearer than re-preparing an image.
SOFT_MASK_CACHE_BYTES = 512 << 20

# A half-open pixel box, as clipped_pixel_box returns it, and the one that
# stands for "this element lands nowhere on the page".
type PixelBox = tuple[int, int, int, int]
EMPTY_PIXEL_BOX: PixelBox = (0, 0, 0, 0)


type SoftMaskCache = ByteBudgetCache[SoftMaskKey, tuple[CapturedSoftMask, SoftMaskPlane | None]]


def graphics_soft_mask(item: DisplayItem) -> CapturedSoftMask | None:
    if isinstance(item, (PathPaintItem, ImagePaintItem)):
        return item.graphics_soft_mask
    mask: CapturedSoftMask | None = item.data.get("graphics_soft_mask")
    return mask


def resolve_soft_mask(target: RasterTarget, mask: CapturedSoftMask) -> SoftMaskPlane | None:
    key = (id(mask.program), id(mask.transfer), mask.offset)
    cached = target.soft_mask_cache.get(key)
    if cached is not None:
        return cached[1]
    if key in target.active_soft_masks:
        return None
    target.active_soft_masks.add(key)
    result: SoftMaskPlane | None = None
    try:
        nested, view = target.blank_sibling()
        display = DisplayList(target.width, target.height, preserve_object_boundaries=True)
        append_captured_program(display, mask.program, include_text=True)
        nested.paint_items(display.items, translation=mask.offset)
        alpha = view[..., 3]
        if mask.transfer is None:
            result = alpha.astype(numpy.float32) / 255.0
        else:
            samples, inverse = numpy.unique(alpha, return_inverse=True)
            values = [mask.transfer(int(sample) / 255.0)[0] for sample in samples]
            result = numpy.asarray(values, dtype=numpy.float32)[inverse].reshape(alpha.shape)
        result.setflags(write=False)
    except Exception:
        result = None
    finally:
        target.active_soft_masks.remove(key)
    target.soft_mask_cache.store(key, (mask, result), 0 if result is None else result.nbytes)
    return result


AFFINE_BLIT_SCRATCH_BYTES = 1 << 20


def box_downsample(
    samples: numpy.ndarray[Any, Any],
    source_width: int,
    source_height: int,
    channels: int,
    target_width: int,
    target_height: int,
) -> tuple[numpy.ndarray[Any, Any], int, int]:
    if target_width <= 0 or target_height <= 0:
        return samples, source_width, source_height
    if source_width <= target_width and source_height <= target_height:
        return samples, source_width, source_height
    target_width = min(target_width, source_width)
    target_height = min(target_height, source_height)
    grid = samples.reshape(source_height, source_width, channels)
    row_edges = (numpy.arange(target_height + 1, dtype=numpy.int64) * source_height) // (
        target_height
    )
    column_edges = (numpy.arange(target_width + 1, dtype=numpy.int64) * source_width) // (
        target_width
    )
    totals = numpy.add.reduceat(grid, row_edges[:-1], axis=0, dtype=numpy.uint32)
    totals = numpy.add.reduceat(totals, column_edges[:-1], axis=1, dtype=numpy.uint32)
    counts = numpy.diff(row_edges)[:, None, None] * numpy.diff(column_edges)[None, :, None]
    reduced = (totals // numpy.maximum(counts, 1)).astype(numpy.uint8)
    return reduced.reshape(-1), target_width, target_height


def sample_image_plane(
    plane: UInt8Array, u: numpy.ndarray[Any, Any], v: numpy.ndarray[Any, Any]
) -> UInt8Array:
    height, width = plane.shape
    source_x = numpy.clip((u * width).astype(numpy.intp), 0, width - 1)
    source_y = numpy.clip(((1.0 - v) * height).astype(numpy.intp), 0, height - 1)
    return plane[source_y, source_x]


class RasterTarget:
    __slots__ = (
        "pixels",
        "pixel_array",
        "semantic_context",
        "buffer_stack",
        "group_source_alpha",
        "group_source_shape",
        "paint_window",
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
        "prepared_image_cache",
        "tiling_cell_cache",
        "active_soft_masks",
        "elementary_scratch",
    )

    def blend_coverage_pixel(
        self,
        offset: int,
        rgba: tuple[int, int, int, int],
        covered: int,
        sample_total: int,
        *,
        track_shape: bool,
        blend_resolved_mode: str | None,
    ) -> None:
        """Paint one antialiased pixel: coverage fraction to alpha, then blend."""
        alpha = max(0, min(255, round(rgba[3] * covered / sample_total)))
        shape = round(255 * covered / sample_total) if track_shape else 255
        self.blend_px(offset, (rgba[0], rgba[1], rgba[2], alpha), blend_resolved_mode, shape=shape)

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: ClipState,
        width: int,
        height: int,
        scale: float,
        crop_x0: float,
        crop_y0: float,
        crop_y1: float,
        page_view: UInt8Array,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.semantic_context = blend_context(semantic_context)
        self.buffer_stack = [RasterGroup(pixels, view=page_view)]
        # pixels, pixel_array, group_source_alpha, group_source_shape and
        # paint_window mirror the innermost group for the paint loops, which
        # read them per pixel.
        # sync_group_mirrors is the only thing that sets them.
        self.pixels: bytearray
        self.pixel_array: UInt8Array
        self.group_source_alpha: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
        self.group_source_shape: numpy.ndarray[Any, numpy.dtype[numpy.float32]] | None
        self.paint_window: list[int] | None
        self.sync_group_mirrors()
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
        self.soft_mask_cache: SoftMaskCache = ByteBudgetCache(SOFT_MASK_CACHE_BYTES)
        self.prepared_image_cache: PreparedImageCache = ByteBudgetCache(PREPARED_IMAGE_CACHE_BYTES)
        self.tiling_cell_cache: TilingCellCache = {}
        self.active_soft_masks: set[SoftMaskKey] = set()
        self.elementary_scratch: dict[int, ElementaryScratch] = {}
        if group_alpha is not None:
            self.push_group(bytearray(len(pixels)), group_alpha, None)
            self.group_floor = len(self.buffer_stack)

    def push_scope(self, clip_path: CapturedPath | None = None) -> None:
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
        try:
            if clip_path is not None:
                self.clip.push(clip_path, "nonzero")
            self.clip_floor = self.clip.depth
            self.group_floor = len(self.buffer_stack)
        except BaseException:
            self.pop_scope()
            raise

    def blank_sibling(self) -> tuple[RasterTarget, UInt8Array]:
        pixels = bytearray(self.width * self.height * 4)
        view = uint8_image_view(pixels, (self.height, self.width, 4))
        sibling = RasterTarget(
            pixels,
            None,
            clip=ClipState(
                crop_x0=self.crop_x0,
                crop_y1=self.crop_y1,
                scale=self.scale,
                width=self.width,
                height=self.height,
            ),
            width=self.width,
            height=self.height,
            scale=self.scale,
            crop_x0=self.crop_x0,
            crop_y0=self.crop_y0,
            crop_y1=self.crop_y1,
            page_view=view,
            semantic_context=self.semantic_context,
        )
        sibling.soft_mask_cache = self.soft_mask_cache
        sibling.prepared_image_cache = self.prepared_image_cache
        sibling.tiling_cell_cache = self.tiling_cell_cache
        sibling.active_soft_masks = self.active_soft_masks
        return sibling, view

    def pop_scope(self) -> None:
        if not self.scope_stack:
            return
        clip_depth, clip_stack, clip_floor, buffer_depth, group_floor = self.scope_stack.pop()
        try:
            while len(self.buffer_stack) > buffer_depth:
                self.composite_group(self.pop_group())
        finally:
            while len(self.buffer_stack) > buffer_depth:
                self.pop_group()
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

    # Past this many boxes the linear scan stops paying for itself, and a
    # knockout group with that many elements is not the text case this is for.
    KNOCKOUT_DISJOINT_LIMIT = 96

    def knockout_paint_box(self, item: DisplayItem) -> PixelBox | None:
        """The pixel box `item` would paint straight into a knockout group, or
        None if it cannot skip its elementary group at all.

        Only a plain fill qualifies: an edge-array fill, where fill_path
        provably paints inside item.bbox, with no pattern and a Normal blend.
        Anything else composites through a group, and composite_group records
        what that group actually painted, so nothing else needs predicting.
        """
        if not (
            isinstance(item, PathPaintItem)
            and item.paint_kind is PathPaintKind.FILL
            and item.edge_array is not None
            and item.bbox is not None
            and item.fill_pattern is None
            and item.blend_mode in (None, "Normal")
        ):
            return None
        clipped = self.clip.clipped_pixel_box(item.bbox)
        if clipped is None:
            # Nothing of it lands on the page, so it paints nothing at all.
            return EMPTY_PIXEL_BOX
        # Exactly the box fill_path will paint into -- it clips the coverage
        # plane to this and blends only inside it -- so no margin is needed.
        # An earlier version inflated by a pixel for antialiasing and skipped
        # only 7.6% of a dense text page, because adjacent glyph boxes tile
        # contiguously and a one-pixel margin makes every neighbour an overlap.
        return clipped[1]

    def record_knockout_paint(self, box: PixelBox) -> None:
        """Record a box painted into the innermost group, if it knocks out.

        Everything painted into the group is recorded, not only the elements
        that skipped its elementary group: one composited in through a group
        leaves pixels behind just the same, and a later element landing on them
        would no longer be painting over the initial backdrop. composite_group
        records those from the window the group actually painted.
        """
        boxes = self.buffer_stack[-1].painted_boxes
        if boxes is None or box == EMPTY_PIXEL_BOX:
            return
        if len(boxes) >= self.KNOCKOUT_DISJOINT_LIMIT:
            boxes.clear()
            boxes.append((0, 0, self.width, self.height))
        else:
            boxes.append(box)

    def knockout_skip_box(self, item: DisplayItem, boxes: list[PixelBox]) -> PixelBox | None:
        """The box `item` paints, when it misses every pixel painted into the
        knockout group so far and so can knock out without an elementary group;
        None when it needs the group.

        Knockout composites each element against the group's *initial*
        backdrop rather than the accumulated result. Where nothing has been
        painted yet those are the same buffer, the accumulated source alpha is
        zero, and the knockout formula collapses: `dst == bak` makes
        `colour * complete - backdrop * initial` vanish, `rga` becomes the
        element's own alpha and `ra` equals it, so the component reduces to the
        element itself. Painting straight into the parent produces that.
        """
        box = self.knockout_paint_box(item)
        if box is None or box == EMPTY_PIXEL_BOX:
            # One that paints nothing would have its group paint nothing either.
            return box
        x0, y0, x1, y1 = box
        for bx0, by0, bx1, by1 in boxes:
            if x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1:
                return None
        return box

    def paint_item(self, item: DisplayItem) -> None:
        if isinstance(item, (PathPaintItem, ImagePaintItem)) or item.kind in {"glyph", "shading"}:
            previous_shape_state = self.paint_alpha_is_shape, self.shape_alpha
            self.paint_alpha_is_shape = (
                item.alpha_is_shape
                if isinstance(item, (PathPaintItem, ImagePaintItem))
                else item.data.get("alpha_is_shape") is True
            )
            self.shape_alpha = 1.0
            try:
                knockout = self.buffer_stack[-1].knockout
                mask = graphics_soft_mask(item)
                mask_alpha = resolve_soft_mask(self, mask) if mask is not None else None
                fillstroke = (
                    isinstance(item, PathPaintItem) and item.paint_kind is PathPaintKind.FILL_STROKE
                )
                elementary_group = knockout or mask_alpha is not None
                boxes = self.buffer_stack[-1].painted_boxes
                skip_box = None
                if knockout and boxes is not None and mask_alpha is None:
                    skip_box = self.knockout_skip_box(item, boxes)
                    if skip_box is not None:
                        elementary_group = False
                if elementary_group:
                    self.push_elementary_group(
                        track_shape=mask_alpha is not None,
                        mask_alpha=None if fillstroke else mask_alpha,
                        alpha_is_shape=self.paint_alpha_is_shape,
                    )
                try:
                    self.paint_display_item(item)
                finally:
                    if elementary_group:
                        child = self.pop_group()
                        try:
                            self.composite_group(child)
                        finally:
                            self.release_elementary_group(child)
                    elif skip_box is not None:
                        self.record_knockout_paint(skip_box)
            finally:
                self.paint_alpha_is_shape, self.shape_alpha = previous_shape_state
            return
        self.paint_display_item(item)

    def paint_display_item(self, item: DisplayItem) -> None:
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
        match item.kind:
            case "scope-begin":
                path = data.get("path")
                self.push_scope(path if isinstance(path, CapturedPath) else None)
            case "scope-end":
                self.pop_scope()
            case "state-push":
                self.clip_stack.append(self.clip.depth)
            case "state-pop":
                if self.clip_stack:
                    self.clip.restore(self.clip_stack.pop())
                else:
                    self.clip.restore(self.clip_floor)
            case "clip":
                path = data.get("path")
                if isinstance(path, CapturedPath) and path.has_segments():
                    self.clip.push(path, data.get("fill_rule") or "nonzero")
            case "group-begin":
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
                    mask_alpha=resolve_soft_mask(self, graphics_mask)
                    if (graphics_mask := data.get("graphics_soft_mask")) is not None
                    else None,
                )
            case "group-end" if len(self.buffer_stack) > self.group_floor:
                self.composite_group(self.pop_group())
            case "glyph" if data.get("visible") is not False:
                rgba = color_rgba(data.get("fill_color"), data.get("fill_opacity"))
                if is_pdf_number(mask := data.get("soft_mask_alpha")):
                    rgba = scale_rgba_alpha(rgba, mask)
                self.set_shape_alpha(rgba[3] / 255.0)
                self.draw_glyph_bitmap(
                    data.get("bbox"),
                    data.get("bitmap"),
                    rgba,
                    blend_mode,
                    data.get("bitmap_width"),
                    data.get("bitmap_height"),
                )
            case "shading":
                self.set_shape_alpha(
                    resolve_constant_alpha(data.get("fill_opacity"), data.get("soft_mask_alpha"))
                )
                self.paint_shading(data, blend_mode)

    def push_elementary_group(
        self,
        *,
        track_shape: bool,
        mask_alpha: SoftMaskPlane | None,
        alpha_is_shape: bool,
    ) -> None:
        parent = self.buffer_stack[-1]
        backdrop = parent.backdrop if parent.knockout else self.pixels
        if backdrop is None:
            self.push_group(
                bytearray(len(self.pixels)),
                None,
                None,
                isolated=False,
                track_shape=track_shape,
                mask_alpha=mask_alpha,
                alpha_is_shape=alpha_is_shape,
            )
            return
        depth = len(self.buffer_stack)
        scratch = self.elementary_scratch.get(depth)
        if scratch is None:
            scratch = self.elementary_scratch[depth] = ElementaryScratch(
                len(self.pixels), self.height, self.width
            )
        buffer = scratch.buffer
        source_alpha = scratch.source_alpha
        source_shape = scratch.source_shape
        if parent.knockout and scratch.synced_parent is parent:
            dirty = scratch.dirty
            if dirty:
                y0, y1, x0, x1 = dirty
                scratch.view[y0:y1, x0:x1] = self.pixel_view(backdrop)[y0:y1, x0:x1]
                source_alpha[y0:y1, x0:x1] = 0.0
                source_shape[y0:y1, x0:x1] = 0.0
        else:
            buffer[:] = backdrop
            source_alpha.fill(0.0)
            source_shape.fill(0.0)
        scratch.synced_parent = parent if parent.knockout else None
        scratch.dirty = None
        self.buffer_stack.append(
            RasterGroup(
                buffer,
                None,
                None,
                view=scratch.view,
                backdrop=backdrop,
                source_alpha=source_alpha,
                source_shape=(
                    source_shape if track_shape or parent.source_shape is not None else None
                ),
                knockout=False,
                alpha_is_shape=alpha_is_shape,
                mask_alpha=mask_alpha,
            )
        )
        self.sync_group_mirrors()

    def release_elementary_group(self, child: RasterGroup) -> None:
        scratch = self.elementary_scratch.get(len(self.buffer_stack))
        if scratch is not None and child.pixels is scratch.buffer:
            scratch.dirty = list(child.paint_window) if child.paint_window else None

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
            RasterGroup(
                buffer,
                group_alpha,
                blend_mode,
                view=uint8_image_view(buffer, (self.height, self.width, 4)),
                backdrop=backdrop,
                source_alpha=source_alpha,
                source_shape=source_shape,
                knockout=knockout,
                alpha_is_shape=alpha_is_shape,
                mask_alpha=mask_alpha,
                # Only a knockout group needs this, and only it reads it.
                painted_boxes=[] if knockout else None,
            )
        )
        self.sync_group_mirrors()

    def pop_group(self) -> RasterGroup:
        child = self.buffer_stack.pop()
        self.sync_group_mirrors()
        return child

    def sync_group_mirrors(self) -> None:
        """Point the paint loops' attributes at the innermost group.

        The page level keeps no paint window: nothing composites it, so
        tracking one there would be pure cost.
        """
        group = self.buffer_stack[-1]
        self.pixels = group.pixels
        self.pixel_array = group.view
        self.group_source_alpha = group.source_alpha
        self.group_source_shape = group.source_shape
        self.paint_window = group.paint_window if len(self.buffer_stack) > 1 else None

    @contextmanager
    def detached_buffer(self, buffer: bytearray) -> Iterator[None]:
        """Paint into `buffer` alone, with no group planes and no paint window.

        For a pass that paints coverage into scratch and records it into the
        group itself afterwards; the group's mirrors come back on the way out.
        """
        self.pixels = buffer
        self.pixel_array = self.pixel_view(buffer)
        self.group_source_alpha = None
        self.group_source_shape = None
        self.paint_window = None
        try:
            yield
        finally:
            self.sync_group_mirrors()

    def set_shape_alpha(self, alpha: float) -> None:
        self.shape_alpha = clamp01(alpha) if self.paint_alpha_is_shape else 1.0

    def extend_paint_window(self, rows: int | slice, columns: int | slice) -> None:
        window = self.paint_window
        if window is None:
            return
        y0, y1 = index_extent(rows, self.height)
        x0, x1 = index_extent(columns, self.width)
        if window:
            if y0 < window[0]:
                window[0] = y0
            if y1 > window[1]:
                window[1] = y1
            if x0 < window[2]:
                window[2] = x0
            if x1 > window[3]:
                window[3] = x1
        else:
            window[:] = (y0, y1, x0, x1)

    def record_source_coverage(
        self,
        rows: int | slice,
        columns: int | slice,
        alpha: int | UInt8Array,
        *,
        shape: int | UInt8Array = 255,
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
    ) -> None:
        if self.group_source_alpha is None and self.group_source_shape is None:
            self.extend_paint_window(rows, columns)
            return
        self.record_source_alpha(rows, columns, alpha, visible=visible)
        self.record_source_shape(rows, columns, shape, visible=visible)

    def record_source_alpha(
        self,
        rows: int | slice,
        columns: int | slice,
        alpha: int | UInt8Array,
        *,
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
    ) -> None:
        plane = self.group_source_alpha
        if plane is None:
            self.extend_paint_window(rows, columns)
        else:
            self.record_plane(plane, rows, columns, alpha / 255.0, visible)

    def pixel_view(self, buffer: bytearray | bytes) -> UInt8Array:
        """A (height, width, 4) view over a buffer no group owns.

        Every group holds its own view -- self.pixel_array is the innermost
        one's -- so this is for one-off buffers: a stroke's scratch coverage,
        and a backdrop reached through a knockout parent.
        """
        if buffer is self.page_buffer:
            return self.page_pixels
        return uint8_image_view(buffer, (self.height, self.width, 4))

    def record_source_shape(
        self,
        rows: int | slice,
        columns: int | slice,
        shape: int | UInt8Array,
        *,
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None = None,
    ) -> None:
        plane = self.group_source_shape
        if plane is None:
            self.extend_paint_window(rows, columns)
        else:
            self.record_plane(plane, rows, columns, shape / 255.0 * self.shape_alpha, visible)

    def record_plane(
        self,
        plane: numpy.ndarray[Any, numpy.dtype[numpy.float32]],
        rows: int | slice,
        columns: int | slice,
        source: float | numpy.ndarray[Any, Any],
        visible: numpy.ndarray[Any, numpy.dtype[numpy.bool_]] | None,
    ) -> None:
        self.extend_paint_window(rows, columns)
        previous = plane[rows, columns]
        updated = previous + (1.0 - previous) * source
        plane[rows, columns] = (
            updated if visible is None else numpy.where(visible, updated, previous)
        )

    def resolved_blend(self, blend_mode: str | None) -> str | None:
        return blend_mode.lower() if isinstance(blend_mode, str) else None

    def blend_px(
        self,
        idx: int,
        rgba: tuple[int, int, int, int],
        mode: str | None,
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
            # Neither plane is recording, so nothing else will note that the
            # group's buffer was written here, and the window it composites
            # over would miss this pixel.
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
        out_r = int(round(((src_r * 255.0) * src_a + dr * dst_a * (1.0 - src_a)) / out_a))
        out_g = int(round(((src_g * 255.0) * src_a + dg * dst_a * (1.0 - src_a)) / out_a))
        out_b = int(round(((src_b * 255.0) * src_a + db * dst_a * (1.0 - src_a)) / out_a))
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

    def group_window(self, group: RasterGroup) -> tuple[slice, slice] | None:
        """The rows and columns `group` has to be composited over.

        Outside the window the group is untouched -- transparent if it was
        isolated, equal to the backdrop it was seeded with if it was not -- and
        every compositing formula here leaves the destination alone where the
        source contributes nothing. So the window is the whole of the work, and
        an empty one means the group painted nothing at all.
        """
        window = group.paint_window
        if not window:
            return None
        y0, y1, x0, x1 = window
        return slice(y0, y1), slice(x0, x1)

    def composite_group(self, group: RasterGroup) -> None:
        parent = self.buffer_stack[-1]
        window = self.group_window(group)
        if window is None:
            return
        rows, columns = window
        if parent.painted_boxes is not None:
            # Whatever reached the knockout parent -- an element's elementary
            # group, a nested group, a pattern cell -- touched these pixels and
            # no others, so they are what a later element has to miss.
            self.record_knockout_paint((columns.start, rows.start, columns.stop, rows.stop))
        shape = group.source_shape[rows, columns] if group.source_shape is not None else None
        if shape is not None and group.alpha_is_shape:
            shape = shape * group.source_scale
            if group.mask_alpha is not None:
                shape = shape * group.mask_alpha[rows, columns]
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
        normalized_blend_mode = (
            group_blend_mode.casefold() if isinstance(group_blend_mode, str) else None
        )
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
                group.mask_alpha[rows, columns],
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
        effective_alpha = numpy.clip(
            numpy.rint(child[..., 3].astype(numpy.float64) * source_scale), 0.0, 255.0
        ).astype(numpy.uint8)
        if normalized_blend_mode in {None, "normal"} and len(group.pixels) >= 4_096:
            composite_normal_group_numpy(destination, child, source_scale)
            return effective_alpha
        composite_blended_group_numpy(
            destination,
            child,
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
            self.push_group(bytearray(len(self.pixels)), None, None, isolated=False, knockout=True)
            try:
                self.paint_item(replace(item, paint_kind=PathPaintKind.FILL))
                self.paint_item(replace(item, paint_kind=PathPaintKind.STROKE))
            finally:
                self.composite_group(self.pop_group())
            return
        if paint_kind is not PathPaintKind.STROKE:
            rgba = color_rgba(item.fill, item.fill_opacity)
            if is_pdf_number(soft_mask_alpha):
                rgba = scale_rgba_alpha(rgba, soft_mask_alpha)
            self.set_shape_alpha(rgba[3] / 255.0)
            if item.fill_pattern is None or not self.paint_fill_pattern(item, blend_mode):
                edge_array = item.edge_array
                self.fill_path(
                    path,
                    rgba,
                    blend_mode,
                    item.fill_rule,
                    bbox=item.bbox if edge_array is not None else None,
                    edge_array=edge_array,
                )
        if paint_kind is not PathPaintKind.FILL:
            stroke_rgba = color_rgba(item.stroke_color, item.stroke_opacity)
            if is_pdf_number(soft_mask_alpha):
                stroke_rgba = scale_rgba_alpha(stroke_rgba, soft_mask_alpha)
            self.set_shape_alpha(stroke_rgba[3] / 255.0)
            if self.group_source_shape is not None:
                paint_stroke_once(
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

    def blit_opaque_sampled_tiles(
        self,
        source_pixels: numpy.ndarray[Any, Any],
        target_region: numpy.ndarray[Any, Any],
        source_y: numpy.ndarray[Any, Any],
        source_x: numpy.ndarray[Any, Any],
        valid_rows: numpy.ndarray[Any, Any],
        valid_columns: numpy.ndarray[Any, Any],
        comps: int,
        *,
        transposed: bool = False,
        target_origin: tuple[int, int] = (0, 0),
    ) -> None:
        target_x, target_y = target_origin
        row_count = len(valid_rows)
        column_count = len(valid_columns)
        all_valid = bool(valid_rows.all() and valid_columns.all())
        sampled_channels = 1 if comps == 1 else 3
        scratch_bytes_per_pixel = sampled_channels if all_valid else sampled_channels + 8
        tile_columns = min(
            column_count,
            max(1, AFFINE_BLIT_SCRATCH_BYTES // scratch_bytes_per_pixel),
        )
        tile_rows = min(
            row_count,
            max(
                1,
                AFFINE_BLIT_SCRATCH_BYTES // max(1, tile_columns * scratch_bytes_per_pixel),
            ),
        )
        for row_start in range(0, row_count, tile_rows):
            row_end = min(row_count, row_start + tile_rows)
            for column_start in range(0, column_count, tile_columns):
                column_end = min(column_count, column_start + tile_columns)
                if transposed:
                    sampled = source_pixels[
                        source_y[None, column_start:column_end],
                        source_x[row_start:row_end, None],
                        :sampled_channels,
                    ]
                else:
                    sampled = source_pixels[
                        source_y[row_start:row_end, None],
                        source_x[None, column_start:column_end],
                        :sampled_channels,
                    ]
                target_tile = target_region[
                    row_start:row_end,
                    column_start:column_end,
                ]
                if all_valid:
                    target_tile[:, :, 0:3] = sampled
                    target_tile[:, :, 3] = 255
                    self.record_source_coverage(
                        slice(target_y + row_start, target_y + row_end),
                        slice(target_x + column_start, target_x + column_end),
                        255,
                    )
                    del sampled
                    continue
                visible = (
                    valid_rows[row_start:row_end, None]
                    & valid_columns[None, column_start:column_end]
                )
                numpy.copyto(target_tile[:, :, 0:3], sampled, where=visible[:, :, None])
                numpy.copyto(target_tile[:, :, 3], 255, where=visible)
                self.record_source_coverage(
                    slice(target_y + row_start, target_y + row_end),
                    slice(target_x + column_start, target_x + column_end),
                    255,
                    visible=visible,
                )
                del sampled, visible

    def blit_affine_image(
        self,
        quad: tuple[tuple[float, float], ...],
        converted: ByteBuffer,
        width_px: int,
        height_px: int,
        comps: int,
        constant_alpha: float | None,
        blend_mode: str | None,
        *,
        source_alpha: UInt8Array | None = None,
        source_shape: UInt8Array | None = None,
        soft_mask: UInt8Array | None = None,
        image_clip: tuple[float, float, float, float] | None = None,
    ) -> bool:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_resolved_mode = self.resolved_blend(blend_mode)
        blit_opaque_sampled_tiles = self.blit_opaque_sampled_tiles
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        scale = self.scale
        if len(quad) < 3:
            return False
        p00 = quad[0]
        p10 = quad[1]
        p01 = quad[2]
        quad_box = points_bbox(quad)
        if quad_box is None:
            return False
        if image_clip is not None:
            quad_box = intersect_box(quad_box, image_clip)
            if quad_box is None:
                return True
        rectangular_clip = current_clip() is not None and clip_paths_are_axis_aligned_rects()
        clipped_box = clipped_pixel_box(quad_box)
        if clipped_box is None:
            return True
        ix0, iy0, ix1, iy1 = clipped_box[1]
        ux = p10[0] - p00[0]
        uy = p10[1] - p00[1]
        vx = p01[0] - p00[0]
        vy = p01[1] - p00[1]
        det = ux * vy - uy * vx
        if abs(det) < 1e-9:
            return False
        inv_det = 1.0 / det
        alpha = 255
        if constant_alpha is not None:
            alpha = max(0, min(255, int(round(alpha * constant_alpha))))
        tracking_shape = self.group_source_shape is not None
        if alpha <= 0 and not tracking_shape:
            return True
        if source_alpha is not None:
            if not numpy.any(source_alpha) and not tracking_shape:
                return True
            if numpy.all(source_alpha == 255):
                source_alpha = None
        if soft_mask is not None:
            if not numpy.any(soft_mask) and not tracking_shape:
                return True
            if numpy.all(soft_mask == 255):
                soft_mask = None
        can_write_opaque = (
            alpha == 255 and blend_mode is None and source_alpha is None and soft_mask is None
        )
        rect_tolerance = max(abs(ux), abs(vy), 1.0) * 1e-6
        if (
            abs(uy) <= rect_tolerance
            and abs(vx) <= rect_tolerance
            and ux > 0
            and vy > 0
            and alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_regions or rectangular_clip)
        ):
            inv_ux = 1.0 / ux
            inv_vy = 1.0 / vy
            page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
            source_u = (page_x - p00[0]) * inv_ux
            source_samples = uint8_view(converted)
            valid_x = (source_u >= 0.0) & (source_u <= 1.0)
            safe_x = numpy.clip(
                (source_u * width_px).astype(numpy.intp),
                0,
                width_px - 1,
            )
            axis_page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
            source_y_array = ((1.0 - (axis_page_y - p00[1]) * inv_vy) * height_px).astype(
                numpy.intp
            )
            valid_y = (axis_page_y - p00[1]) * inv_vy >= 0.0
            valid_y &= (axis_page_y - p00[1]) * inv_vy <= 1.0
            safe_y = numpy.clip(source_y_array, 0, height_px - 1)
            target_region = self.pixel_array[iy0:iy1, ix0:ix1]
            source_pixels = source_samples[: width_px * height_px * comps].reshape(
                height_px,
                width_px,
                comps,
            )
            blit_opaque_sampled_tiles(
                source_pixels,
                target_region,
                safe_y,
                safe_x,
                valid_y,
                valid_x,
                comps,
                target_origin=(ix0, iy0),
            )
            return True
        u_from_x = abs(uy) <= rect_tolerance and abs(ux) > rect_tolerance
        u_from_y = abs(ux) <= rect_tolerance and abs(uy) > rect_tolerance
        v_from_x = abs(vy) <= rect_tolerance and abs(vx) > rect_tolerance
        v_from_y = abs(vx) <= rect_tolerance and abs(vy) > rect_tolerance
        if (
            alpha == 255
            and blend_mode is None
            and can_write_opaque
            and (not clip_regions or rectangular_clip)
            and ((u_from_x and v_from_y) or (u_from_y and v_from_x))
        ):
            target_pixels = self.pixel_array
            source_samples = uint8_view(converted)[: width_px * height_px * comps].reshape(
                height_px, width_px, comps
            )
            if u_from_x:
                inv_ux = 1.0 / ux
                inv_vy = 1.0 / vy
                page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
                page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
                source_u = (page_x - p00[0]) * inv_ux
                source_v = (page_y - p00[1]) * inv_vy
                valid_x = (source_u >= 0.0) & (source_u <= 1.0)
                valid_y = (source_v >= 0.0) & (source_v <= 1.0)
                source_x = numpy.clip(
                    (source_u * width_px).astype(numpy.intp),
                    0,
                    width_px - 1,
                )
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp),
                    0,
                    height_px - 1,
                )
            else:
                inv_uy = 1.0 / uy
                inv_vx = 1.0 / vx
                page_x = crop_x0 + (numpy.arange(ix0, ix1) + 0.5) / scale
                page_y = crop_y1 - (numpy.arange(iy0, iy1) + 0.5) / scale
                source_v = (page_x - p00[0]) * inv_vx
                source_u = (page_y - p00[1]) * inv_uy
                valid_x = (source_v >= 0.0) & (source_v <= 1.0)
                valid_y = (source_u >= 0.0) & (source_u <= 1.0)
                source_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp),
                    0,
                    height_px - 1,
                )
                source_x = numpy.clip(
                    (source_u * width_px).astype(numpy.intp),
                    0,
                    width_px - 1,
                )
            target_region = target_pixels[iy0:iy1, ix0:ix1]
            blit_opaque_sampled_tiles(
                source_samples,
                target_region,
                source_y,
                source_x,
                valid_y,
                valid_x,
                comps,
                transposed=not u_from_x,
                target_origin=(ix0, iy0),
            )
            return True
        source_pixels = uint8_view(converted)[: width_px * height_px * comps].reshape(
            height_px, width_px, comps
        )
        target_pixels = self.pixel_array
        tile_columns = min(ix1 - ix0, max(1, AFFINE_BLIT_SCRATCH_BYTES // 160))
        tile_rows = max(1, AFFINE_BLIT_SCRATCH_BYTES // (160 * tile_columns))
        for row_start in range(iy0, iy1, tile_rows):
            row_end = min(iy1, row_start + tile_rows)
            page_y = crop_y1 - (numpy.arange(row_start, row_end) + 0.5) / scale
            rel_y = page_y[:, None] - p00[1]
            for column_start in range(ix0, ix1, tile_columns):
                column_end = min(ix1, column_start + tile_columns)
                page_x = crop_x0 + (numpy.arange(column_start, column_end) + 0.5) / scale
                rel_x = page_x[None, :] - p00[0]
                source_u = (rel_x * vy - rel_y * vx) * inv_det
                source_v = (ux * rel_y - uy * rel_x) * inv_det
                visible = (
                    (source_u >= 0.0) & (source_u <= 1.0) & (source_v >= 0.0) & (source_v <= 1.0)
                )
                if clip_regions and not rectangular_clip:
                    allowed = numpy.zeros(visible.shape, dtype=numpy.bool_)
                    for local_y, py in enumerate(range(row_start, row_end)):
                        for start, end in clip_row_visible_spans(py):
                            start, end = max(start, column_start), min(end, column_end)
                            if end > start:
                                allowed[local_y, start - column_start : end - column_start] = True
                    visible &= allowed
                if not numpy.any(visible):
                    continue
                sample_x = numpy.clip((source_u * width_px).astype(numpy.intp), 0, width_px - 1)
                sample_y = numpy.clip(
                    ((1.0 - source_v) * height_px).astype(numpy.intp), 0, height_px - 1
                )
                sampled = source_pixels[sample_y, sample_x, : 1 if comps == 1 else 3]
                target = target_pixels[row_start:row_end, column_start:column_end]
                if can_write_opaque:
                    numpy.copyto(target[:, :, :3], sampled, where=visible[:, :, None])
                    numpy.copyto(target[:, :, 3], 255, where=visible)
                    self.record_source_coverage(
                        slice(row_start, row_end),
                        slice(column_start, column_end),
                        255,
                        visible=visible,
                    )
                    continue
                alpha_grid = (
                    sample_image_plane(source_alpha, source_u, source_v)
                    if source_alpha is not None
                    else numpy.full(visible.shape, 255, dtype=numpy.uint8)
                )
                if soft_mask is not None:
                    mask_alpha = sample_image_plane(soft_mask, source_u, source_v)
                    alpha_grid = numpy.rint(
                        alpha_grid.astype(numpy.float64) * mask_alpha / 255.0
                    ).astype(numpy.uint8)
                if tracking_shape:
                    shape_grid: int | UInt8Array = (
                        alpha_grid
                        if self.paint_alpha_is_shape
                        else sample_image_plane(source_shape, source_u, source_v)
                        if source_shape is not None
                        else 255
                    )
                    self.record_source_shape(
                        slice(row_start, row_end),
                        slice(column_start, column_end),
                        shape_grid,
                        visible=visible,
                    )
                if constant_alpha is not None:
                    alpha_grid = numpy.clip(
                        numpy.rint(alpha_grid.astype(numpy.float64) * constant_alpha), 0, 255
                    ).astype(numpy.uint8)
                visible &= alpha_grid > 0
                if not numpy.any(visible):
                    continue
                source_colors = numpy.broadcast_to(sampled, (*visible.shape, 3))[visible]
                blend_visible_pixels(
                    target,
                    visible,
                    source_colors[:, 0] / 255.0,
                    source_colors[:, 1] / 255.0,
                    source_colors[:, 2] / 255.0,
                    alpha_grid[visible] / 255.0,
                    blend_resolved_mode,
                    semantic_context=self.semantic_context,
                )
                self.record_source_alpha(
                    slice(row_start, row_end),
                    slice(column_start, column_end),
                    alpha_grid,
                    visible=visible,
                )
        return True

    def blit_image(self, item: ImagePaintItem) -> None:
        quad = image_placement(item)
        if quad is None or item.source is None:
            return
        box = points_bbox(quad)
        if box is None:
            return
        blend_mode = item.blend_mode
        if blend_mode == "Normal":
            blend_mode = None
        prepared = prepared_image(self.prepared_image_cache, item.source)
        if prepared is None:
            return
        if prepared.is_stencil:
            self.blit_image_mask(item, prepared, blend_mode)
            return
        raster = prepared.raster
        width_px, height_px = raster.width, raster.height
        components = 1 if raster.color_model == "gray" else 3
        converted = raster.array[:, :, :components].reshape(-1)
        native_soft_mask = prepared.soft_mask
        soft_mask = native_soft_mask.array[:, :, 0] if native_soft_mask is not None else None
        source_alpha: numpy.ndarray[Any, Any] | None = None
        if raster.has_alpha and soft_mask is None:
            source_alpha = raster.array[:, :, components].reshape(-1)
        device_extent = max(
            1,
            int(math.ceil((box[2] - box[0]) * self.scale)),
            int(math.ceil((box[3] - box[1]) * self.scale)),
        )
        if width_px > device_extent or height_px > device_extent:
            reduced, reduced_width, reduced_height = box_downsample(
                converted, width_px, height_px, components, device_extent, device_extent
            )
            if source_alpha is not None:
                source_alpha = box_downsample(
                    source_alpha, width_px, height_px, 1, device_extent, device_extent
                )[0]
            converted = reduced
            width_px, height_px = reduced_width, reduced_height
        if source_alpha is not None:
            source_alpha = source_alpha.reshape(height_px, width_px)
        source_shape = (
            source_alpha if image_color_key_mask_is_shape(item.source.dictionary) else None
        )
        scalar_mask = item.soft_mask_alpha if native_soft_mask is None else None
        opacity = item.fill_opacity
        constant_alpha = (
            resolve_constant_alpha(opacity, scalar_mask)
            if is_pdf_number(opacity) or is_pdf_number(scalar_mask)
            else None
        )
        self.set_shape_alpha(1.0 if constant_alpha is None else constant_alpha)
        self.blit_affine_image(
            quad,
            converted,
            width_px,
            height_px,
            components,
            constant_alpha,
            blend_mode,
            source_alpha=source_alpha,
            source_shape=source_shape,
            soft_mask=soft_mask,
            image_clip=rect_tuple(item.image_clip),
        )

    def blit_image_mask(
        self,
        item: ImagePaintItem,
        prepared: PreparedImage,
        blend_mode: str | None,
    ) -> None:
        quad = image_placement(item)
        raster = prepared.raster
        if quad is None or not raster.has_alpha:
            return
        red, green, blue, alpha = color_rgba(item.fill, item.fill_opacity)
        if is_pdf_number(item.soft_mask_alpha) and prepared.soft_mask is None:
            alpha = max(0, min(255, round(alpha * item.soft_mask_alpha)))
        self.set_shape_alpha(alpha / 255.0)
        if alpha <= 0 and self.group_source_shape is None:
            return
        self.blit_affine_image(
            quad,
            bytes((red, green, blue)),
            1,
            1,
            3,
            alpha / 255.0,
            blend_mode,
            source_alpha=raster.array[:, :, raster.channels - 1],
            source_shape=raster.array[:, :, raster.channels - 1],
            image_clip=rect_tuple(item.image_clip),
        )

    def fill_rect(
        self,
        box: tuple[float, float, float, float] | None,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        if box is None:
            return
        if blend_mode == "Normal" and rgba[3] == 255:
            blend_mode = None
        clipped_box = self.clip.clipped_pixel_box(box)
        if clipped_box is None:
            return
        (x0, y0, x1, y1), (ix0, iy0, ix1, iy1) = clipped_box
        rectangular_clip = self.clip.clip_paths_are_axis_aligned_rects()
        pixels = self.pixels
        scale = self.scale
        left = (x0 - self.crop_x0) * scale
        right = (x1 - self.crop_x0) * scale
        top = (self.crop_y1 - y1) * scale
        bottom = (self.crop_y1 - y0) * scale
        if (
            rectangular_clip
            and blend_mode is None
            and not (
                left <= ix0 + 1e-9
                and right >= ix1 - 1e-9
                and top <= iy0 + 1e-9
                and bottom >= iy1 - 1e-9
            )
        ):
            alpha_plane = rect_coverage_plane(ix0, ix1, iy0, iy1, left, right, top, bottom, rgba[3])
            blend_normal_alpha_array_numpy(
                self.pixel_array[iy0:iy1, ix0:ix1],
                rgba,
                alpha_plane,
            )
            self.record_source_alpha(slice(iy0, iy1), slice(ix0, ix1), alpha_plane)
            if self.group_source_shape is not None:
                self.record_source_shape(
                    slice(iy0, iy1),
                    slice(ix0, ix1),
                    rect_coverage_plane(ix0, ix1, iy0, iy1, left, right, top, bottom, 255),
                )
            return
        if rgba[3] == 255 and blend_mode is None and rectangular_clip:
            span = ix1 - ix0
            if span <= 0:
                return
            if pixels is self.page_buffer:
                self.page_pixels[iy0:iy1, ix0:ix1] = rgba
                self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
                return
            target_pixels = self.pixel_array
            blend_normal_solid_array_numpy(
                target_pixels[iy0:iy1, ix0:ix1],
                rgba,
            )
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
            return
        normal_fast = blend_mode is None
        normal_target = self.pixel_array if normal_fast else None
        if rectangular_clip and normal_fast and ix1 > ix0 and iy1 > iy0:
            assert normal_target is not None
            blend_normal_solid_array_numpy(
                normal_target[iy0:iy1, ix0:ix1],
                rgba,
            )
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
            return
        width = self.width
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        if normal_target is None:
            if rgba[3] <= 0 and self.group_source_shape is None:
                return
            blend_target = self.pixel_array
            if rectangular_clip:
                blend_solid_array_numpy(
                    blend_target[iy0:iy1, ix0:ix1],
                    rgba,
                    blend_mode,
                    semantic_context=self.semantic_context,
                )
                self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3])
                return
        for y in range(iy0, iy1):
            row = y * width * 4
            visible_spans = clip_row_visible_spans(y)
            if not visible_spans:
                continue
            for start, end in visible_spans:
                start = max(ix0, start)
                end = min(ix1, end)
                if end <= start:
                    continue
                if normal_target is not None:
                    if end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
                        blend_normal_solid_array_numpy(normal_target[y, start:end], rgba)
                        self.record_source_coverage(y, slice(start, end), rgba[3])
                    else:
                        for x in range(start, end):
                            blend_px(row + x * 4, rgba, None)
                elif end - start >= RASTER_NUMPY_SPAN_MIN_PIXELS:
                    blend_solid_array_numpy(
                        blend_target[y, start:end],
                        rgba,
                        blend_mode,
                        semantic_context=self.semantic_context,
                    )
                    self.record_source_coverage(y, slice(start, end), rgba[3])
                else:
                    for x in range(start, end):
                        blend_px(row + x * 4, rgba, blend_resolved_mode)

    def draw_glyph_bitmap(
        self,
        box: tuple[float, float, float, float] | None,
        bitmap: Any,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        bitmap_width: Any = None,
        bitmap_height: Any = None,
    ) -> None:
        clip = self.clip
        clip_regions = clip.regions
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        fill_rect = self.fill_rect
        page_box_to_pixels = clip.page_box_to_pixels
        scale = self.scale
        bitmap_type = type(bitmap)
        if box is None or (bitmap_type is not list and bitmap_type is not tuple) or not bitmap:
            return
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        rows = [int(row) for row in bitmap if type(row) is int]
        if not rows:
            return
        bitmap_h = parse_int(bitmap_height, 0) or len(rows)
        bitmap_w = parse_int(bitmap_width, 0) or max((row.bit_length() for row in rows), default=0)
        if bitmap_w <= 0 or bitmap_h <= 0:
            return
        cell_w = (x1 - x0) / bitmap_w
        cell_h = (y1 - y0) / bitmap_h
        if cell_w <= 0 or cell_h <= 0:
            return
        opaque_glyph = rgba[3] == 255 and (blend_mode is None or blend_mode == "Normal")
        if opaque_glyph and not clip_regions and bitmap_w <= 64:
            pixel_box = page_box_to_pixels(x0, y0, x1, y1)
            pixel_width = (x1 - x0) * scale
            pixel_height = (y1 - y0) * scale
            origin_x = (x0 - crop_x0) * scale
            origin_y = (crop_y1 - y1) * scale
            cell_pixel_width = pixel_width / bitmap_w
            cell_pixel_height = pixel_height / bitmap_h
            aligned = False
            if pixel_box is not None:
                aligned = (
                    abs(origin_x - round(origin_x)) <= 1e-9
                    and abs(origin_y - round(origin_y)) <= 1e-9
                    and abs(cell_pixel_width - round(cell_pixel_width)) <= 1e-9
                    and abs(cell_pixel_height - round(cell_pixel_height)) <= 1e-9
                    and cell_pixel_width >= 1.0
                    and cell_pixel_height >= 1.0
                    and pixel_box[2] - pixel_box[0] == round(pixel_width)
                    and pixel_box[3] - pixel_box[1] == round(pixel_height)
                )
            if aligned and pixel_box is not None:
                ix0, iy0, ix1, iy1 = pixel_box
                cell_pixel_width = int(round(cell_pixel_width))
                cell_pixel_height = int(round(cell_pixel_height))
                row_values = numpy.asarray(
                    rows[:bitmap_h] + [0] * max(0, bitmap_h - len(rows)),
                    dtype=numpy.uint64,
                )
                columns = numpy.arange(bitmap_w, dtype=numpy.uint64)
                bits = ((row_values[:, None] >> columns[None, :]) & 1).astype(bool)
                expanded = numpy.repeat(
                    numpy.repeat(bits, cell_pixel_height, axis=0),
                    cell_pixel_width,
                    axis=1,
                )
                target_pixels = self.pixel_array
                target_region = target_pixels[iy0:iy1, ix0:ix1]
                target_region[expanded] = rgba
                self.record_source_coverage(
                    slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=expanded
                )
                return
        for row_index, row in enumerate(rows[:bitmap_h]):
            cell_y1 = y1 - row_index * cell_h
            cell_y0 = y1 - (row_index + 1) * cell_h
            if opaque_glyph:
                remaining = row
                while remaining:
                    run_start = (remaining & -remaining).bit_length() - 1
                    if run_start >= bitmap_w:
                        break
                    shifted = remaining >> run_start
                    run_length = (~shifted & (shifted + 1)).bit_length() - 1
                    run_end = min(bitmap_w, run_start + run_length)
                    fill_rect(
                        (
                            x0 + run_start * cell_w,
                            cell_y0,
                            x0 + run_end * cell_w,
                            cell_y1,
                        ),
                        rgba,
                        blend_mode,
                    )
                    remaining &= ~(((1 << run_length) - 1) << run_start)
                continue
            for col_index in range(bitmap_w):
                if not (row & (1 << col_index)):
                    continue
                cell_x0 = x0 + col_index * cell_w
                cell_x1 = x0 + (col_index + 1) * cell_w
                fill_rect((cell_x0, cell_y0, cell_x1, cell_y1), rgba, blend_mode)

    def fill_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
    ) -> None:
        clip = self.clip
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        page_box_to_pixels = clip.page_box_to_pixels
        pixels = self.pixels
        scale = self.scale
        width = self.width
        circle_box = (cx - radius, cy - radius, cx + radius, cy + radius)
        clip_box = current_clip() if clip_regions else None
        if clip_box is not None:
            clipped_circle_box = intersect_box(circle_box, clip_box)
            if clipped_circle_box is None:
                return
            circle_box = clipped_circle_box
        pixel_box = page_box_to_pixels(*circle_box)
        if pixel_box is None:
            return
        ix0, iy0, ix1, iy1 = pixel_box
        radius2 = radius * radius
        normal_fast = blend_mode is None
        rectangular_clip = not clip_regions or clip_paths_are_axis_aligned_rects()
        if normal_fast and rgba[3] >= 255 and rectangular_clip:
            if (ix1 - ix0) * (iy1 - iy0) > RASTER_CIRCLE_MIN_PIXEL_AREA:
                x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
                y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
                circle_page_xs = crop_x0 + (x_coords + 0.5) / scale
                circle_page_ys = crop_y1 - (y_coords + 0.5) / scale
                inside = (circle_page_xs[None, :] - cx) ** 2 + (
                    circle_page_ys[:, None] - cy
                ) ** 2 <= radius2
                self.pixel_array[iy0:iy1, ix0:ix1][inside] = rgba
                self.record_source_coverage(
                    slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=inside
                )
                return
            red, green, blue, alpha = rgba
            for py in range(iy0, iy1):
                page_y = crop_y1 - (py + 0.5) / scale
                dy = page_y - cy
                row = py * width * 4
                for px in range(ix0, ix1):
                    page_x = crop_x0 + (px + 0.5) / scale
                    dx = page_x - cx
                    if dx * dx + dy * dy > radius2:
                        continue
                    index = row + px * 4
                    pixels[index] = red
                    pixels[index + 1] = green
                    pixels[index + 2] = blue
                    pixels[index + 3] = 255
                    self.record_source_coverage(py, px, 255)
            return
        for py in range(iy0, iy1):
            page_y = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for clip_start, clip_end in visible_spans:
                start = max(ix0, clip_start)
                end = min(ix1, clip_end)
                if end <= start:
                    continue
                for px in range(start, end):
                    page_x = crop_x0 + (px + 0.5) / scale
                    dx = page_x - cx
                    dy = page_y - cy
                    if dx * dx + dy * dy > radius2:
                        continue
                    blend_px(row + px * 4, rgba, blend_resolved_mode)

    def fill_path_scanlines(
        self,
        edge_segments: list[tuple[float, float, float, float, float, float]],
        pixel_box: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        blend_mode: str | None,
        fill_rule: str,
    ) -> None:
        blend_normal_solid_span = self.blend_normal_solid_span
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_paths_are_axis_aligned_rects = self.clip.clip_paths_are_axis_aligned_rects
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_buffer = self.page_buffer
        page_pixels = self.page_pixels
        pixels = self.pixels
        scale = self.scale
        width = self.width
        ix0, iy0, ix1, iy1 = pixel_box
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        simple_opaque = rgba[3] == 255 and blend_mode is None and rectangular_clip
        normal_fast = blend_mode is None
        normal_target = self.pixel_array if normal_fast and not simple_opaque else None
        blend_target = self.pixel_array if not normal_fast and rgba[3] > 0 else None

        def span_pixels(start_x: float, end_x: float) -> tuple[int, int] | None:
            if end_x <= start_x:
                return None
            start = math.ceil((start_x - crop_x0) * scale - 0.5)
            end = math.ceil((end_x - crop_x0) * scale - 0.5)
            start = max(ix0, min(ix1, start))
            end = max(ix0, min(ix1, end))
            if end <= start:
                return None
            return start, end

        edge_count = len(edge_segments)
        pending_order = sorted(range(edge_count), key=lambda i: -edge_segments[i][5])
        pending_index = 0
        active_heap: list[tuple[float, int]] = []
        for py in range(iy0, iy1):
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            page_y = crop_y1 - (py + 0.5) / scale
            while (
                pending_index < edge_count
                and edge_segments[pending_order[pending_index]][5] > page_y
            ):
                edge_index = pending_order[pending_index]
                heapq.heappush(active_heap, (edge_segments[edge_index][4], edge_index))
                pending_index += 1
            while active_heap and active_heap[0][0] > page_y:
                heapq.heappop(active_heap)
            crossings: list[tuple[float, int]] = []
            for low, edge_index in active_heap:
                ex0, ey0, ex1, ey1, edge_low, edge_high = edge_segments[edge_index]
                if not (edge_low <= page_y < edge_high):
                    continue
                t = (page_y - ey0) / (ey1 - ey0)
                x_intersection = ex0 + t * (ex1 - ex0)
                crossings.append((x_intersection, 1 if ey1 > ey0 else -1))
            if not crossings:
                continue
            row = py * width * 4
            scan_spans = fill_path_crossing_spans(crossings, fill_rule)
            for start_x, end_x in scan_spans:
                span = span_pixels(start_x, end_x)
                if span is None:
                    continue
                start, end = span
                for clip_start, clip_end in visible_spans:
                    visible_start = max(start, clip_start)
                    visible_end = min(end, clip_end)
                    if visible_end <= visible_start:
                        continue
                    if simple_opaque:
                        if pixels is page_buffer:
                            page_pixels[py, visible_start:visible_end] = rgba
                        else:
                            self.pixel_array[py, visible_start:visible_end] = rgba
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    if rectangular_clip and normal_fast:
                        blend_normal_solid_span(row, visible_start, visible_end, rgba)
                        continue
                    if (
                        normal_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        blend_normal_solid_array_numpy(
                            normal_target[py, visible_start:visible_end],
                            rgba,
                        )
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    if (
                        blend_target is not None
                        and visible_end - visible_start >= RASTER_NUMPY_SPAN_MIN_PIXELS
                    ):
                        blend_solid_array_numpy(
                            blend_target[py, visible_start:visible_end],
                            rgba,
                            blend_mode,
                            semantic_context=self.semantic_context,
                        )
                        self.record_source_coverage(py, slice(visible_start, visible_end), rgba[3])
                        continue
                    for px in range(visible_start, visible_end):
                        blend_px(row + px * 4, rgba, blend_resolved_mode)

    def fast_fill_path(
        self,
        edges: list[tuple[float, float, float, float]],
        bbox: tuple[float, float, float, float],
    ) -> bool:
        blend_normal_solid_span = self.blend_normal_solid_span
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        page_box_to_pixels = self.clip.page_box_to_pixels
        scale = self.scale
        width = self.width
        x0, y0, x1, y1 = bbox
        pixel_box = page_box_to_pixels(x0, y0, x1, y1)
        if pixel_box is None:
            return True
        ix0, iy0, ix1, iy1 = pixel_box
        if ix1 - ix0 < 10 or iy1 - iy0 < 10:
            return False
        edge_bounds = [
            (ex0, ey0, ex1, ey1, min(ey1, ey0), max(ey0, ey1)) for ex0, ey0, ex1, ey1 in edges
        ]
        edge_count = len(edge_bounds)
        pending_order = sorted(range(edge_count), key=lambda i: -edge_bounds[i][5])
        pending_index = 0
        active_heap: list[tuple[float, int]] = []
        for py in range(iy0, iy1):
            scan_y = crop_y1 - (py + 0.5) / scale
            while (
                pending_index < edge_count and edge_bounds[pending_order[pending_index]][5] > scan_y
            ):
                edge_index = pending_order[pending_index]
                heapq.heappush(active_heap, (edge_bounds[edge_index][4], edge_index))
                pending_index += 1
            while active_heap and active_heap[0][0] > scan_y:
                heapq.heappop(active_heap)
            intersections: list[tuple[float, int]] = []
            for low, edge_index in active_heap:
                ex0, ey0, ex1, ey1, edge_low, edge_high = edge_bounds[edge_index]
                if not (edge_low <= scan_y < edge_high):
                    continue
                intersections.append(
                    (
                        ex0 + (scan_y - ey0) * (ex1 - ex0) / (ey1 - ey0),
                        1 if ey1 > ey0 else -1,
                    )
                )
            intersections.sort()
            winding = 0
            start_x = 0.0
            for end_x, delta in intersections:
                if winding:
                    start = max(ix0, math.ceil((start_x - crop_x0) * scale - 0.5))
                    end = min(ix1, math.ceil((end_x - crop_x0) * scale - 0.5))
                    blend_normal_solid_span(py * width * 4, start, end, (0, 0, 0, 255))
                if winding == 0:
                    start_x = end_x
                winding += delta
        return True

    def fill_path(
        self,
        path: CapturedPath,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        fill_rule: str = "nonzero",
        *,
        bbox: tuple[float, float, float, float] | None = None,
        edge_array: numpy.ndarray[Any, Any] | None = None,
    ) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        current_clip = clip.current_clip
        fast_fill_path = self.fast_fill_path
        fill_path_scanlines = self.fill_path_scanlines
        fill_rect = self.fill_rect
        pixel_in_clip = clip.pixel_in_clip
        scale = self.scale
        width = self.width
        rect = path.axis_aligned_rect()
        if rect is not None:
            fill_rect(rect, rgba, blend_mode)
            return
        edges: list[tuple[float, float, float, float]] | None
        if edge_array is None:
            edges = path.fill_edges()
            if not edges:
                return
        else:
            edges = None
            if len(edge_array) == 0:
                return
        if bbox is None:
            bbox = clip.path_bbox(path)
        if bbox is None:
            return
        fast_bbox: tuple[float, float, float, float] | None = bbox
        if clip_regions:
            if not clip_paths_are_axis_aligned_rects():
                fast_bbox = None
            else:
                clip_box = current_clip()
                if clip_box is not None:
                    fast_bbox = intersect_box(bbox, clip_box)
        if (
            rgba == (0, 0, 0, 255)
            and blend_mode is None
            and self.group_source_shape is None
            and fast_bbox is not None
            and fill_rule == "nonzero"
        ):
            if edges is None:
                edges = edge_tuples(edge_array)
            if fast_fill_path(edges, fast_bbox):
                return
        clipped_box = clipped_pixel_box(bbox)
        if clipped_box is None:
            return
        pixel_box = clipped_box[1]
        ix0, iy0, ix1, iy1 = pixel_box
        pixel_area = (ix1 - ix0) * (iy1 - iy0)
        rectangular_clip = clip_paths_are_axis_aligned_rects()
        normal_fast = blend_mode is None
        if normal_fast and rectangular_clip and fill_rule == "nonzero" and pixel_area < 10_000:
            source = (
                edge_array if edge_array is not None else numpy.asarray(edges, dtype=numpy.float64)
            )
            # The transform and the flat-edge filter fuse into the kernel's own
            # pass over the edges. None means no edge spans any y, which is the
            # early return the sloped mask used to give: coverage would be zero
            # everywhere and the blend a no-op.
            coverage = glyph_coverage_plane(
                source, crop_x0, crop_y1, scale, ix0, iy0, ix1 - ix0, iy1 - iy0
            )
            if coverage is None:
                return
            alpha_plane = numpy.rint(coverage * rgba[3]).astype(numpy.uint8)
            blend_normal_alpha_array_numpy(
                self.pixel_array[iy0:iy1, ix0:ix1],
                rgba,
                alpha_plane,
            )
            self.record_source_alpha(slice(iy0, iy1), slice(ix0, ix1), alpha_plane)
            if self.group_source_shape is not None:
                self.record_source_shape(
                    slice(iy0, iy1), slice(ix0, ix1), numpy.rint(coverage * 255).astype(numpy.uint8)
                )
            return
        if edges is None:
            edges = edge_tuples(edge_array)
        edge_segments = [
            (
                ex0,
                ey0,
                ex1,
                ey1,
                min(ey1, ey0),
                max(ey0, ey1),
            )
            for ex0, ey0, ex1, ey1 in edges
            if ey0 != ey1
        ]
        if not edge_segments:
            return
        edge_segments_array = (
            numpy.asarray(edge_segments, dtype=numpy.float64) if len(edge_segments) >= 8 else None
        )
        if pixel_area >= 10_000:
            fill_path_scanlines(edge_segments, pixel_box, rgba, blend_mode, fill_rule)
            return
        samples = 4
        track_shape = self.group_source_shape is not None
        all_row_crossings = None
        if edge_segments_array is not None:
            row_count = iy1 - iy0
            sample_offsets = (numpy.arange(samples, dtype=numpy.float64) + 0.5) / samples
            page_ys = (
                crop_y1
                - (
                    numpy.repeat(numpy.arange(iy0, iy1, dtype=numpy.float64), samples)
                    + numpy.tile(sample_offsets, row_count)
                )
                / scale
            )
            all_row_crossings = fill_path_sample_crossings_numpy(edge_segments_array, page_ys)
        for py in range(iy0, iy1):
            row = py * width * 4
            sample_spans: list[list[tuple[float, float]]] = []
            if all_row_crossings is not None:
                base = (py - iy0) * samples
                sample_spans.extend(
                    fill_path_crossing_spans(all_row_crossings[base + sy], fill_rule)
                    for sy in range(samples)
                )
            else:
                for sy in range(samples):
                    page_y = crop_y1 - (py + (sy + 0.5) / samples) / scale
                    crossings = fill_path_sample_crossings(edge_segments, page_y)
                    sample_spans.append(fill_path_crossing_spans(crossings, fill_rule))
            if normal_fast and rectangular_clip:
                deltas = [0] * (ix1 - ix0 + 1)
                covered_any = False
                for spans in sample_spans:
                    for start_x, end_x in spans:
                        span_start = (start_x - crop_x0) * scale
                        span_end = (end_x - crop_x0) * scale
                        for sx in range(samples):
                            sample_offset = (sx + 0.5) / samples
                            start = max(ix0, math.ceil(span_start - sample_offset))
                            end = min(ix1, math.ceil(span_end - sample_offset))
                            if end > start:
                                deltas[start - ix0] += 1
                                deltas[end - ix0] -= 1
                                covered_any = True
                if covered_any:
                    coverage = numpy.cumsum(numpy.asarray(deltas[:-1], dtype=numpy.int16)).astype(
                        numpy.uint8
                    )
                    target = self.pixel_array[py, ix0:ix1]
                    alpha_plane = numpy.rint(
                        coverage.astype(numpy.float32) * rgba[3] / (samples * samples)
                    ).astype(numpy.uint8)
                    blend_normal_alpha_array_numpy(
                        target,
                        rgba,
                        alpha_plane,
                    )
                    self.record_source_alpha(py, slice(ix0, ix1), alpha_plane)
                    if self.group_source_shape is not None:
                        self.record_source_shape(
                            py,
                            slice(ix0, ix1),
                            numpy.rint(
                                coverage.astype(numpy.float32) * 255 / (samples * samples)
                            ).astype(numpy.uint8),
                        )
                continue
            for px in range(ix0, ix1):
                covered = 0
                sample_x0 = crop_x0 + (px + 0.5 / samples) / scale
                sample_step = 1.0 / (samples * scale)
                for spans in sample_spans:
                    if not spans:
                        continue
                    for sx in range(samples):
                        page_x = sample_x0 + sx * sample_step
                        for start_x, end_x in spans:
                            if start_x <= page_x < end_x:
                                covered += 1
                                break
                if covered:
                    if not rectangular_clip and not pixel_in_clip(px, py):
                        continue
                    self.blend_coverage_pixel(
                        row + px * 4,
                        rgba,
                        covered,
                        samples * samples,
                        track_shape=track_shape,
                        blend_resolved_mode=blend_resolved_mode,
                    )

    def fill_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        blend_mode: str | None = None,
        line_cap: int = 0,
    ) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        clip = self.clip
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_regions = clip.regions
        clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        fill_circle = self.fill_circle
        fill_rect = self.fill_rect
        pixel_in_clip = clip.pixel_in_clip
        pixels = self.pixels
        scale = self.scale
        width = self.width
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
            half = max(0.5 / scale, float(line_width) * 0.5)
            cap_extension = half if line_cap == 2 else 0.0
            if abs(dy) <= 1e-12:
                fill_rect(
                    (
                        min(x0, x1) - cap_extension,
                        y0 - half,
                        max(x0, x1) + cap_extension,
                        y0 + half,
                    ),
                    rgba,
                    blend_mode,
                )
                if line_cap == 1:
                    fill_circle(x0, y0, half, rgba, blend_mode)
                    fill_circle(x1, y1, half, rgba, blend_mode)
            else:
                fill_rect(
                    (
                        x0 - half,
                        min(y0, y1) - cap_extension,
                        x0 + half,
                        max(y0, y1) + cap_extension,
                    ),
                    rgba,
                    blend_mode,
                )
                if line_cap == 1:
                    fill_circle(x0, y0, half, rgba, blend_mode)
                    fill_circle(x1, y1, half, rgba, blend_mode)
            return
        seg_len2 = dx * dx + dy * dy
        half = max(0.5 / scale, float(line_width) * 0.5)
        if seg_len2 <= 1e-12:
            if line_cap == 1:
                fill_circle(x0, y0, half, rgba, blend_mode)
            else:
                fill_rect(
                    (x0 - half, y0 - half, x0 + half, y0 + half),
                    rgba,
                    blend_mode,
                )
            return

        seg_len = seg_len2**0.5
        ux = dx / seg_len
        uy = dy / seg_len
        cap_extension = half if line_cap == 2 else 0.0
        box = (
            min(x0, x1) - half - abs(ux) * cap_extension,
            min(y0, y1) - half - abs(uy) * cap_extension,
            max(x0, x1) + half + abs(ux) * cap_extension,
            max(y0, y1) + half + abs(uy) * cap_extension,
        )
        clipped_box = clipped_pixel_box(box)
        if clipped_box is None:
            return
        box, pixel_box = clipped_box
        ix0, iy0, ix1, iy1 = pixel_box
        samples = 4
        sample_total = samples * samples
        half2 = half * half
        inv_seg_len2 = 1.0 / seg_len2
        projection_extension = cap_extension * seg_len
        normal_fast = blend_mode is None
        track_shape = self.group_source_shape is not None
        if (
            (not clip_regions or clip_paths_are_axis_aligned_rects())
            and normal_fast
            and (ix1 - ix0) * (iy1 - iy0) > RASTER_KERNEL_MIN_PIXEL_AREA
        ):
            x_coords = numpy.arange(ix0, ix1, dtype=numpy.float64)
            y_coords = numpy.arange(iy0, iy1, dtype=numpy.float64)
            shape_plane = (
                numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
                if self.group_source_shape is not None
                else None
            )
            alpha_plane = rasterize_unclipped_line_normal(
                pixels,
                width,
                crop_x0,
                crop_y1,
                scale,
                x0,
                y0,
                x1,
                y1,
                line_width,
                rgba,
                line_cap,
                pixel_box,
                target_pixels=self.pixel_array,
                x_coords=x_coords,
                y_coords=y_coords,
                return_source_alpha=self.group_source_alpha is not None,
                source_shape=shape_plane,
            )
            if alpha_plane is not None:
                self.record_source_alpha(slice(iy0, iy1), slice(ix0, ix1), alpha_plane)
            if shape_plane is not None:
                self.record_source_shape(slice(iy0, iy1), slice(ix0, ix1), shape_plane)
            return
        for py in range(iy0, iy1):
            row = py * width * 4
            page_y_samples = tuple(
                crop_y1 - (py + sample_offset) / scale for sample_offset in RASTER_SAMPLE_OFFSETS
            )
            for px in range(ix0, ix1):
                if clip_regions and not pixel_in_clip(px, py):
                    continue
                page_x_samples = tuple(
                    crop_x0 + (px + sample_offset) / scale
                    for sample_offset in RASTER_SAMPLE_OFFSETS
                )
                covered = 0
                if line_cap in {0, 2}:
                    cross_limit = half2 * seg_len2
                    for page_y in page_y_samples:
                        offset_y = page_y - y0
                        for page_x in page_x_samples:
                            offset_x = page_x - x0
                            projection = offset_x * dx + offset_y * dy
                            if (
                                projection < -projection_extension
                                or projection > seg_len2 + projection_extension
                            ):
                                continue
                            cross = offset_x * dy - offset_y * dx
                            if cross * cross <= cross_limit:
                                covered += 1
                else:
                    cross_limit = half2 * seg_len2
                    for page_y in page_y_samples:
                        offset_y = page_y - y0
                        for page_x in page_x_samples:
                            offset_x = page_x - x0
                            t = (offset_x * dx + offset_y * dy) * inv_seg_len2
                            if 0.0 <= t <= 1.0:
                                cross = offset_x * dy - offset_y * dx
                                if cross * cross <= cross_limit:
                                    covered += 1
                            elif t < 0.0:
                                if offset_x * offset_x + offset_y * offset_y <= half2:
                                    covered += 1
                            else:
                                end_x = page_x - x1
                                end_y = page_y - y1
                                if end_x * end_x + end_y * end_y <= half2:
                                    covered += 1
                if covered:
                    self.blend_coverage_pixel(
                        row + px * 4,
                        rgba,
                        covered,
                        sample_total,
                        track_shape=track_shape,
                        blend_resolved_mode=blend_resolved_mode,
                    )

    def fill_join(
        self,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_join: int = 0,
        blend_mode: str | None = None,
    ) -> None:
        self.fill_terminal(px, py, line_width, rgba, line_join == LineJoin.ROUND, blend_mode)

    def fill_cap(
        self,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        line_cap: int,
        blend_mode: str | None = None,
    ) -> None:
        if line_cap == LineCap.BUTT:
            return
        self.fill_terminal(px, py, line_width, rgba, line_cap == LineCap.ROUND, blend_mode)

    def fill_terminal(
        self,
        px: float,
        py: float,
        line_width: float,
        rgba: tuple[int, int, int, int],
        round_shape: bool,
        blend_mode: str | None,
    ) -> None:
        radius = max(0.5 / self.scale, float(line_width) * 0.5)
        if round_shape:
            self.fill_circle(px, py, radius, rgba, blend_mode)
        else:
            self.fill_rect(
                (px - radius, py - radius, px + radius, py + radius),
                rgba,
                blend_mode,
            )

    def stroke_path(
        self,
        path: CapturedPath,
        line_width: float,
        rgba: tuple[int, int, int, int],
        dash_pattern: tuple[list[float], float] | None = None,
        blend_mode: str | None = None,
        line_cap: int = 0,
        line_join: int = 0,
    ) -> None:
        scale = self.scale
        if self.clip.regions:
            clip_box = self.clip.current_clip()
            path_box = self.clip.path_bbox(path)
            if clip_box is not None and path_box is not None:
                stroke_pad = max(0.5 / scale, float(line_width) * 0.5)
                stroke_box = (
                    path_box[0] - stroke_pad,
                    path_box[1] - stroke_pad,
                    path_box[2] + stroke_pad,
                    path_box[3] + stroke_pad,
                )
                if intersect_box(stroke_box, clip_box) is None:
                    return
        for subpath in path.subpaths:
            if dash_pattern and dash_pattern[0]:
                self.stroke_path(
                    CapturedPath(dash_subpath(subpath, dash_pattern)),
                    line_width,
                    rgba,
                    None,
                    blend_mode,
                    line_cap,
                    line_join,
                )
                continue
            points = subpath.points
            if len(points) < 2:
                continue
            if len(points) == 2 and not subpath.closed:
                (x0, y0), (x1, y1) = points
                if (x0, y0) == (x1, y1):
                    if line_cap == LineCap.ROUND:
                        radius = line_width * 0.5 if line_width > 0.0 else 0.5 / scale
                        self.fill_path(circle_path(x0, y0, radius), rgba, blend_mode)
                    continue
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    blend_mode,
                    0,
                )
                if line_cap != 0:
                    self.fill_cap(x0, y0, line_width, rgba, line_cap, blend_mode)
                    self.fill_cap(x1, y1, line_width, rgba, line_cap, blend_mode)
                continue
            for index in range(len(points) - 1):
                x0, y0 = points[index]
                x1, y1 = points[index + 1]
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    blend_mode,
                    0,
                )
            if subpath.closed and points[0] != points[-1]:
                x0, y0 = points[-1]
                x1, y1 = points[0]
                self.fill_line(
                    x0,
                    y0,
                    x1,
                    y1,
                    line_width,
                    rgba,
                    blend_mode,
                    0,
                )
            for x, y in points[1:-1]:
                self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
            if subpath.closed:
                x, y = points[0]
                self.fill_join(x, y, line_width, rgba, line_join, blend_mode)
            elif line_cap != 0:
                self.fill_cap(
                    points[0][0],
                    points[0][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )
                self.fill_cap(
                    points[-1][0],
                    points[-1][1],
                    line_width,
                    rgba,
                    line_cap,
                    blend_mode,
                )

    def shading_box(
        self,
        data: dict[str, Any],
        shading: PreparedShading,
    ) -> tuple[float, float, float, float]:
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        scale = self.scale
        width = self.width
        box = shading.bbox
        if box is None:
            box = rect_tuple(data.get("bbox"))
        if box is None:
            box = (crop_x0, crop_y0, crop_x0 + width / scale, crop_y1)
        return normalize_rect(box)

    def paint_shading(self, data: dict[str, Any], blend_mode: str | None) -> None:
        clipped_pixel_box = self.clip.clipped_pixel_box
        blend_px = self.blend_px
        blend_resolved_mode = self.resolved_blend(blend_mode)
        clip_row_visible_spans = self.clip.clip_row_visible_spans
        crop_x0 = self.crop_x0
        crop_y1 = self.crop_y1
        scale = self.scale
        shading_box = self.shading_box
        width = self.width
        shading = prepare_shading(
            data.get("dictionary"), rendering=data.get("color_rendering", DEFAULT_COLOR_RENDERING)
        )
        if shading is None:
            return
        shading_type = shading.shading_type
        coords = shading.coords
        domain = shading.domain
        extend0 = shading.extend_start
        extend1 = shading.extend_end
        clipped_box = clipped_pixel_box(shading_box(data, shading))
        if clipped_box is None:
            return
        ix0, iy0, ix1, iy1 = clipped_box[1]
        soft_mask_alpha = data.get("soft_mask_alpha")
        fill_opacity = data.get("fill_opacity")
        shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
        domain_span = domain[1] - domain[0]
        page_x_values = [crop_x0 + (px + 0.5) / scale for px in range(ix0, ix1)]
        shading_t = axial_shading_t if shading_type == 2 else radial_shading_t
        color_model = shading.color_model
        evaluate = shading.evaluate
        color_rendering = shading.color_rendering
        # Every input to the colour is loop-invariant except `value`, and `value` repeats
        # heavily across a gradient -- exactly so in the clamped extend regions. Memoising
        # on the exact key keeps the result bit-identical to evaluating per pixel.
        rgba_cache: dict[float, tuple[int, int, int, int]] = {}
        for py in range(iy0, iy1):
            page_y = crop_y1 - (py + 0.5) / scale
            row = py * width * 4
            visible_spans = clip_row_visible_spans(py)
            if not visible_spans:
                continue
            for span_start, span_end in visible_spans:
                for px in range(max(ix0, span_start), min(ix1, span_end)):
                    page_x = page_x_values[px - ix0]
                    unit_t = shading_t(coords, page_x, page_y)
                    if unit_t is None:
                        continue
                    if unit_t < 0.0:
                        if not extend0:
                            continue
                        unit_t = 0.0
                    elif unit_t > 1.0:
                        if not extend1:
                            continue
                        unit_t = 1.0
                    value = domain[0] + unit_t * domain_span
                    rgba = rgba_cache.get(value)
                    if rgba is None:
                        rgba = shading_color_rgba(
                            color_model,
                            evaluate(value),
                            fill_opacity,
                            color_rendering,
                        )
                        if shading_alpha is not None:
                            rgba = scale_rgba_alpha(rgba, shading_alpha)
                        rgba_cache[value] = rgba
                    blend_px(row + px * 4, rgba, blend_resolved_mode)

    def paint_tiling_pattern(
        self,
        pattern: TilingPattern,
        target_data: PathPaintItem,
        blend_mode: str | None,
    ) -> bool:
        crop_x0 = self.crop_x0
        crop_y0 = self.crop_y0
        crop_y1 = self.crop_y1
        scale = self.scale
        width = self.width
        cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
        x_step = abs(pattern.x_step)
        y_step = abs(pattern.y_step)
        if x_step <= 0.0 or y_step <= 0.0:
            return False
        program = pattern.program
        if not program.drawings and not program.glyphs and not program.inline_images:
            return False
        display, cell_clip = tiling_cell(self, pattern)
        target_box = target_data.bbox or self.clip.path_bbox(target_data.path)
        if type(target_box) is list or type(target_box) is tuple:
            if len(target_box) == 4:
                try:
                    x0, y0, x1, y1 = (float(value) for value in target_box)
                except TypeError, ValueError:
                    return False
            else:
                x0, y0, x1, y1 = (
                    crop_x0,
                    crop_y0,
                    crop_x0 + width / scale,
                    crop_y1,
                )
        else:
            x0, y0, x1, y1 = crop_x0, crop_y0, crop_x0 + width / scale, crop_y1
        clip_box = self.clip.current_clip()
        if clip_box is not None:
            clipped = intersect_box((x0, y0, x1, y1), clip_box)
            if clipped is None:
                return True
            x0, y0, x1, y1 = clipped
        start_x = cell_x0 + math.floor((x0 - cell_x0) / x_step) * x_step
        start_y = cell_y0 + math.floor((y0 - cell_y0) / y_step) * y_step
        cells = 0
        y = start_y
        opacity = target_data.fill_opacity
        alpha = clamp01(opacity) if is_pdf_number(opacity) else 1.0
        if is_pdf_number(target_data.soft_mask_alpha):
            alpha *= clamp01(target_data.soft_mask_alpha)
        self.push_group(
            bytearray(len(self.pixels)),
            alpha,
            blend_mode,
            isolated=tiling_pattern_uses_normal_blends(pattern),
            alpha_is_shape=target_data.alpha_is_shape,
        )
        try:
            while y < y1 + y_step and cells < 10000:
                x = start_x
                while x < x1 + x_step and cells < 10000:
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

    def paint_fill_pattern(self, data: PathPaintItem, blend_mode: str | None) -> bool:
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
                if not isinstance(dictionary, dict):
                    return False
                shading_data = {
                    "dictionary": dictionary,
                    "bbox": data.bbox or clip_state.path_bbox(path),
                    "fill_opacity": data.fill_opacity,
                    "soft_mask_alpha": data.soft_mask_alpha,
                    "color_rendering": pattern.color_rendering,
                }
                self.paint_shading(shading_data, blend_mode)
                return True
            return self.paint_tiling_pattern(pattern, data, blend_mode)
        finally:
            if pushed_clip:
                clip_state.pop()
