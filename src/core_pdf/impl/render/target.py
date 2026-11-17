# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.base.arrays import UInt8Array
from core_pdf.impl.render import raster_blend as blend
from core_pdf.impl.render import raster_coverage as coverage
from core_pdf.impl.render import raster_dispatch as dispatch
from core_pdf.impl.render import raster_fills as fills
from core_pdf.impl.render import raster_groups as groups
from core_pdf.impl.render import raster_images as images
from core_pdf.impl.render import raster_shading as shading
from core_pdf.impl.render import raster_spaces as spaces
from core_pdf.impl.render import raster_strokes as strokes
from core_pdf.impl.render.blend import blend_context
from core_pdf.impl.render.clipping import ClipState, DeviceGrid
from core_pdf.impl.render.model import PixelWindow, RasterGroup, SpaceMode
from core_pdf.impl.render.raster_support import ElementaryScratch, FloatPlane, RenderResources
from core_pdf.impl.spec.standards import SemanticContext


class RasterTarget:
    """The raster a page is painted onto: its state, in slots, and its methods, which
    live by family in the raster_* modules and are listed after __init__."""

    __slots__ = (
        "fill_memo",
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
        "grid",
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
        "resources",
        "elementary_scratch",
        "group_member_boxes",
        "space_mode",
        "space_profile",
        "space_colors",
        "space_images",
    )
    pixels: bytearray
    pixel_array: UInt8Array
    semantic_context: SemanticContext
    buffer_stack: list[RasterGroup]
    group_source_alpha: FloatPlane | None
    group_source_shape: FloatPlane | None
    paint_window: PixelWindow | None
    paint_alpha_is_shape: bool
    shape_alpha: float
    clip: ClipState
    grid: DeviceGrid
    width: int
    height: int
    scale: float
    crop_x0: float
    crop_y1: float
    page_pixels: UInt8Array
    page_buffer: bytearray
    crop_y0: float
    clip_stack: list[int]
    clip_floor: int
    group_floor: int
    scope_stack: list[tuple[int, list[int], int, int, int]]
    resources: RenderResources
    elementary_scratch: dict[int, ElementaryScratch]
    group_member_boxes: dict[int, tuple[float, float, float, float] | None] | None
    fill_memo: tuple[object, object, object, tuple[int, int, int, int]] | None
    # What the colour components hold (raster_spaces): sRGB unless inside a group that
    # blends in DeviceGray or DeviceCMYK.
    space_mode: SpaceMode
    space_profile: bytes | None
    space_colors: dict[tuple[int, int, int], tuple[int, int, int, int]]
    space_images: dict[tuple[int, int], tuple[object, UInt8Array]]

    KNOCKOUT_DISJOINT_LIMIT = 96

    def __init__(
        self,
        pixels: bytearray,
        group_alpha: float | None,
        *,
        clip: ClipState,
        page_view: UInt8Array,
        semantic_context: SemanticContext | None = None,
        resources: RenderResources | None = None,
    ) -> None:
        self.semantic_context = blend_context(semantic_context)
        self.buffer_stack = [RasterGroup(pixels, view=page_view)]
        self.sync_group_mirrors()
        self.paint_alpha_is_shape = False
        self.shape_alpha = 1.0
        grid = clip.grid
        self.clip = clip
        self.grid = grid
        self.width = grid.width
        self.height = grid.height
        self.scale = grid.scale
        self.crop_x0 = grid.crop_x0
        self.crop_y1 = grid.crop_y1
        self.page_pixels = page_view
        self.page_buffer = pixels
        self.crop_y0 = grid.crop_y0
        self.clip_stack = []
        self.clip_floor = 0
        self.group_floor = 1
        self.scope_stack = []
        self.resources = RenderResources() if resources is None else resources
        self.elementary_scratch = {}
        self.fill_memo = None
        self.group_member_boxes = None
        self.space_mode = SpaceMode.RGB
        self.space_profile = None
        self.space_colors = {}
        self.space_images = {}
        if group_alpha is not None:
            self.push_group(bytearray(len(pixels)), group_alpha, None)
            self.group_floor = len(self.buffer_stack)

    # Where paint lands: the paint window, coverage, alpha and shape planes, knockouts.
    set_shape_alpha = coverage.set_shape_alpha
    extend_paint_window = coverage.extend_paint_window
    extend_paint_box = coverage.extend_paint_box
    record_source_coverage = coverage.record_source_coverage
    record_source_alpha = coverage.record_source_alpha
    pixel_view = coverage.pixel_view
    record_source_shape = coverage.record_source_shape
    accumulate_coverage = coverage.accumulate_coverage
    record_plane = coverage.record_plane
    glyph_fill_rgba = coverage.glyph_fill_rgba
    knockout_glyph_fill = coverage.knockout_glyph_fill
    paint_glyph_fill = coverage.paint_glyph_fill
    knockout_group_region = coverage.knockout_group_region
    record_knockout_paint = coverage.record_knockout_paint

    # Transparency groups and scopes: pushing, compositing and popping them.
    group_window = groups.group_window
    composite_group = groups.composite_group
    composite_group_into = groups.composite_group_into
    push_scope = groups.push_scope
    pop_scope = groups.pop_scope
    push_scratch_group = groups.push_scratch_group
    push_group = groups.push_group
    pop_group = groups.pop_group
    sync_group_mirrors = groups.sync_group_mirrors
    detached_buffer = groups.detached_buffer

    # Blending source pixels into the target, and the clip mask they pass through.
    resolved_blend = blend.resolved_blend
    blend_px = blend.blend_px
    blend_normal_solid_span = blend.blend_normal_solid_span
    clip_pixel_mask = blend.clip_pixel_mask
    blend_rules = blend.blend_rules
    blend_counts = blend.blend_counts

    # Filling rectangles, circles, glyph bitmaps and paths.
    fill_rect = fills.fill_rect
    draw_glyph_bitmap = fills.draw_glyph_bitmap
    fill_circle = fills.fill_circle
    fill_path_scanlines = fills.fill_path_scanlines
    fast_fill_path = fills.fast_fill_path
    fill_path = fills.fill_path

    # Stroking paths: segments, joins and caps.
    fill_line = strokes.fill_line
    fill_terminal = strokes.fill_terminal
    stroke_path = strokes.stroke_path

    # Drawing images and image masks.
    blit_opaque_sampled_tiles = images.blit_opaque_sampled_tiles
    blit_affine_image = images.blit_affine_image
    blit_image = images.blit_image
    blit_image_mask = images.blit_image_mask

    # Shadings, and tiling and shading patterns.
    shading_box = shading.shading_box
    prepared_shading = shading.prepared_shading
    paint_shading = shading.paint_shading
    paint_tiling_pattern = shading.paint_tiling_pattern
    stamp_tiling_cells = shading.stamp_tiling_cells
    paint_fill_pattern = shading.paint_fill_pattern
    paint_stroke_pattern = shading.paint_stroke_pattern
    paint_pattern_over = shading.paint_pattern_over
    paint_pattern_through = shading.paint_pattern_through
    blank_sibling = shading.blank_sibling

    # Painting a display list: each item goes to the handler for its type.
    paint_items = dispatch.paint_items
    paint_item = dispatch.paint_item
    paint_display_item = dispatch.paint_display_item
    paint_scope_begin = dispatch.paint_scope_begin
    paint_clip = dispatch.paint_clip
    paint_control = dispatch.paint_control
    paint_group_begin = dispatch.paint_group_begin
    paint_glyph_bitmap = dispatch.paint_glyph_bitmap
    paint_shading_item = dispatch.paint_shading_item
    paint_typed_path = dispatch.paint_typed_path

    # Groups that blend in another colour space than their parent's.
    set_space_mode = spaces.set_space_mode
    srgb_cmyk = spaces.srgb_cmyk
    color_cmyk = spaces.color_cmyk
    space_color = spaces.space_color
    paint_rgba = spaces.paint_rgba
    item_fill_rgba = spaces.item_fill_rgba
    shading_space_color = spaces.shading_space_color
    space_samples = spaces.space_samples
    switches_space = spaces.switches_space
    paint_space_group = spaces.paint_space_group
