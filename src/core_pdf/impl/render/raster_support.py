# SPDX-License-Identifier: AGPL-3.0-only

"""What RasterTarget's methods share: its caches and resources, the scratch planes
of elementary groups, and helpers that need no target state."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.base.caches import ByteBudgetCache, IdentityCache
from core_pdf.impl.base.geometry import clamp01, rect_tuple
from core_pdf.impl.base.types import MISSING, MissingObject
from core_pdf.impl.capture.records import (
    CapturedPath,
    CapturedSoftMask,
    CapturedSubpath,
    TilingPattern,
)
from core_pdf.impl.graphics.device_profiles import cmyk_floats_to_srgb
from core_pdf.impl.graphics.images import PreparedImage, prepare_image
from core_pdf.impl.graphics.shading import PreparedShading, ShadingEvaluatorCache, prepare_shading
from core_pdf.impl.render.blend import (
    BlendOp,
    blend_context,
    blend_op,
    blend_visible_pixels,
    color_component,
    kernel_blend_code,
    scale_rgba_alpha,
)
from core_pdf.impl.render.commands import append_captured_program
from core_pdf.impl.render.display import DisplayList
from core_pdf.impl.render.model import (
    ClipItem,
    ControlItem,
    DisplayItem,
    GlyphBitmapItem,
    GroupBeginItem,
    ImagePaintItem,
    PathPaintItem,
    PixelWindow,
    RasterGroup,
    ScopeBeginItem,
    SoftMaskPlane,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import is_pdf_number
from core_pdf.impl.spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf.impl.spec.s_08_graphics.image_spec import ImageSource
from core_pdf.impl.spec.s_11_transparency.blend import revised_blending
from core_pdf.impl.spec.standards import SemanticContext
lazy from core_pdf_cythonized import (
    alpha_channel,
    box_downsample_blocks,
    composite_elementary_normal,
    composite_masked_normal,
    composite_nonisolated_blend,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


SoftMaskKey = tuple[int, int, tuple[float, float]]


SOFT_MASK_CACHE_BYTES = 512 << 20


PREPARED_IMAGE_CACHE_BYTES = 256 << 20


SHADING_CACHE_LIMIT = 4096


TILING_CELL_CACHE_LIMIT = 4096


type SoftMaskCache = ByteBudgetCache[SoftMaskKey, tuple[CapturedSoftMask, SoftMaskPlane | None]]


type PreparedImageCache = ByteBudgetCache[int, tuple[ImageSource, PreparedImage | None]]


type PreparedShadingCache = IdentityCache[PreparedShading | None]


type TilingCellCache = IdentityCache[tuple[DisplayList, CapturedPath]]


class RenderResources:
    __slots__ = (
        "soft_masks",
        "images",
        "tiling_cells",
        "shadings",
        "shading_evaluators",
        "active_soft_masks",
    )

    def __init__(self) -> None:
        self.soft_masks: SoftMaskCache = ByteBudgetCache(SOFT_MASK_CACHE_BYTES)
        self.images: PreparedImageCache = ByteBudgetCache(PREPARED_IMAGE_CACHE_BYTES)
        self.tiling_cells: TilingCellCache = IdentityCache(TILING_CELL_CACHE_LIMIT)
        self.shadings: PreparedShadingCache = IdentityCache(SHADING_CACHE_LIMIT)
        self.shading_evaluators: ShadingEvaluatorCache = IdentityCache(SHADING_CACHE_LIMIT)
        self.active_soft_masks: set[SoftMaskKey] = set()


def tiling_cell(
    resources: RenderResources,
    pattern: TilingPattern,
    width: float,
    height: float,
    *,
    preserve_object_boundaries: bool,
) -> tuple[DisplayList, CapturedPath]:
    cache = resources.tiling_cells
    key = (id(pattern), preserve_object_boundaries)
    cached = cache.get_key(pattern, key)
    if cached is not None:
        return cached
    cell_x0, cell_y0, cell_x1, cell_y1 = pattern.bbox
    display = DisplayList(width, height, preserve_object_boundaries=preserve_object_boundaries)
    cell_clip = CapturedPath()
    cell_clip.rect(cell_x0, cell_y0, cell_x1 - cell_x0, cell_y1 - cell_y0)
    append_captured_program(display, pattern.program, include_text=True)
    return cache.put_key(pattern, key, (display, cell_clip))


def path_cell_extent(item: PathPaintItem) -> tuple[object, float]:
    path = item.path
    box = item.bbox
    if box is None and type(path) is CapturedPath:
        box = path.bbox()
    return box, 10.0 * abs(float(item.line_width or 0.0))


def box_cell_extent(item: ImagePaintItem | GlyphBitmapItem) -> tuple[object, float]:
    return item.bbox, 0.0


CELL_EXTENTS: dict[type, Callable[[Any], tuple[object, float]]] = {
    PathPaintItem: path_cell_extent,
    ImagePaintItem: box_cell_extent,
    GlyphBitmapItem: box_cell_extent,
}


NON_PAINTING_ITEMS = frozenset({ClipItem, ControlItem, GroupBeginItem, ScopeBeginItem})


def cell_paints_nothing(
    items: Iterable[DisplayItem], cell_clip: CapturedPath, scale: float
) -> bool:
    clip_box = cell_clip.bbox()
    if clip_box is None:
        return False
    margin = 2.0 / scale if scale > 0 else 2.0
    left, bottom, right, top = clip_box
    left -= margin
    bottom -= margin
    right += margin
    top += margin
    for item in items:
        item_type = type(item)
        if item_type in NON_PAINTING_ITEMS:
            continue
        cell_extent = CELL_EXTENTS.get(item_type)
        if cell_extent is None:
            return False
        box, spread = cell_extent(item)
        extent = rect_tuple(box)
        if extent is None:
            return False
        pad = spread + margin
        if (
            extent[0] - pad <= right
            and extent[2] + pad >= left
            and extent[1] - pad <= top
            and extent[3] + pad >= bottom
        ):
            return False
    return True


def shading_color_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    opacity: Any,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[int, int, int, int]:
    alpha = color_component(opacity, 255) if is_pdf_number(opacity) else 255
    name = color_model or "DeviceRGB"
    if name.endswith("DeviceGray") or len(components) == 1:
        gray = color_component(components[0] if components else 0.0)
        return gray, gray, gray, alpha
    if name.endswith("DeviceCMYK") and len(components) >= 4:
        c, m, y, k = (clamp01(v) for v in components[:4])
        red, green, blue = cmyk_floats_to_srgb(c, m, y, k, rendering=rendering)
        return red, green, blue, alpha
    rgb = [color_component(c) for c in components[:3]]
    while len(rgb) < 3:
        rgb.append(rgb[-1] if rgb else 0)
    return rgb[0], rgb[1], rgb[2], alpha


def tiling_pattern_uses_normal_blends(
    pattern: TilingPattern, active: set[int] | None = None
) -> bool:
    if active is None:
        active = set()
    identity = id(pattern)
    if identity in active:
        return False
    active.add(identity)
    try:
        program = pattern.program
        modes = [drawing.blend_mode for drawing in program.drawings]
        modes.extend(glyph.blend_mode for glyph in program.glyphs)
        modes.extend(image.blend_mode for image in program.inline_images)
        if any(
            mode is not None and blend_op(mode, casefold=True) is not BlendOp.NORMAL
            for mode in modes
        ):
            return False
        for drawing in program.drawings:
            for nested in (drawing.fill_pattern, drawing.stroke_pattern):
                if isinstance(nested, TilingPattern) and not tiling_pattern_uses_normal_blends(
                    nested, active
                ):
                    return False
        return True
    finally:
        active.remove(identity)


type PixelBox = tuple[int, int, int, int]


EMPTY_PIXEL_BOX: PixelBox = (0, 0, 0, 0)


type FloatPlane = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


class ElementaryScratch:
    __slots__ = ("buffer", "dirty", "source_alpha", "source_shape", "synced_parent", "view")

    def __init__(self, size: int, height: int, width: int) -> None:
        self.buffer = bytearray(size)
        self.view = uint8_image_view(self.buffer, (height, width, 4))
        self.source_alpha = numpy.zeros((height, width), dtype=numpy.float32)
        self.source_shape = numpy.zeros((height, width), dtype=numpy.float32)
        self.synced_parent: RasterGroup | None = None
        self.dirty: tuple[slice, slice] | None = None


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
    return composite_nonisolated_blend(
        destination,
        rendered,
        source_alpha,
        opacity,
        mask_alpha,
        kernel_blend_code(mode),
        opacity == 1.0 and normal,
        revised_blending(blend_context(semantic_context)),
    )


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


def bounded_edges(
    edge_array: numpy.ndarray[Any, Any], *, skip_horizontal: bool = False
) -> list[tuple[float, float, float, float, float, float]]:
    """Edge rows with each edge's low and high y appended, as the scanline
    fills take them; the low and high pick exactly as min(y1, y0) and
    max(y0, y1) do."""
    ey0 = edge_array[:, 1]
    ey1 = edge_array[:, 3]
    rows = numpy.column_stack(
        (edge_array, numpy.where(ey0 < ey1, ey0, ey1), numpy.where(ey1 > ey0, ey1, ey0))
    )
    if skip_horizontal:
        rows = rows[ey0 != ey1]
    return list(map(tuple, rows.tolist()))


def subpath_columns(
    subpaths: list[CapturedSubpath],
) -> tuple[
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    numpy.ndarray[Any, numpy.dtype[numpy.float64]],
    list[tuple[int, int, bool]],
    bytes,
    bytes,
]:
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int, bool]] = []
    ends_differ = bytearray()
    coincident = bytearray()
    for subpath in subpaths:
        points = subpath.points
        start = len(xs)
        for x, y in points:
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs), bool(subpath.closed)))
        ends_differ.append(1 if len(points) >= 2 and points[0] != points[-1] else 0)
        same = False
        if len(points) == 2:
            (x0, y0), (x1, y1) = points
            same = (x0, y0) == (x1, y1)
        coincident.append(1 if same else 0)
    return (
        numpy.asarray(xs, dtype=numpy.float64),
        numpy.asarray(ys, dtype=numpy.float64),
        spans,
        bytes(ends_differ),
        bytes(coincident),
    )


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
    size = len(target.pixels)
    coverage_buffer = target.stroke_scratch
    if coverage_buffer is None or len(coverage_buffer) != size:
        coverage_buffer = bytearray(size)
    target.stroke_scratch = None
    window = PixelWindow()
    scratch = target.pixel_view(coverage_buffer)
    try:
        with target.detached_buffer(coverage_buffer, window):
            target.stroke_path(
                path, line_width, (0, 0, 0, 255), dash_pattern, None, line_cap, line_join
            )
        if window.empty:
            return
        y0, y1, x0, x1 = window.y0, window.y1, window.x0, window.x1
        coverage = scratch[y0:y1, x0:x1, 3]
        covered_rows = numpy.flatnonzero(coverage.any(axis=1))
        if covered_rows.size == 0:
            return
        covered_columns = numpy.flatnonzero(coverage.any(axis=0))
        rows = slice(y0 + int(covered_rows[0]), y0 + int(covered_rows[-1]) + 1)
        columns = slice(x0 + int(covered_columns[0]), x0 + int(covered_columns[-1]) + 1)
        coverage = scratch[rows, columns, 3].copy()
    finally:
        if not window.empty:
            scratch[window.y0 : window.y1, window.x0 : window.x1] = 0
        target.stroke_scratch = coverage_buffer
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


def prepared_image_bytes(prepared: PreparedImage | None) -> int:
    if prepared is None:
        return 0
    soft_mask = prepared.soft_mask
    return prepared.raster.array.nbytes + (0 if soft_mask is None else soft_mask.array.nbytes)


def prepared_image(resources: RenderResources, source: ImageSource) -> PreparedImage | None:
    cache = resources.images
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
    if grid.dtype == numpy.uint8:
        reduced = box_downsample_blocks(grid, row_edges, column_edges)
        return reduced.reshape(-1), target_width, target_height
    totals = numpy.add.reduceat(grid, row_edges[:-1], axis=0, dtype=numpy.uint32)
    totals = numpy.add.reduceat(totals, column_edges[:-1], axis=1, dtype=numpy.uint32)
    counts = numpy.diff(row_edges)[:, None, None] * numpy.diff(column_edges)[None, :, None]
    reduced = (totals // numpy.maximum(counts, 1)).astype(numpy.uint8)
    return reduced.reshape(-1), target_width, target_height


def sample_image_plane(
    plane: UInt8Array, u: numpy.ndarray[Any, Any], v: numpy.ndarray[Any, Any]
) -> UInt8Array:
    height, width = plane.shape
    if u.shape[0] == 1 and v.ndim == 2 and v.shape[1] == 1:
        # Separable coordinates (u by column, v by row): gather rows, then columns, the same
        # samples the broadcast fancy index takes without building two index grids.
        source_x = numpy.clip((u[0] * width).astype(numpy.intp), 0, width - 1)
        source_y = numpy.clip(((1.0 - v[:, 0]) * height).astype(numpy.intp), 0, height - 1)
        return plane[source_y][:, source_x]
    source_x = numpy.clip((u * width).astype(numpy.intp), 0, width - 1)
    source_y = numpy.clip(((1.0 - v) * height).astype(numpy.intp), 0, height - 1)
    return plane[source_y, source_x]


def shading_rgba(
    color_model: str,
    components: list[float] | tuple[float, ...],
    fill_opacity: object,
    rendering: ColorRendering,
    shading_alpha: float | None,
) -> tuple[int, int, int, int]:
    rgba = shading_color_rgba(color_model, components, fill_opacity, rendering)
    return rgba if shading_alpha is None else scale_rgba_alpha(rgba, shading_alpha)


def prepared_shading(
    resources: RenderResources, dictionary: object, rendering: ColorRendering
) -> PreparedShading | None:
    cache = resources.shadings
    key = (id(dictionary), rendering)
    cached = cache.get_key(dictionary, key, MISSING)
    if not isinstance(cached, MissingObject):
        return cached
    shading = prepare_shading(
        dictionary, rendering=rendering, evaluators=resources.shading_evaluators
    )
    return cache.put_key(dictionary, key, shading)


def resolve_soft_mask(target: RasterTarget, mask: CapturedSoftMask) -> SoftMaskPlane | None:
    resources = target.resources
    key = (id(mask.program), id(mask.transfer), mask.offset)
    cached = resources.soft_masks.get(key)
    if cached is not None:
        return cached[1]
    if key in resources.active_soft_masks:
        return None
    resources.active_soft_masks.add(key)
    result: SoftMaskPlane | None = None
    try:
        nested, view = target.blank_sibling()
        display = DisplayList(target.width, target.height, preserve_object_boundaries=True)
        append_captured_program(display, mask.program, include_text=True)
        nested.paint_items(display.items, translation=mask.offset)
        alpha, present = alpha_channel(view, mask.transfer is not None)
        alpha.setflags(write=False)
        if mask.transfer is None or present is None:
            result = SoftMaskPlane(alpha, None)
        else:
            samples = numpy.flatnonzero(present)
            values = [mask.transfer(int(sample) / 255.0)[0] for sample in samples]
            table = numpy.zeros(256, dtype=numpy.float32)
            table[samples] = numpy.asarray(values, dtype=numpy.float32)
            table.setflags(write=False)
            result = SoftMaskPlane(alpha, table)
    except Exception:
        result = None
    finally:
        resources.active_soft_masks.remove(key)
    resources.soft_masks.store(key, (mask, result), 0 if result is None else result.nbytes)
    return result
