# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: shadings, and tiling and shading patterns.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from copy import replace
from functools import partial
from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.arrays import UInt8Array, uint8_image_view
from core_pdf.impl.base.geometry import clamp01, normalize_rect, rect_tuple
from core_pdf.impl.capture.records import CapturedPath, ShadingPattern, TilingPattern
from core_pdf.impl.graphics.device_profiles import cmyk_components_to_srgb
from core_pdf.impl.graphics.shading import (
    DEVICE_COMPONENTS,
    FunctionGeometry,
    MeshGeometry,
    PreparedShading,
    SpaceEvaluator,
)
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
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, invert_affine
from core_pdf.impl.spec.s_08_graphics.pdf_function import batch_evaluator
lazy from core_pdf_cythonized import (
    composite_normal_group,
    function_values,
    mesh_values,
    patch_triangles,
    shading_blend,
    shading_values,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

# Cells one pattern paints at most, so a tiny cell over a large area stays bounded.
MAX_TILING_CELLS = 10000
# An axial or radial shading is worked out a band of at most this many pixels at a time.
SHADING_BAND_CELLS = 1 << 20
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
    background = shading_background(item, shading)
    if shading.geometry is not None:
        paint_geometry_shading(self, item, shading, blend_mode, background)
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
    if background is None:
        box = self.shading_box(item, shading)
    else:
        # The background fills the whole area painted, and BBox clips only the shading.
        box = painted_area_box(self, item)
        if to_shading is None and shading.bbox is not None:
            to_shading = IDENTITY_MATRIX
            shading_bbox = normalize_rect(shading.bbox)
    clipped_box = self.clip.clipped_pixel_box(box)
    if clipped_box is None:
        return
    ix0, iy0, ix1, iy1 = clipped_box[1]
    mode = kernel_blend_code(self.resolved_blend(blend_mode))
    domain = shading.domain
    # A band of rows at a time: each pixel's value depends on its centre alone, and the
    # bands paint in row order, so a colour that fails stops where it would over the box.
    band = max(1, SHADING_BAND_CELLS // max(1, ix1 - ix0))
    for top in range(iy0, iy1, band):
        bottom = min(iy1, top + band)
        allowed = numpy.frombuffer(
            self.clip_pixel_mask(ix0, top, ix1, bottom), dtype=numpy.uint8
        ).reshape(bottom - top, ix1 - ix0)
        values, painted = shading_values(
            shading.shading_type,
            shading.coords,
            self.crop_x0,
            self.crop_y1,
            self.scale,
            ix0,
            top,
            ix1,
            bottom,
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
        blend_shading_colors(
            self,
            item,
            shading,
            mode,
            (ix0, top),
            painted,
            ordered,
            partial(positions_of, ordered),
            allowed if background is not None else None,
            background,
            many_colors=True,
        )


def positions_of(values: Any, positions: Any) -> list[Any]:
    """values at positions, as a list."""
    return values[positions].tolist()


@cites(
    "ISO 32000-2",
    "8.7.4.3",
    introduced="1.3",
    rule="Background fills the area a shading pattern paints, not one sh paints",
)
def shading_background(item: ShadingItem, shading: PreparedShading) -> tuple[float, ...] | None:
    """The shading's Background components when it paints as a pattern; sh ignores them."""
    if shading.background is None or item.payload.get("background") is not True:
        return None
    return shading.background


def painted_area_box(self: RasterTarget, item: ShadingItem) -> tuple[float, float, float, float]:
    box = rect_tuple(item.bbox)
    return normalize_rect(box if box is not None else self.grid.page_box())


def blend_shading_colors(
    self: RasterTarget,
    item: ShadingItem,
    shading: PreparedShading,
    mode: int,
    origin: tuple[int, int],
    painted: Any,
    keys: Any,
    arguments: Callable[[Any], list[Any]],
    area: Any,
    background: tuple[float, ...] | None,
    *,
    many_colors: bool = False,
) -> None:
    """Paint each pixel the shading paints in its colour, and with a background, the rest of
    area in the background colour. keys holds one entry per painted pixel, in row order, the
    same for pixels of one colour; arguments gives the evaluator's input (a number, or a tuple
    of them) at the given positions of keys. Colours are found in the order they first
    appear; when one fails, the pixels before it are painted and its error raised. With
    many_colors, the colours of a device colour model are converted in one pass."""
    ix0, iy0 = origin
    soft_mask_alpha = item.soft_mask_alpha
    fill_opacity = item.fill_opacity
    shading_alpha = float(soft_mask_alpha) if is_pdf_number(soft_mask_alpha) else None
    color_model = shading.color_model
    rendering = shading.color_rendering
    in_space = self.space_mode is not SpaceMode.RGB
    fill = None
    if background is not None:
        try:
            fill = shading_rgba(color_model, background, fill_opacity, rendering, shading_alpha)
            if in_space:
                fill = self.shading_space_color(color_model, background, fill)
        except Exception:  # noqa: BLE001 - a colour that cannot be found paints nothing
            fill = None
    if not len(keys) and fill is None:
        return
    _, first, inverse = numpy.unique(keys, return_index=True, return_inverse=True)
    by_first = numpy.argsort(first, kind="stable")
    colors = numpy.zeros((len(first) + (fill is not None), 4), dtype=numpy.int32)
    reached = len(keys)
    color_error: Exception | None = None
    evaluate = shading.evaluator
    order = by_first.tolist()
    inputs = arguments(first[by_first]) if len(first) else []
    # A colour space other than a device one converts every colour in one pass.
    batch = evaluate.many(inputs) if isinstance(evaluate, SpaceEvaluator) and inputs else None
    palette = (
        device_palette(evaluate, color_model, inputs, batch, fill_opacity, rendering, shading_alpha)
        if many_colors and not in_space and inputs
        else None
    )
    if palette is not None:
        colors[by_first] = palette
        order = []
    for step, unique_index in enumerate(order):
        position = int(first[unique_index])
        try:
            if batch is not None:
                components = batch[step]
            else:
                value = inputs[step]
                components = evaluate(*value) if type(value) is tuple else evaluate(float(value))
            rgba = shading_rgba(color_model, components, fill_opacity, rendering, shading_alpha)
            if in_space:
                rgba = self.shading_space_color(color_model, components, rgba)
            colors[unique_index] = rgba
        except Exception as error:
            color_error = error
            reached = position
            break
    color_index = inverse.reshape(-1)[:reached]
    if fill is not None:
        # The background's pixels join the shading's, in row order, with the last colour.
        colors[-1] = fill
        shading_mask = painted.view(numpy.bool_)
        everything = shading_mask | area.view(numpy.bool_)
        index = numpy.full(painted.shape, len(colors) - 1, dtype=numpy.int64)
        index[shading_mask] = inverse.reshape(-1)
        if color_error is not None:
            # Stop where the colour that failed first appears among all the pixels.
            stop = int(numpy.flatnonzero(shading_mask.reshape(-1))[reached])
            everything.reshape(-1)[stop:] = False
        painted = everything.view(numpy.uint8)
        color_index = index[everything]
    revised, blend_error = self.blend_rules(mode)
    shape_plane = self.group_source_shape
    window, stopped = shading_blend(
        self.pixel_array,
        ix0,
        iy0,
        painted,
        numpy.ascontiguousarray(color_index, dtype=numpy.int64),
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


def device_palette(
    evaluate: Callable[..., tuple[float, ...]],
    color_model: str,
    inputs: list[Any],
    batch: list[tuple[float, ...]] | None,
    fill_opacity: object,
    rendering: ColorRendering,
    shading_alpha: float | None,
) -> Any:
    """The RGBA of each input's colour, (n, 4), found as shading_rgba finds one but for all at
    once: a mesh can have tens of thousands of colours. None when a colour fails, or the
    colour model is not a device one, so that they are found one at a time instead."""
    count = DEVICE_COMPONENTS.get(color_model)
    if count is None:
        return None
    many = batch_evaluator(evaluate) if batch is None and type(inputs[0]) is not tuple else None
    try:
        if many is not None:
            components = many(numpy.asarray(inputs, dtype=numpy.float64))
        else:
            rows = (
                batch
                if batch is not None
                else [
                    evaluate(*value) if type(value) is tuple else evaluate(value)
                    for value in inputs
                ]
            )
            components = numpy.asarray(rows, dtype=numpy.float64)
        alpha = shading_rgba(color_model, (0.0,), fill_opacity, rendering, shading_alpha)[3]
    except Exception:  # noqa: BLE001 - one at a time reports the colour that fails
        return None
    # A component that is not finite is found one at a time, as shading_rgba finds it.
    if (
        components.ndim != 2
        or len(components) != len(inputs)
        or components.shape[1] < 1
        or not numpy.isfinite(components).all()
    ):
        return None
    palette = numpy.empty((len(components), 4), dtype=numpy.int32)
    palette[:, 3] = alpha
    if color_model == "DeviceGray" or components.shape[1] == 1:
        gray = numpy.clip(numpy.rint(components[:, 0] * 255.0), 0, 255)
        palette[:, :3] = gray[:, None]
    elif color_model == "DeviceCMYK" and components.shape[1] >= count:
        cmyk = numpy.clip(numpy.rint(numpy.clip(components[:, :4], 0, 1) * 255.0), 0, 255)
        palette[:, :3] = cmyk_components_to_srgb(cmyk / 255, rendering=rendering)
    elif components.shape[1] >= 3:
        palette[:, :3] = numpy.clip(numpy.rint(components[:, :3] * 255.0), 0, 255)
    else:
        return None
    return palette


# A mesh's colour values are sampled at 1 / 255 of each component's Decode range, and a
# Function's parametric t at 1 / 1024 of its range; a type 1 shading's Function on a grid
# of at most 256 by 256 points of its Domain, and no finer than a device pixel (8.7.4.4
# lets a shading compute some points' colours and approximate the rest).
COMPONENT_LEVELS = 255
PARAMETER_LEVELS = 1024
DOMAIN_SAMPLES = 256
# A patch is divided until its triangles stray at most this many device pixels from the
# surface, and their colours at most 1 / 512 of each component's range from the bilinear
# colours, with at most PATCH_DIVISIONS parts along each side.
PATCH_TOLERANCE = 0.5
PATCH_COLOR_TOLERANCE = 1.0 / 512.0
PATCH_DIVISIONS = 64


def transformed_bounds(
    xs: Any, ys: Any, matrix: tuple[float, ...] | None
) -> tuple[float, float, float, float] | None:
    """The bounds in default user space of the points (xs, ys) of target space."""
    if not len(xs):
        return None
    if matrix is not None:
        a, b, c, d, e, f = matrix
        xs, ys = a * xs + c * ys + e, b * xs + d * ys + f
    box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    return box if all(math.isfinite(value) for value in box) else None


def geometry_bounds(
    shading: PreparedShading, matrix: tuple[float, ...] | None
) -> tuple[float, float, float, float] | None:
    """Where a type 1 or mesh shading can paint, in default user space."""
    geometry = shading.geometry
    if isinstance(geometry, FunctionGeometry):
        x0, x1, y0, y1 = geometry.domain
        a, b, c, d, e, f = geometry.matrix
        corners = numpy.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=numpy.float64)
        xs = a * corners[:, 0] + c * corners[:, 1] + e
        ys = b * corners[:, 0] + d * corners[:, 1] + f
        return transformed_bounds(xs, ys, matrix)
    if isinstance(geometry, MeshGeometry):
        points = geometry.triangles if geometry.triangles is not None else geometry.patches
        if points is None:
            return None
        return transformed_bounds(points[..., 0].reshape(-1), points[..., 1].reshape(-1), matrix)
    return None


def quantized(values: Any, low: float, high: float, levels: int) -> tuple[Any, Any]:
    """Each value's level of levels + 1 even steps from low to high, and that level's value."""
    span = high - low
    if span == 0.0 or not math.isfinite(span):
        codes = numpy.zeros(len(values), dtype=numpy.int64)
        return codes, numpy.full(len(values), low, dtype=numpy.float64)
    codes = numpy.clip(numpy.rint((values - low) / span * levels), 0, levels).astype(numpy.int64)
    return codes, low + codes * (span / levels)


def domain_samples(extent: float) -> int:
    if not math.isfinite(extent):
        return DOMAIN_SAMPLES
    return max(1, min(DOMAIN_SAMPLES, math.ceil(extent)))


@cites(
    "ISO 32000-2",
    "8.7.4.5.2",
    introduced="1.3",
    rule="a function-based shading paints its Function over Domain, mapped by Matrix",
)
def function_shading_values(
    self: RasterTarget,
    item: ShadingItem,
    geometry: FunctionGeometry,
    box: tuple[int, int, int, int],
    allowed: Any,
    inverse: tuple[float, ...] | None,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[Any, Any, Callable[[Any], list[Any]]] | None:
    to_domain = invert_affine(geometry.matrix)
    if to_domain is None:
        return None
    ix0, iy0, ix1, iy1 = box
    x0, x1, y0, y1 = geometry.domain
    values, painted = function_values(
        self.crop_x0,
        self.crop_y1,
        self.scale,
        ix0,
        iy0,
        ix1,
        iy1,
        allowed,
        inverse,
        bbox,
        to_domain,
        geometry.domain,
    )
    points = values[painted.view(numpy.bool_)]
    # The grid's step along each side of Domain is at least a device pixel.
    a, b, c, d = geometry.matrix[:4]
    outer = item.matrix if item.matrix is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    steps = []
    for dx, dy in ((a * (x1 - x0), b * (x1 - x0)), (c * (y1 - y0), d * (y1 - y0))):
        px = outer[0] * dx + outer[2] * dy
        py = outer[1] * dx + outer[3] * dy
        steps.append(domain_samples(math.hypot(px, py) * self.scale))
    columns, rows = steps
    across = numpy.clip(
        numpy.floor((points[:, 0] - x0) / ((x1 - x0) or 1.0) * columns), 0, columns - 1
    ).astype(numpy.int64)
    down = numpy.clip(
        numpy.floor((points[:, 1] - y0) / ((y1 - y0) or 1.0) * rows), 0, rows - 1
    ).astype(numpy.int64)
    sample_x = x0 + (across + 0.5) * ((x1 - x0) / columns)
    sample_y = y0 + (down + 0.5) * ((y1 - y0) / rows)

    def arguments(positions: Any) -> list[Any]:
        return list(zip(sample_x[positions].tolist(), sample_y[positions].tolist(), strict=True))

    return painted, across * rows + down, arguments


@cites(
    "ISO 32000-2",
    "8.7.4.5.5",
    introduced="1.3",
    rule="mesh shadings interpolate each triangle's vertex colours, or t, by Gouraud shading",
)
def mesh_shading_values(
    self: RasterTarget,
    item: ShadingItem,
    geometry: MeshGeometry,
    box: tuple[int, int, int, int],
    allowed: Any,
    inverse: tuple[float, ...] | None,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[Any, Any, Callable[[Any], list[Any]]] | None:
    triangles = geometry.triangles
    ranges = geometry.ranges
    if triangles is None:
        if geometry.patches is None or geometry.corners is None:
            return None
        outer = item.matrix if item.matrix is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        scale = self.scale
        linear = (outer[0] * scale, outer[1] * scale, outer[2] * scale, outer[3] * scale)
        tolerances = numpy.asarray(
            [abs(high - low) * PATCH_COLOR_TOLERANCE for low, high in ranges], dtype=numpy.float64
        )
        triangles = patch_triangles(
            geometry.patches, geometry.corners, linear, PATCH_TOLERANCE, tolerances, PATCH_DIVISIONS
        )
    ix0, iy0, ix1, iy1 = box
    values, painted = mesh_values(
        triangles,
        self.crop_x0,
        self.crop_y1,
        self.scale,
        ix0,
        iy0,
        ix1,
        iy1,
        allowed,
        item.matrix,
        inverse,
        bbox,
    )
    rows = values[painted.view(numpy.bool_)]
    levels = PARAMETER_LEVELS if geometry.parametric else COMPONENT_LEVELS
    columns = [
        quantized(rows[:, index], low, high, levels) for index, (low, high) in enumerate(ranges)
    ]
    codes = numpy.stack([code for code, _ in columns], axis=1)
    samples = numpy.stack([sample for _, sample in columns], axis=1)
    if len(ranges) * 11 <= 63:
        keys = numpy.zeros(len(codes), dtype=numpy.int64)
        for index in range(len(ranges)):
            keys = (keys << 11) | codes[:, index]
    else:
        keys = numpy.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)

    def arguments(positions: Any) -> list[Any]:
        return [tuple(row) for row in samples[positions].tolist()]

    return painted, keys, arguments


def no_arguments(positions: Any) -> list[Any]:  # noqa: ARG001
    return []


def paint_geometry_shading(
    self: RasterTarget,
    item: ShadingItem,
    shading: PreparedShading,
    blend_mode: str | None,
    background: tuple[float, ...] | None,
) -> None:
    """Paint a function-based (type 1) or mesh (types 4 to 7) shading. Its geometry is in
    its target space, which item.matrix (or the identity) maps to default user space; BBox
    clips there."""
    geometry = shading.geometry
    matrix = item.matrix
    inverse = None
    if matrix is not None:
        inverse = invert_affine(matrix)
        if inverse is None:
            return
    bbox = normalize_rect(shading.bbox) if shading.bbox is not None else None
    reach = geometry_bounds(shading, matrix)
    if background is not None:
        box: tuple[float, float, float, float] | None = painted_area_box(self, item)
    elif reach is None:
        return
    else:
        box = intersect_box(self.shading_box(item, shading), reach)
    if box is None:
        return
    clipped_box = self.clip.clipped_pixel_box(box)
    if clipped_box is None:
        return
    ix0, iy0, ix1, iy1 = clipped_box[1]
    allowed = numpy.frombuffer(self.clip_pixel_mask(ix0, iy0, ix1, iy1), dtype=numpy.uint8).reshape(
        iy1 - iy0, ix1 - ix0
    )
    found: tuple[Any, Any, Callable[[Any], list[Any]]] | None = None
    if reach is not None:
        pixel_box = (ix0, iy0, ix1, iy1)
        if isinstance(geometry, FunctionGeometry):
            found = function_shading_values(self, item, geometry, pixel_box, allowed, inverse, bbox)
        elif isinstance(geometry, MeshGeometry):
            found = mesh_shading_values(self, item, geometry, pixel_box, allowed, inverse, bbox)
    if found is None:
        found = (numpy.zeros_like(allowed), numpy.zeros(0, dtype=numpy.int64), no_arguments)
    painted, keys, arguments = found
    blend_shading_colors(
        self,
        item,
        shading,
        kernel_blend_code(self.resolved_blend(blend_mode)),
        (ix0, iy0),
        painted,
        keys,
        arguments,
        allowed if background is not None else None,
        background,
        many_colors=True,
    )


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


@cites(
    "ISO 32000-2",
    "8.7.3.1",
    introduced="1.2",
    rule="f with a pattern as its colour paints the pattern through the path's outline",
)
def paint_fill_pattern(self: RasterTarget, data: PathPaintItem, blend_mode: str | None) -> bool:
    """Paint a fill whose colour is a pattern: the path's coverage, the same a plain colour
    fill of the path has, antialiased, is the mask of a group the pattern fills over the
    path's box (8.7.1, 8.7.3.1). The current clip stays as hard as it is for a plain fill."""
    pattern = data.fill_pattern
    if not isinstance(pattern, (ShadingPattern, TilingPattern)):
        return False
    clip_state = self.clip
    path = data.path
    box = data.bbox or clip_state.path_bbox(path)
    if type(path) is not CapturedPath or not path.has_segments() or box is None:
        self.paint_pattern_over(pattern, data, box, data.fill_opacity, blend_mode)
        return True
    clipped = clip_state.clipped_pixel_box(box)
    if clipped is None:
        return True
    # The fill's coverage, drawn opaque on a blank page of its own under the same clip and
    # in the same blend mode, which choose how a plain fill antialiases. The page, the
    # coverage's alpha and the pattern's group are kept zero between pattern fills at this
    # depth, so a fill costs its pixel box, not the page.
    scratch = self.pattern_scratch_for()
    ix0, iy0, ix1, iy1 = clipped[1]
    rows, columns = slice(iy0, iy1), slice(ix0, ix1)
    coverage, view = self.blank_sibling(clip_state, scratch.page)
    try:
        coverage.fill_path(
            path,
            (255, 255, 255, 255),
            blend_mode,
            data.fill_rule or "nonzero",
            bbox=data.bbox if data.edge_array is not None else None,
            edge_array=data.edge_array,
        )
        scratch.alpha[rows, columns] = view[rows, columns, 3]
    finally:
        view[rows, columns] = 0
    del coverage, view
    # A tiling whose cells blend in other modes than Normal blends them with the backdrop.
    isolated = isinstance(pattern, ShadingPattern) or tiling_pattern_uses_normal_blends(pattern)
    try:
        self.paint_pattern_through(
            pattern,
            data,
            box,
            data.fill_opacity,
            blend_mode,
            scratch.alpha,
            isolated,
            scratch if isolated else None,
        )
    finally:
        scratch.alpha[rows, columns] = 0
    return True


def paint_pattern_through(
    self: RasterTarget,
    pattern: ShadingPattern | TilingPattern,
    data: PathPaintItem,
    box: tuple[float, float, float, float],
    opacity: float | None,
    blend_mode: str | None,
    coverage: UInt8Array,
    isolated: bool,
    scratch: PatternScratch | None = None,
) -> None:
    """Paint pattern over box at opacity in a group masked by coverage, an alpha plane of
    the page, which blends in blend_mode. An isolated group may take scratch's zeroed page,
    which it leaves zero again."""
    mask = SoftMaskPlane(coverage, None)
    self.push_group(
        bytearray(len(self.pixels)) if scratch is None else scratch.page,
        None,
        blend_mode,
        isolated=isolated,
        alpha_is_shape=data.alpha_is_shape,
        mask_alpha=mask,
    )
    group = self.buffer_stack[-1]
    limit = self.paint_limit
    self.paint_limit = coverage
    try:
        self.paint_pattern_over(pattern, data, box, opacity)
        shape = group.source_shape
        window = group.paint_window.slices()
        if shape is not None and window is not None and not group.alpha_is_shape:
            # The coverage is the paint's shape, not only its opacity: in a knockout group,
            # such as text's, the pixels it misses keep what earlier members painted (11.4.6).
            rows, columns = window
            shape[rows, columns] *= mask[rows, columns]
    finally:
        self.paint_limit = limit
        try:
            self.composite_group(self.pop_group())
        finally:
            if scratch is not None:
                # Every paint widens the group's window, so zeroing it leaves the page blank.
                window = group.paint_window.slices()
                if window is not None:
                    group.view[window] = 0


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
        data.miter_limit,
        data.stroke_matrix,
    )
    stroke_box = self.clip.clipped_pixel_box(box)
    if stroke_box is None:
        return True
    alpha = coverage_alpha(view, stroke_box[1])
    del coverage, view
    self.paint_pattern_through(pattern, data, box, data.stroke_opacity, blend_mode, alpha, True)
    return True


def paint_pattern_over(
    self: RasterTarget,
    pattern: ShadingPattern | TilingPattern,
    data: PathPaintItem,
    box: tuple[float, float, float, float] | None,
    opacity: float | None,
    blend_mode: str | None = None,
) -> None:
    """Paint pattern over box (or the page), in blend_mode, at opacity and data's soft mask
    alpha."""
    if isinstance(pattern, ShadingPattern):
        shading_data: dict[str, Any] = {
            "dictionary": pattern.dictionary,
            "bbox": box,
            "fill_opacity": opacity,
            "soft_mask_alpha": data.soft_mask_alpha,
            "color_rendering": pattern.color_rendering,
            "background": True,
        }
        if pattern.matrix is not None:
            shading_data["matrix"] = pattern.matrix
        self.paint_shading(ShadingItem.from_data(data.seqno, shading_data), blend_mode)
        return
    self.paint_tiling_pattern(pattern, replace(data, bbox=box, fill_opacity=opacity), blend_mode)


def coverage_alpha(view: UInt8Array, pixel_box: tuple[int, int, int, int]) -> UInt8Array:
    """The alpha of a blank page that coverage was drawn on within pixel_box, as a plane of
    the page whose memory is touched only there; the page can then be freed."""
    ix0, iy0, ix1, iy1 = pixel_box
    alpha = numpy.zeros(view.shape[:2], dtype=numpy.uint8)
    alpha[iy0:iy1, ix0:ix1] = view[iy0:iy1, ix0:ix1, 3]
    return alpha


class PatternScratch:
    """Zeroed planes a pattern fill at one depth of the group stack draws on and leaves zero:
    page, the blank page its coverage is drawn on and then the pattern's group, which never
    live at once, and alpha, the coverage's alpha."""

    __slots__ = ("alpha", "page")

    def __init__(self, width: int, height: int) -> None:
        self.page = bytearray(width * height * 4)
        self.alpha = numpy.zeros((height, width), dtype=numpy.uint8)


def pattern_scratch_for(self: RasterTarget) -> PatternScratch:
    depth = len(self.buffer_stack)
    scratch = self.pattern_scratch.get(depth)
    if scratch is None:
        scratch = self.pattern_scratch[depth] = PatternScratch(self.width, self.height)
    return scratch


def blank_sibling(
    self: RasterTarget, clip: ClipState | None = None, pixels: bytearray | None = None
) -> tuple[RasterTarget, UInt8Array]:
    """A blank target of the same page, unclipped or sharing clip, which it must not change,
    on pixels (zero) or a new page."""
    if pixels is None:
        pixels = bytearray(self.width * self.height * 4)
    view = uint8_image_view(pixels, (self.height, self.width, 4))
    sibling = type(self)(
        pixels,
        None,
        clip=ClipState(self.grid) if clip is None else clip,
        page_view=view,
        semantic_context=self.semantic_context,
        resources=self.resources,
    )
    return sibling, view
