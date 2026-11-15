# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: stroking paths: segments, joins and caps.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

lazy import numpy

from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath
from core_pdf.impl.render.model import LineCap, LineJoin
from core_pdf.impl.render.paths import (
    RASTER_KERNEL_MIN_PIXEL_AREA,
    circle_path,
    dash_subpath,
    intersect_box,
    rasterize_unclipped_line_normal,
    stroke_half_width,
)
from core_pdf.impl.render.raster_support import subpath_columns
from core_pdf.impl.spec.citations import cites
lazy from core_pdf_cythonized import stroke_polylines, stroke_segment_samples

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget


def fill_line(
    self: RasterTarget,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_width: float,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None = None,
) -> None:
    clipped_pixel_box = self.clip.clipped_pixel_box
    clip = self.clip
    clip_regions = clip.regions
    clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
    crop_x0 = self.crop_x0
    crop_y1 = self.crop_y1
    fill_rect = self.fill_rect
    scale = self.scale
    dx = x1 - x0
    dy = y1 - y0
    cap_extension = 0.0
    half = stroke_half_width(scale, float(line_width))
    if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
        if abs(dy) <= 1e-12:
            fill_rect(
                (min(x0, x1) - cap_extension, y0 - half, max(x0, x1) + cap_extension, y0 + half),
                rgba,
                blend_mode,
            )
        else:
            fill_rect(
                (x0 - half, min(y0, y1) - cap_extension, x0 + half, max(y0, y1) + cap_extension),
                rgba,
                blend_mode,
            )
        return
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        fill_rect((x0 - half, y0 - half, x0 + half, y0 + half), rgba, blend_mode)
        return

    seg_len = seg_len2**0.5
    ux = dx / seg_len
    uy = dy / seg_len
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
    half2 = half * half
    projection_extension = cap_extension * seg_len
    if (
        (not clip_regions or clip_paths_are_axis_aligned_rects())
        and blend_mode is None
        and (ix1 - ix0) * (iy1 - iy0) > RASTER_KERNEL_MIN_PIXEL_AREA
    ):
        shape_plane = (
            numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
            if self.group_source_shape is not None
            else None
        )
        alpha_plane = rasterize_unclipped_line_normal(
            self.pixel_array,
            crop_x0,
            crop_y1,
            scale,
            x0,
            y0,
            x1,
            y1,
            line_width,
            rgba,
            pixel_box,
            return_source_alpha=self.group_source_alpha is not None,
            source_shape=shape_plane,
        )
        if alpha_plane is not None:
            self.record_source_alpha(slice(iy0, iy1), slice(ix0, ix1), alpha_plane)
        if shape_plane is not None:
            self.record_source_shape(slice(iy0, iy1), slice(ix0, ix1), shape_plane)
        if alpha_plane is None and shape_plane is None:
            self.extend_paint_window(slice(iy0, iy1), slice(ix0, ix1))
        return
    allowed = self.clip_pixel_mask(ix0, iy0, ix1, iy1) if clip_regions else None
    counts = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    stroke_segment_samples(
        self.pixel_array,
        0,
        0,
        ix0,
        iy0,
        ix1,
        iy1,
        crop_x0,
        crop_y1,
        scale,
        x0,
        y0,
        dx,
        dy,
        seg_len2,
        half2,
        projection_extension,
        *rgba,
        allowed,
        counts,
    )
    self.blend_counts(counts, ix0, iy0, None, rgba, blend_mode)


def fill_terminal(
    self: RasterTarget,
    px: float,
    py: float,
    line_width: float,
    rgba: tuple[int, int, int, int],
    round_shape: bool,
    blend_mode: str | None,
) -> None:
    radius = stroke_half_width(self.scale, float(line_width))
    if round_shape:
        self.fill_circle(px, py, radius, rgba, blend_mode)
    else:
        self.fill_rect((px - radius, py - radius, px + radius, py + radius), rgba, blend_mode)


@cites("ISO 32000-2", "8.4.3.3", introduced="1.0", rule="butt, round and projecting square caps")
@cites("ISO 32000-2", "8.4.3.4", introduced="1.0", rule="miter, round and bevel joins")
@cites("ISO 32000-2", "8.4.3.5", introduced="1.0", rule="a miter past the miter limit is a bevel")
def stroke_path(
    self: RasterTarget,
    path: CapturedPath,
    line_width: float,
    rgba: tuple[int, int, int, int],
    dash_pattern: tuple[list[float], float] | None = None,
    blend_mode: str | None = None,
    line_cap: int = 0,
    line_join: int = 0,
    miter_limit: float = 10.0,
) -> None:
    """Stroke each subpath: its segments, the joins between them and the caps at the ends
    of an open one, which stroke_polylines works out from the segments' directions."""
    scale = self.scale
    if self.clip.regions:
        clip_box = self.clip.current_clip()
        path_box = self.clip.path_bbox(path)
        if clip_box is not None and path_box is not None:
            # A miter reaches out from its corner by half the line width times the limit.
            reach = miter_limit if line_join == LineJoin.MITER else 1.0
            stroke_pad = stroke_half_width(scale, float(line_width)) * max(reach, 1.5)
            stroke_box = (
                path_box[0] - stroke_pad,
                path_box[1] - stroke_pad,
                path_box[2] + stroke_pad,
                path_box[3] + stroke_pad,
            )
            if intersect_box(stroke_box, clip_box) is None:
                return
    if dash_pattern and dash_pattern[0]:
        for subpath in path.subpaths:
            points: list[tuple[float, float, float, float]] = []
            self.stroke_path(
                CapturedPath(dash_subpath(subpath, dash_pattern, points)),
                line_width,
                rgba,
                None,
                blend_mode,
                line_cap,
                line_join,
                miter_limit,
            )
            if line_cap == LineCap.PROJECTING_SQUARE:
                for x, y, ux, uy in points:
                    self.fill_path(square_dash(x, y, ux, uy, line_width, scale), rgba, blend_mode)
        return
    deferred = path.deferred_columns()
    if deferred is not None:
        xs, ys, spans, outline = deferred
    else:
        xs, ys, spans = subpath_columns(path.subpaths)
        outline = False
    width_type = type(line_width)
    native = (
        blend_mode is None
        and self.group_source_alpha is None
        and self.group_source_shape is None
        and (width_type is float or width_type is int)
        and all(type(channel) is int for channel in rgba)
    )
    region = self.clip.current_region()
    clip_mode = 0 if region is None else 1 if region.rectangular else 2
    row_offsets, row_spans = (
        self.clip.row_span_arrays(region) if region is not None and clip_mode == 2 else (None, None)
    )

    def line(x0: float, y0: float, x1: float, y1: float) -> None:
        self.fill_line(x0, y0, x1, y1, line_width, rgba, blend_mode)

    def polygon(coordinates: tuple[float, ...]) -> None:
        corners = list(zip(coordinates[::2], coordinates[1::2], strict=True))
        self.fill_path(CapturedPath([CapturedSubpath(corners, closed=True)]), rgba, blend_mode)

    def square(x: float, y: float) -> None:
        self.fill_terminal(x, y, line_width, rgba, False, blend_mode)

    def circle(x: float, y: float) -> None:
        self.fill_terminal(x, y, line_width, rgba, True, blend_mode)

    def dot(x: float, y: float) -> None:
        radius = line_width * 0.5 if line_width > 0.0 else 0.5 / scale
        self.fill_path(circle_path(x, y, radius), rgba, blend_mode)

    stroke_polylines(
        self.pixel_array,
        xs,
        ys,
        spans,
        outline,
        self.crop_x0,
        self.crop_y1,
        scale,
        clip_mode,
        None if region is None else region.box,
        region is not None and region.empty,
        0 if region is None else region.rows_origin,
        row_offsets,
        row_spans,
        float(line_width),
        *(rgba if native else (0, 0, 0, 0)),
        native,
        int(line_cap),
        int(line_join),
        max(1.0, float(miter_limit)),
        0.5,
        line,
        polygon,
        square,
        circle,
        dot,
        self.extend_paint_box,
    )


def square_dash(
    x: float, y: float, ux: float, uy: float, line_width: float, scale: float
) -> CapturedPath:
    """A zero-length dash's projecting square caps: the square a line width wide around
    it, along the path's direction there (ISO 32000-2 8.4.3.6)."""
    half = stroke_half_width(scale, float(line_width))
    ax, ay = ux * half, uy * half
    nx, ny = -ay, ax
    corners = [
        (x - ax + nx, y - ay + ny),
        (x + ax + nx, y + ay + ny),
        (x + ax - nx, y + ay - ny),
        (x - ax - nx, y - ay - ny),
    ]
    return CapturedPath([CapturedSubpath(corners, closed=True)])
