# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: stroking paths: segments, joins and caps.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

lazy import numpy

from core_pdf.impl.capture.records import (
    CapturedPath,
)
from core_pdf.impl.render.model import (
    LineCap,
    LineJoin,
)
from core_pdf.impl.render.paths import (
    RASTER_KERNEL_MIN_PIXEL_AREA,
    circle_path,
    dash_subpath,
    intersect_box,
    rasterize_unclipped_line_normal,
    stroke_half_width,
)
from core_pdf.impl.render.raster_support import (
    subpath_columns,
)
lazy from core_pdf_cythonized import (
    stroke_polylines,
    stroke_segment_samples,
)

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
                (
                    min(x0, x1) - cap_extension,
                    y0 - half,
                    max(x0, x1) + cap_extension,
                    y0 + half,
                ),
                rgba,
                blend_mode,
            )
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


def fill_join(
    self: RasterTarget,
    px: float,
    py: float,
    line_width: float,
    rgba: tuple[int, int, int, int],
    line_join: int = 0,
    blend_mode: str | None = None,
) -> None:
    self.fill_terminal(px, py, line_width, rgba, line_join == LineJoin.ROUND, blend_mode)


def fill_cap(
    self: RasterTarget,
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
        self.fill_rect(
            (px - radius, py - radius, px + radius, py + radius),
            rgba,
            blend_mode,
        )


def stroke_path(
    self: RasterTarget,
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
            stroke_pad = stroke_half_width(scale, float(line_width))
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
            self.stroke_path(
                CapturedPath(dash_subpath(subpath, dash_pattern)),
                line_width,
                rgba,
                None,
                blend_mode,
                line_cap,
                line_join,
            )
        return
    deferred = path.deferred_columns()
    if deferred is not None:
        xs, ys, spans, outline = deferred
        ends_differ: bytes | None = None
        coincident: bytes | None = None
    else:
        xs, ys, spans, ends_differ, coincident = subpath_columns(path.subpaths)
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

    def join(x: float, y: float) -> None:
        self.fill_join(x, y, line_width, rgba, line_join, blend_mode)

    def cap(x: float, y: float) -> None:
        self.fill_cap(x, y, line_width, rgba, line_cap, blend_mode)

    def dot(x: float, y: float) -> None:
        radius = line_width * 0.5 if line_width > 0.0 else 0.5 / scale
        self.fill_path(circle_path(x, y, radius), rgba, blend_mode)

    stroke_polylines(
        self.pixel_array,
        xs,
        ys,
        spans,
        outline,
        ends_differ,
        coincident,
        self.crop_x0,
        self.crop_y1,
        scale,
        clip_mode,
        None if region is None else region.box,
        region is not None and region.empty,
        0 if region is None else region.rows_origin,
        row_offsets,
        row_spans,
        float(line_width) if native else 0.0,
        *(rgba if native else (0, 0, 0, 0)),
        native,
        bool(line_cap != 0),
        bool(line_cap == LineCap.BUTT),
        bool(line_cap == LineCap.ROUND),
        bool(line_join == LineJoin.ROUND),
        0.5,
        line,
        join,
        cap,
        dot,
        self.extend_paint_box,
    )
