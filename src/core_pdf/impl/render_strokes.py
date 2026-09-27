# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any

import numpy

from core_pdf.impl.capture_records import CapturedPath, CapturedSubpath
from core_pdf.impl.render_blend import blend_visible_pixels
from core_pdf.impl.render_fills import RasterFills
from core_pdf.impl.render_model import LineCap, LineJoin, PixelWindow
from core_pdf.impl.render_paths import (
    RASTER_KERNEL_MIN_PIXEL_AREA,
    circle_path,
    dash_subpath,
    intersect_box,
    rasterize_unclipped_line_normal,
    stroke_half_width,
)
from core_pdf_cythonized import stroke_polylines, stroke_segment_samples


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
    target: RasterStrokes,
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


class RasterStrokes(RasterFills):
    __slots__ = ()

    def fill_line(
        self,
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
            self.clip.row_span_arrays(region)
            if region is not None and clip_mode == 2
            else (None, None)
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
