# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: stroking paths: segments, joins and caps.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

lazy import numpy

from core_pdf.impl.capture.records import CapturedPath, CapturedSubpath, StrokeMatrix
from core_pdf.impl.render.model import LineCap
from core_pdf.impl.render.paths import (
    RASTER_KERNEL_MIN_PIXEL_AREA,
    circle_path,
    dash_subpath,
    intersect_box,
    rasterize_unclipped_line_normal,
    seamless_in_pieces,
    stroke_half_width,
    stroke_reach,
)
from core_pdf.impl.render.raster_support import subpath_columns
from core_pdf.impl.spec.citations import cites
lazy from core_pdf_cythonized import stroke_coverage, stroke_polylines, stroke_segment_samples

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
    matrix: StrokeMatrix | None = None,
) -> None:
    """Stroke each subpath: its segments, the joins between them and the caps at the ends
    of an open one, which stroke_polylines works out from the segments' directions, and
    paints once (ISO 32000-2 8.5.3.2): each stroke of a coalesced path in turn, piece by
    piece only where that paints the same pixels. Under a blend mode or a transparency
    group, or matrix, the stroke's map from stroke space to page space, stroke_once paints
    it, unless it is one segment, or opaque and painted the same piece by piece."""
    scale = self.scale
    if self.clip.regions:
        clip_box = self.clip.current_clip()
        path_box = self.clip.path_bbox(path)
        if clip_box is not None and path_box is not None:
            stroke_pad = stroke_reach(scale, float(line_width), line_join, miter_limit, matrix)
            stroke_box = (
                path_box[0] - stroke_pad,
                path_box[1] - stroke_pad,
                path_box[2] + stroke_pad,
                path_box[3] + stroke_pad,
            )
            if intersect_box(stroke_box, clip_box) is None:
                return
    width_type = type(line_width)
    native = (
        blend_mode is None
        and self.group_source_alpha is None
        and self.group_source_shape is None
        and (width_type is float or width_type is int)
        and all(type(channel) is int for channel in rgba)
    )
    dashed = bool(dash_pattern and dash_pattern[0])
    # Painted once in the kernel, a coalesced path's strokes each in turn. Dashes, which
    # stroke_path strokes subpath by subpath, only when that paints the same as the
    # stroke at once, opaque along the axes; stroke_once unions all the others.
    once = (
        matrix is None
        and native
        and (not dashed or (rgba[3] >= 255 and seamless_in_pieces(path, line_join, miter_limit)))
    )
    if not once:
        pieces = (
            matrix is None
            and blend_mode is None
            and rgba[3] >= 255
            and self.group_source_shape is None
            and seamless_in_pieces(path, line_join, miter_limit)
        )
        if not pieces and (
            matrix is not None or not single_segment(self, path, dash_pattern, line_cap)
        ):
            for part in path.split_parts():
                stroke_once(
                    self,
                    part,
                    line_width,
                    rgba,
                    dash_pattern,
                    blend_mode,
                    line_cap,
                    line_join,
                    miter_limit,
                    matrix,
                )
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
        radius = stroke_half_width(scale, float(line_width))
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
        once,
        path.part_ends() if once else None,
        line,
        polygon,
        square,
        circle,
        dot,
        self.extend_paint_box,
    )


def single_segment(
    self: RasterTarget,
    path: CapturedPath,
    dash_pattern: tuple[list[float], float] | None,
    line_cap: int,
) -> bool:
    """Whether the stroke is one open segment with butt caps: a single piece, which
    painting on its own paints once."""
    if (dash_pattern and dash_pattern[0]) or line_cap != LineCap.BUTT:
        return False
    if self.group_source_shape is not None or path.subpath_count() != 1:
        return False
    deferred = path.deferred_columns()
    if deferred is not None:
        _, _, spans, outline = deferred
        start, end, closed = spans[0]
        return not outline and not closed and end - start == 2
    subpath = path.subpaths[0]
    return not subpath.closed and len(subpath.points) == 2


@cites(
    "ISO 32000-2",
    "8.5.3.2",
    introduced="1.0",
    rule="a stroke is one shape: its segments, joins, caps and dashes are painted once",
)
@cites(
    "ISO 32000-2",
    "8.4.3.2",
    introduced="1.0",
    rule="the line width and dashes are measured in user space, where the pen is a circle",
)
def stroke_once(
    self: RasterTarget,
    path: CapturedPath,
    line_width: float,
    rgba: tuple[int, int, int, int],
    dash_pattern: tuple[list[float], float] | None,
    blend_mode: str | None,
    line_cap: int,
    line_join: int,
    miter_limit: float,
    matrix: StrokeMatrix | None,
) -> None:
    """Paint the stroke as one shape: count the samples of each pixel that any of its
    segments, joins, caps or dashes covers, once however many pieces cover it, then blend
    the stroke by that coverage in one pass, so that its pieces never paint over one
    another under a constant alpha, a soft mask or a blend mode (11.7). Under matrix, the
    pieces are worked out, and the dashes measured, in stroke space, where the pen is a
    circle."""
    scale = self.scale
    path_box = path.bbox()
    if path_box is None:
        return
    pad = stroke_reach(scale, float(line_width), line_join, miter_limit, matrix)
    clipped = self.clip.clipped_pixel_box(
        (path_box[0] - pad, path_box[1] - pad, path_box[2] + pad, path_box[3] + pad)
    )
    if clipped is None:
        return
    ix0, iy0, ix1, iy1 = clipped[1]
    dashed = bool(dash_pattern and dash_pattern[0])
    deferred = path.deferred_columns() if matrix is None and not dashed else None
    squares: list[float] = []
    if deferred is not None:
        xs, ys, spans, outline = deferred
    else:
        subpaths = path.subpaths
        if matrix is not None:
            a, b, c, d = matrix
            determinant = a * d - b * c
            inverse = (d / determinant, -b / determinant, -c / determinant, a / determinant)
            subpaths = [subpath.transformed((*inverse, 0.0, 0.0)) for subpath in subpaths]
        if dash_pattern is not None and dashed:
            dashes: list[CapturedSubpath] = []
            points: list[tuple[float, float, float, float]] = []
            for subpath in subpaths:
                dashes.extend(dash_subpath(subpath, dash_pattern, points))
            subpaths = dashes
            if line_cap == LineCap.PROJECTING_SQUARE:
                squares = [value for point in points for value in point]
        xs, ys, spans = subpath_columns(subpaths)
        outline = False
    counts = numpy.zeros((iy1 - iy0, ix1 - ix0), dtype=numpy.uint8)
    if not stroke_coverage(
        counts,
        ix0,
        iy0,
        xs,
        ys,
        spans,
        outline,
        numpy.asarray(squares, dtype=numpy.float64),
        self.crop_x0,
        self.crop_y1,
        scale,
        matrix,
        float(line_width),
        int(line_cap),
        int(line_join),
        max(1.0, float(miter_limit)),
        0.5,
    ):
        return
    allowed = self.clip_pixel_mask(ix0, iy0, ix1, iy1) if self.clip.regions else None
    red, green, blue, alpha = rgba
    self.blend_counts(
        counts, ix0, iy0, allowed, (int(red), int(green), int(blue), int(alpha)), blend_mode
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
