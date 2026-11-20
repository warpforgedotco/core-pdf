# SPDX-License-Identifier: AGPL-3.0-only

"""RasterTarget methods: filling rectangles, circles, glyph bitmaps and paths.

Each function takes the target as self; RasterTarget lists them as its methods.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

lazy import numpy

from core_pdf.impl.base.values import lenient_int
from core_pdf.impl.capture.records import CapturedPath
from core_pdf.impl.render.blend import (
    RASTER_NUMPY_SPAN_MIN_PIXELS,
    blend_normal_solid_array_numpy,
    blend_solid_array_numpy,
    declared_blend,
)
from core_pdf.impl.render.paths import RASTER_CIRCLE_MIN_PIXEL_AREA, circle_path, intersect_box
lazy from core_pdf_cythonized import (
    blend_normal_alpha_array_numpy,
    fill_glyph_coverage_at,
    fill_rect_coverage,
    supersampled_coverage_plane,
)

if TYPE_CHECKING:
    from core_pdf.impl.render.target import RasterTarget

# The 4x4 route counts a band of at most this many pixels at a time, and at least this many
# rows, so its planes stay bounded on a page of any size.
SAMPLED_BAND_CELLS = 1 << 18
SAMPLED_BAND_MIN_ROWS = 8


def fill_rect(
    self: RasterTarget,
    box: tuple[float, float, float, float] | None,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None = None,
) -> None:
    """Fill the rectangle box antialiased, as fill_path fills it: paint_rect paints it
    where it can give its edges the coverage fill_path would, and otherwise it is covered
    by the 4x4 samples, the whole rectangle in one pass, or, while an edge batch is open,
    with the rest of the batch's shape."""
    if box is None:
        return
    blend_mode = declared_blend(blend_mode)
    if rect_paints_whole(self, box, rgba, blend_mode):
        paint_rect(self, box, rgba, blend_mode)
        return
    x0, y0, x1, y1 = box
    edges = ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))
    batch = self.edge_batch
    if batch is not None:
        batch.rows.extend(edges)
        batch.extend(box)
        return
    fill_edges(self, numpy.array(edges, dtype=numpy.float64), box, rgba, blend_mode, "nonzero")


class EdgeBatch:
    """The pieces of one stroke, painted piece by piece, that fill_rect, fill_path
    and fill_circle could not paint whole: their edges, filled once as one shape under the
    nonzero rule when the shape ends, so that pieces meeting inside a pixel cover it once
    rather than leaving a seam where two partial coverages are painted in turn."""

    __slots__ = ("arrays", "box", "rows")

    def __init__(self) -> None:
        self.rows: list[tuple[float, float, float, float]] = []
        self.arrays: list[numpy.ndarray[Any, Any]] = []
        self.box: tuple[float, float, float, float] | None = None

    def extend(self, box: tuple[float, float, float, float]) -> None:
        known = self.box
        self.box = (
            box
            if known is None
            else (
                min(known[0], box[0]),
                min(known[1], box[1]),
                max(known[2], box[2]),
                max(known[3], box[3]),
            )
        )

    def fill(
        self, target: RasterTarget, rgba: tuple[int, int, int, int], blend_mode: str | None
    ) -> None:
        box = self.box
        if box is None:
            return
        arrays = self.arrays
        if self.rows:
            arrays = [numpy.array(self.rows, dtype=numpy.float64), *arrays]
        edges = arrays[0] if len(arrays) == 1 else numpy.concatenate(arrays)
        fill_edges(target, edges, box, rgba, declared_blend(blend_mode), "nonzero")


def paint_as_one(
    self: RasterTarget,
    paint: Callable[[], object],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> None:
    """Call paint, which paints the pieces of one shape in rgba and blend_mode, under an
    edge batch, then fill the pieces the batch took as one shape."""
    previous = self.edge_batch
    batch = self.edge_batch = EdgeBatch()
    try:
        paint()
    finally:
        self.edge_batch = previous
    batch.fill(self, rgba, blend_mode)


def paint_rect(
    self: RasterTarget,
    box: tuple[float, float, float, float],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None = None,
) -> None:
    """Paint box, with partial coverage at its edges in the Normal mode under a rectangular
    clip and otherwise each pixel whose centre it holds whole: as fill_path would only
    where rect_paints_whole says so."""
    if rgba[3] == 255 and declared_blend(blend_mode) is None:
        blend_mode = None
    clipped_box = self.clip.clipped_pixel_box(box)
    if clipped_box is None:
        return
    (x0, y0, x1, y1), (ix0, iy0, ix1, iy1) = clipped_box
    rectangular_clip = self.clip.clip_paths_are_axis_aligned_rects()
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
        rows = slice(iy0, iy1)
        columns = slice(ix0, ix1)
        source_alpha = self.group_source_alpha
        source_shape = self.group_source_shape
        fill_rect_coverage(
            ix0,
            ix1,
            iy0,
            iy1,
            left,
            right,
            top,
            bottom,
            rgba,
            self.pixel_array[rows, columns],
            source_alpha[rows, columns] if source_alpha is not None else None,
            source_shape[rows, columns] if source_shape is not None else None,
            self.shape_alpha,
        )
        self.extend_paint_window(rows, columns)
        return
    normal_fast = blend_mode is None
    normal_target = self.pixel_array if normal_fast else None
    if rectangular_clip and normal_fast and ix1 > ix0 and iy1 > iy0:
        assert normal_target is not None
        blend_normal_solid_array_numpy(normal_target[iy0:iy1, ix0:ix1], rgba)
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
                blend_resolved_mode,
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
                    blend_resolved_mode,
                    semantic_context=self.semantic_context,
                )
                self.record_source_coverage(y, slice(start, end), rgba[3])
            else:
                for x in range(start, end):
                    blend_px(row + x * 4, rgba, blend_resolved_mode)


def draw_glyph_bitmap(
    self: RasterTarget,
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
    page_box_to_pixels = self.grid.page_box_to_pixels
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
    bitmap_h = lenient_int(bitmap_height, 0) or len(rows)
    bitmap_w = lenient_int(bitmap_width, 0) or max((row.bit_length() for row in rows), default=0)
    if bitmap_w <= 0 or bitmap_h <= 0:
        return
    cell_w = (x1 - x0) / bitmap_w
    cell_h = (y1 - y0) / bitmap_h
    if cell_w <= 0 or cell_h <= 0:
        return
    opaque_glyph = rgba[3] == 255 and declared_blend(blend_mode) is None
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
                rows[:bitmap_h] + [0] * max(0, bitmap_h - len(rows)), dtype=numpy.uint64
            )
            columns = numpy.arange(bitmap_w, dtype=numpy.uint64)
            bits = ((row_values[:, None] >> columns[None, :]) & 1).astype(bool)
            expanded = numpy.repeat(
                numpy.repeat(bits, cell_pixel_height, axis=0), cell_pixel_width, axis=1
            )
            target_pixels = self.pixel_array
            target_region = target_pixels[iy0:iy1, ix0:ix1]
            target_region[expanded] = rgba
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=expanded)
            return
    if not cells_on_pixels(self, x0, y1, cell_w, cell_h):
        # The cells meet inside pixels: the glyph is one shape, its cells filled together,
        # so that a pixel two cells share is covered once, with no seam between them.
        cells = cell_edges(tuple(rows[:bitmap_h]), bitmap_w)
        if cells is None:
            return
        across, down, (first_column, first_row, last_column, last_row) = cells
        edges = numpy.empty((len(across), 4), dtype=numpy.float64)
        edges[:, 0::2] = x0 + across * cell_w
        edges[:, 1::2] = y1 - down * cell_h
        fill_edges(
            self,
            edges,
            (
                x0 + first_column * cell_w,
                y1 - last_row * cell_h,
                x0 + last_column * cell_w,
                y1 - first_row * cell_h,
            ),
            rgba,
            declared_blend(blend_mode),
            "nonzero",
        )
        return
    # Every cell's edges on pixel edges: each cell, or run of cells, painted whole.
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
                    (x0 + run_start * cell_w, cell_y0, x0 + run_end * cell_w, cell_y1),
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


@functools.lru_cache(maxsize=256)
def cell_edges(
    rows: tuple[int, ...], bitmap_w: int
) -> (
    tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any], tuple[float, float, float, float]]
    | None
):
    """The edges of a bitmap's set cells, rows top first and bit i of a row the cell i from
    the left, in cell units: each cell's four edges in turn, as a closed subpath of its
    corners has them, in reading order, their x as columns from the left and their y as
    rows from the top, (x0, x1) and (y0, y1) of each edge; and the columns and rows they
    span, (first column, first row, last column, last row). None when no cell is set."""
    bits = numpy.array(
        [[bool(row & (1 << column)) for column in range(bitmap_w)] for row in rows], dtype=bool
    ).reshape(len(rows), bitmap_w)
    lines, columns = bits.nonzero()
    if not len(columns):
        return None
    left = columns.astype(numpy.float64)
    right = left + 1.0
    top = lines.astype(numpy.float64)
    bottom = top + 1.0
    across = numpy.stack((left, right, right, right, right, left, left, left), axis=1)
    down = numpy.stack((bottom, bottom, bottom, top, top, top, top, bottom), axis=1)
    return (
        across.reshape(-1, 2),
        down.reshape(-1, 2),
        (float(left.min()), float(top.min()), float(right.max()), float(bottom.max())),
    )


def cells_on_pixels(self: RasterTarget, x0: float, y1: float, cell_w: float, cell_h: float) -> bool:
    """Whether a grid of cells from the top left corner (x0, y1), cell_w by cell_h, has
    every edge on a pixel edge."""
    scale = self.scale
    for value in (
        (x0 - self.crop_x0) * scale,
        (self.crop_y1 - y1) * scale,
        cell_w * scale,
        cell_h * scale,
    ):
        if abs(value - round(value)) > 1e-9:
            return False
    return True


def fill_circle(
    self: RasterTarget,
    cx: float,
    cy: float,
    radius: float,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None = None,
) -> None:
    blend_mode = declared_blend(blend_mode)
    if blend_mode is not None or rgba[3] < 255 or not self.clip.clip_paths_are_axis_aligned_rects():
        # Antialiased, as the same circle filled as a path.
        self.fill_path(circle_path(cx, cy, radius), rgba, blend_mode)
        return
    clip = self.clip
    blend_px = self.blend_px
    blend_resolved_mode = self.resolved_blend(blend_mode)
    clip_regions = clip.regions
    clip_paths_are_axis_aligned_rects = clip.clip_paths_are_axis_aligned_rects
    clip_row_visible_spans = clip.clip_row_visible_spans
    crop_x0 = self.crop_x0
    crop_y1 = self.crop_y1
    current_clip = clip.current_clip
    page_box_to_pixels = self.grid.page_box_to_pixels
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
            circle_page_xs = self.grid.x_centers(ix0, ix1)
            circle_page_ys = self.grid.y_centers(iy0, iy1)
            inside = (circle_page_xs[None, :] - cx) ** 2 + (
                circle_page_ys[:, None] - cy
            ) ** 2 <= radius2
            self.pixel_array[iy0:iy1, ix0:ix1][inside] = rgba
            self.record_source_coverage(slice(iy0, iy1), slice(ix0, ix1), rgba[3], visible=inside)
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


def rect_paints_whole(
    self: RasterTarget,
    rect: tuple[float, float, float, float],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
) -> bool:
    """Whether paint_rect paints rect as fill_path would: with partial coverage at its edges,
    which it gives in the Normal mode under a rectangular clip, or where its device edges
    fall on pixel edges, so no pixel is partly covered."""
    if self.clip.clip_paths_are_axis_aligned_rects() and (
        blend_mode is None or (rgba[3] == 255 and declared_blend(blend_mode) is None)
    ):
        return True
    x0, y0, x1, y1 = rect
    scale = self.scale
    for value in (
        (x0 - self.crop_x0) * scale,
        (x1 - self.crop_x0) * scale,
        (self.crop_y1 - y1) * scale,
        (self.crop_y1 - y0) * scale,
    ):
        if abs(value - round(value)) > 1e-9:
            return False
    return True


def fill_path(
    self: RasterTarget,
    path: CapturedPath,
    rgba: tuple[int, int, int, int],
    blend_mode: str | None = None,
    fill_rule: str = "nonzero",
    *,
    bbox: tuple[float, float, float, float] | None = None,
    edge_array: numpy.ndarray[Any, Any] | None = None,
) -> None:
    """Fill path antialiased, whatever its size: a pixel's alpha is the share of it the
    path covers. Under the nonzero rule, in the Normal blend mode and a rectangular clip,
    that share is the signed area of the pixel inside the path; a row where regions of
    opposite or doubled winding meet is measured exactly instead, from the area of the
    outline's pieces alone between bands cut at its edges' ends and crossings, or, for a row
    the search for them could not afford or one of over 256 edges, sampled 4x4 by winding
    within the row. Otherwise it
    is the share of a 4x4 grid of sample points inside the path, under fill_rule. Memory and
    work follow the edges and the painted runs, not the pixel box."""
    blend_mode = declared_blend(blend_mode)
    rect = path.axis_aligned_rect()
    if rect is not None and rect_paints_whole(self, rect, rgba, blend_mode):
        paint_rect(self, rect, rgba, blend_mode)
        return
    if edge_array is None:
        edge_array = path.fill_edge_array()
    if edge_array is None:
        edges = path.fill_edges()
        if not edges:
            return
        edge_array = numpy.asarray(edges, dtype=numpy.float64)
    elif len(edge_array) == 0:
        return
    if bbox is None:
        bbox = self.clip.path_bbox(path)
    if bbox is None:
        return
    batch = self.edge_batch
    if (
        batch is not None
        and fill_rule == "nonzero"
        and (blend_mode is not None or not self.clip.clip_paths_are_axis_aligned_rects())
    ):
        # A piece of the batch's shape the 4x4 samples cover: filled with the rest of it.
        batch.arrays.append(edge_array)
        batch.extend(bbox)
        return
    fill_edges(self, edge_array, bbox, rgba, blend_mode, fill_rule)


def fill_edges(
    self: RasterTarget,
    edge_array: numpy.ndarray[Any, Any],
    bbox: tuple[float, float, float, float],
    rgba: tuple[int, int, int, int],
    blend_mode: str | None,
    fill_rule: str,
) -> None:
    """Fill the outline whose edges are edge_array's rows (x0, y0, x1, y1), within bbox,
    as fill_path describes."""
    clip = self.clip
    clipped_box = clip.clipped_pixel_box(bbox)
    if clipped_box is None:
        return
    ix0, iy0, ix1, iy1 = clipped_box[1]
    crop_x0 = self.crop_x0
    crop_y1 = self.crop_y1
    scale = self.scale
    rectangular_clip = clip.clip_paths_are_axis_aligned_rects()
    normal_fast = blend_mode is None
    if normal_fast and rectangular_clip and fill_rule == "nonzero":
        drawn = fill_glyph_coverage_at(
            edge_array,
            crop_x0,
            crop_y1,
            scale,
            ix0,
            iy0,
            ix1 - ix0,
            iy1 - iy0,
            rgba,
            self.pixel_array,
            self.group_source_alpha,
            self.group_source_shape,
            self.shape_alpha,
        )
        if drawn is not None:
            self.extend_paint_box(iy0, iy1, ix0, ix1)
        return
    # The 4x4 counts go a band of rows at a time, cut to the columns they reach, so their
    # planes stay small however large the box. The edges are sorted once by their high y, so
    # a band finds the edges that reach down to it as a prefix and keeps those reaching up to
    # it; the counts do not depend on the edges' order.
    evenodd = fill_rule == "evenodd"
    band = max(SAMPLED_BAND_MIN_ROWS, SAMPLED_BAND_CELLS // max(1, ix1 - ix0))
    if not scale > 0:
        # Rows rise up the page only for a positive scale: one band takes every edge.
        band = max(1, iy1 - iy0)
    lows = numpy.minimum(edge_array[:, 1], edge_array[:, 3])
    highs = numpy.maximum(edge_array[:, 1], edge_array[:, 3])
    by_high = numpy.argsort(-highs, kind="stable")
    sorted_edges = edge_array[by_high]
    sorted_lows = lows[by_high]
    falling_highs = -highs[by_high]
    for band_top in range(iy0, iy1, band):
        band_bottom = min(iy1, band_top + band)
        if scale > 0:
            reach = int(
                numpy.searchsorted(falling_highs, -(crop_y1 - (band_bottom + 1) / scale), "left")
            )
            near = sorted_edges[:reach][sorted_lows[:reach] < crop_y1 - (band_top - 1) / scale]
        else:
            near = sorted_edges
        if len(near) == 0:
            continue
        sampled = supersampled_coverage_plane(
            near, crop_x0, crop_y1, scale, ix0, band_top, ix1, band_bottom, evenodd
        )
        if sampled is None:
            continue
        counts, first_row, first_column = sampled
        top = band_top + first_row
        left = ix0 + first_column
        bottom = top + counts.shape[0]
        right = left + counts.shape[1]
        if normal_fast and rectangular_clip:
            rows = slice(top, bottom)
            columns = slice(left, right)
            coverage = counts.astype(numpy.float32)
            alpha_plane = numpy.rint(coverage * rgba[3] / 16).astype(numpy.uint8)
            blend_normal_alpha_array_numpy(self.pixel_array[rows, columns], rgba, alpha_plane)
            self.record_source_alpha(rows, columns, alpha_plane)
            if self.group_source_shape is not None:
                self.record_source_shape(
                    rows, columns, numpy.rint(coverage * 255 / 16).astype(numpy.uint8)
                )
            continue
        self.blend_counts(
            counts,
            left,
            top,
            None if rectangular_clip else self.clip_pixel_mask(left, top, right, bottom),
            rgba,
            blend_mode,
        )
