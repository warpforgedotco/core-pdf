# SPDX-License-Identifier: AGPL-3.0-only
"""Raster clip-stack state and derived row visibility."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from core_pdf.impl.render.kernels import internal_make_page_geometry
from core_pdf.impl.render.paths import (
    internal_fill_path_crossing_spans,
    internal_intersect_box,
)
from core_pdf.impl.spec.s_07_content.capture import CapturedPath


class internal_ClipState:
    """Clip stack and derived row visibility.

    Lifted out of ``RenderedPage.rasterize``. ``clip_path_stack`` is shared by
    reference with the rasterizer, which pushes and pops it as the content
    stream nests; mutations clear the cached visible rows.

    Instance attributes are hoisted into locals inside the hot methods, matching
    the convention the content-stream dispatch loop already uses.
    """

    __slots__ = (
        "clip_path_stack",
        "path_edge_cache",
        "clip_visible_row_cache",
        "crop_x0",
        "crop_y1",
        "scale",
        "width",
        "height",
        "page_box_to_pixels",
        "page_x_to_pixel_span",
    )

    def __init__(
        self,
        clip_path_stack: list[tuple[CapturedPath, str]],
        *,
        crop_x0: float,
        crop_y1: float,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        self.clip_path_stack = clip_path_stack
        self.path_edge_cache: dict[int, list[tuple[float, float, float, float]]] = {}
        self.clip_visible_row_cache: dict[int, tuple[tuple[int, int], ...]] = {}
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.scale = scale
        self.width = width
        self.height = height
        self.page_box_to_pixels, self.page_x_to_pixel_span = internal_make_page_geometry(
            crop_x0, crop_y1, scale, width, height
        )

    def clip_metadata(self) -> tuple[tuple[float, float, float, float] | None, bool]:
        clip: tuple[float, float, float, float] | None = None
        rectangular = True
        axis_aligned_rect_box = self.axis_aligned_rect_box
        path_bbox = self.path_bbox
        for path, internal_rule in self.clip_path_stack:
            rect = axis_aligned_rect_box(path)
            if rect is None:
                rectangular = False
                box = path_bbox(path)
            else:
                box = rect
            if box is None:
                continue
            clip = box if clip is None else internal_intersect_box(clip, box)
            if clip is None:
                break
        return clip, rectangular

    def mark_clip_metadata_dirty(self) -> None:
        self.clip_visible_row_cache.clear()

    def current_clip(self) -> tuple[float, float, float, float] | None:
        return self.clip_metadata()[0]

    def clip_paths_are_axis_aligned_rects(self) -> bool:
        return self.clip_metadata()[1]

    def clipped_pixel_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]] | None:
        """Narrow a page box to the clip, then convert it to pixel bounds.

        Returns the clipped page box together with its pixel bounds, or None
        when the box is clipped away entirely or lands outside the raster.
        Every painter opens this way; only the sentinel each hands back to its
        own caller differs, so that part stays at the call site.

        An empty clip stack has no clip box, so it goes directly to pixel
        conversion.
        """
        if self.clip_path_stack:
            clip_box = self.current_clip()
            if clip_box is not None:
                clipped = internal_intersect_box(box, clip_box)
                if clipped is None:
                    return None
                box = clipped
        pixel_box = self.page_box_to_pixels(*box)
        if pixel_box is None:
            return None
        return box, pixel_box

    def path_bbox(self, path: Any) -> tuple[float, float, float, float] | None:
        if type(path) is not CapturedPath:
            return None
        return path.bbox()

    def axis_aligned_rect_box(self, path: CapturedPath) -> tuple[float, float, float, float] | None:
        return path.axis_aligned_rect()

    def path_edges(self, path: CapturedPath) -> list[tuple[float, float, float, float]]:
        path_edge_cache = self.path_edge_cache
        cache_key = id(path)
        cached = path_edge_cache.get(cache_key)
        if cached is not None:
            return cached
        edges = path.fill_edges()
        path_edge_cache[cache_key] = edges
        return edges

    def clip_path_row_spans(
        self, path: CapturedPath, py: int, fill_rule: str
    ) -> tuple[tuple[int, int], ...]:
        edges = self.path_edges(path)
        if not edges:
            return ()
        page_y = self.crop_y1 - (py + 0.5) / self.scale
        crossings: list[tuple[float, int]] = []
        for x0, y0, x1, y1 in edges:
            if y0 == y1:
                continue
            low = y0 if y0 < y1 else y1
            high = y1 if y1 > y0 else y0
            if not (low <= page_y < high):
                continue
            t = (page_y - y0) / (y1 - y0)
            crossings.append((x0 + t * (x1 - x0), 1 if y1 > y0 else -1))
        if not crossings:
            return ()
        spans: list[tuple[int, int]] = []
        page_x_to_pixel_span = self.page_x_to_pixel_span
        # Same winding sweep as the kernel helper; the helper drops the
        # degenerate evenodd pairs that `page_x_to_pixel_span` would reject
        # anyway. Only the mapping to pixel spans is local.
        for start_x, end_x in internal_fill_path_crossing_spans(crossings, fill_rule):
            span = page_x_to_pixel_span(start_x, end_x)
            if span is not None:
                spans.append(span)
        return tuple(spans)

    def pixel_in_clip(self, px: int, py: int) -> bool:
        spans = self.clip_row_visible_spans(py)
        if not spans:
            return False
        index = bisect_left(spans, (px + 1, -1))
        if index <= 0:
            return False
        start, end = spans[index - 1]
        return start <= px < end

    def clip_row_visible_spans(self, py: int) -> tuple[tuple[int, int], ...]:
        clip_visible_row_cache = self.clip_visible_row_cache
        cache_key = py
        cached = clip_visible_row_cache.get(cache_key)
        if cached is not None:
            return cached
        clip_path_stack = self.clip_path_stack
        if not clip_path_stack:
            clip_visible_row_cache[cache_key] = ((0, self.width),)
            return clip_visible_row_cache[cache_key]
        spans: tuple[tuple[int, int], ...] | None = None
        clip_path_row_spans = self.clip_path_row_spans
        for path, fill_rule in clip_path_stack:
            path_spans = clip_path_row_spans(path, py, fill_rule)
            if not path_spans:
                clip_visible_row_cache[cache_key] = ()
                return ()
            if spans is None:
                spans = path_spans
                continue
            left_index = 0
            right_index = 0
            merged: list[tuple[int, int]] = []
            while left_index < len(spans) and right_index < len(path_spans):
                left_start, left_end = spans[left_index]
                right_start, right_end = path_spans[right_index]
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if end > start:
                    merged.append((start, end))
                if left_end < right_end:
                    left_index += 1
                else:
                    right_index += 1
            spans = tuple(merged)
            if not spans:
                clip_visible_row_cache[cache_key] = ()
                return ()
        result = spans or ()
        clip_visible_row_cache[cache_key] = result
        return result
