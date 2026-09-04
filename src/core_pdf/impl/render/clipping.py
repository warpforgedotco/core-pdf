# SPDX-License-Identifier: AGPL-3.0-only
"""Raster clip regions materialized at graphics-state transitions."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

from core_pdf.impl.render.kernels import internal_make_page_geometry
from core_pdf.impl.render.paths import internal_fill_path_crossing_spans, internal_intersect_box
from core_pdf.impl.spec.s_07_content.capture import CapturedPath

internal_PixelSpan = tuple[int, int]
internal_RowSpans = tuple[internal_PixelSpan, ...]
internal_EMPTY_CLIP_BOX = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class internal_ClipRegion:
    """The effective raster clip after one PDF clipping operation."""

    box: tuple[float, float, float, float] | None
    pixel_box: tuple[int, int, int, int] | None
    rectangular: bool
    rows: tuple[internal_RowSpans, ...] | None

    @property
    def empty(self) -> bool:
        return self.pixel_box is None


def internal_intersect_spans(
    left: internal_RowSpans,
    right: internal_RowSpans,
) -> internal_RowSpans:
    left_index = 0
    right_index = 0
    intersections: list[internal_PixelSpan] = []
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            intersections.append((start, end))
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return tuple(intersections)


class internal_ClipState:
    """The effective clip stack for one raster target.

    A clip path is converted to pixel-row spans when it enters the graphics
    state. Painting then reads the effective region directly instead of
    rebuilding path edges and intersections for every queried pixel.
    """

    __slots__ = (
        "regions",
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
        *,
        crop_x0: float,
        crop_y1: float,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        self.regions: list[internal_ClipRegion] = []
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.scale = scale
        self.width = width
        self.height = height
        self.page_box_to_pixels, self.page_x_to_pixel_span = internal_make_page_geometry(
            crop_x0, crop_y1, scale, width, height
        )

    @property
    def clip_path_stack(self) -> list[internal_ClipRegion]:
        """Expose stack presence to painters during the target migration."""
        return self.regions

    @property
    def depth(self) -> int:
        return len(self.regions)

    def restore(self, depth: int) -> None:
        del self.regions[max(0, depth) :]

    def pop(self) -> None:
        if self.regions:
            self.regions.pop()

    def current_region(self) -> internal_ClipRegion | None:
        return self.regions[-1] if self.regions else None

    def internal_rect_row_spans(
        self,
        pixel_box: tuple[int, int, int, int] | None,
        py: int,
    ) -> internal_RowSpans:
        if pixel_box is None:
            return ()
        ix0, iy0, ix1, iy1 = pixel_box
        return ((ix0, ix1),) if iy0 <= py < iy1 else ()

    def internal_region_row_spans(
        self,
        region: internal_ClipRegion | None,
        py: int,
    ) -> internal_RowSpans:
        if region is None:
            return ((0, self.width),)
        if region.rows is None:
            return self.internal_rect_row_spans(region.pixel_box, py)
        return region.rows[py]

    def internal_path_row_spans(
        self,
        edges: tuple[tuple[float, float, float, float], ...],
        py: int,
        fill_rule: str,
    ) -> internal_RowSpans:
        page_y = self.crop_y1 - (py + 0.5) / self.scale
        crossings: list[tuple[float, int]] = []
        for x0, y0, x1, y1 in edges:
            if y0 == y1:
                continue
            low = y0 if y0 < y1 else y1
            high = y1 if y1 > y0 else y0
            if low <= page_y < high:
                offset = (page_y - y0) / (y1 - y0)
                crossings.append((x0 + offset * (x1 - x0), 1 if y1 > y0 else -1))
        spans: list[internal_PixelSpan] = []
        for start_x, end_x in internal_fill_path_crossing_spans(crossings, fill_rule):
            span = self.page_x_to_pixel_span(start_x, end_x)
            if span is not None:
                spans.append(span)
        return tuple(spans)

    def push(self, path: CapturedPath, fill_rule: str) -> None:
        """Intersect ``path`` with the current region and push the result."""
        parent = self.current_region()
        rect = path.axis_aligned_rect()
        path_box = rect if rect is not None else path.bbox()
        parent_box = parent.box if parent is not None else None
        if parent is not None and parent.empty:
            box = None
        elif parent_box is None:
            box = path_box
        elif path_box is None:
            box = parent_box
        else:
            box = internal_intersect_box(parent_box, path_box)
        if box is None:
            box = internal_EMPTY_CLIP_BOX
            pixel_box = None
        else:
            pixel_box = self.page_box_to_pixels(*box)

        if rect is not None and (parent is None or parent.rectangular):
            self.regions.append(internal_ClipRegion(box, pixel_box, True, None))
            return

        rect_pixel_box = self.page_box_to_pixels(*rect) if rect is not None else None
        edges = tuple(path.fill_edges()) if rect is None else ()
        rows: list[internal_RowSpans] = []
        for py in range(self.height):
            path_spans = (
                self.internal_rect_row_spans(rect_pixel_box, py)
                if rect is not None
                else self.internal_path_row_spans(edges, py, fill_rule)
            )
            rows.append(
                internal_intersect_spans(self.internal_region_row_spans(parent, py), path_spans)
            )
        self.regions.append(internal_ClipRegion(box, pixel_box, False, tuple(rows)))

    def current_clip(self) -> tuple[float, float, float, float] | None:
        region = self.current_region()
        return region.box if region is not None else None

    def clip_paths_are_axis_aligned_rects(self) -> bool:
        region = self.current_region()
        return region is None or region.rectangular

    def clipped_pixel_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]] | None:
        region = self.current_region()
        if region is not None:
            if region.empty:
                return None
            if region.box is not None:
                clipped = internal_intersect_box(box, region.box)
                if clipped is None:
                    return None
                box = clipped
        pixel_box = self.page_box_to_pixels(*box)
        return None if pixel_box is None else (box, pixel_box)

    @staticmethod
    def path_bbox(path: Any) -> tuple[float, float, float, float] | None:
        return path.bbox() if type(path) is CapturedPath else None

    @staticmethod
    def axis_aligned_rect_box(
        path: CapturedPath,
    ) -> tuple[float, float, float, float] | None:
        return path.axis_aligned_rect()

    def pixel_in_clip(self, px: int, py: int) -> bool:
        spans = self.clip_row_visible_spans(py)
        index = bisect_left(spans, (px + 1, -1))
        return index > 0 and spans[index - 1][0] <= px < spans[index - 1][1]

    def clip_row_visible_spans(self, py: int) -> internal_RowSpans:
        if py < 0 or py >= self.height:
            return ()
        return self.internal_region_row_spans(self.current_region(), py)
