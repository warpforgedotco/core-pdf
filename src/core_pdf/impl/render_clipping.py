# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy

from core_pdf.impl.caches import IdentityCache
from core_pdf.impl.capture_records import CapturedPath
from core_pdf.impl.render_grid import DeviceGrid
from core_pdf.impl.render_paths import (
    fill_path_crossing_spans,
    intersect_box,
)
from core_pdf.impl.types import Record, frozen_setattr

PixelSpan = tuple[int, int]
RowSpans = tuple[PixelSpan, ...]
RowSpanArrays = tuple[
    numpy.ndarray[Any, numpy.dtype[numpy.int64]], numpy.ndarray[Any, numpy.dtype[numpy.int64]]
]
EMPTY_CLIP_BOX = (0.0, 0.0, 0.0, 0.0)


class ClipRegion(Record):
    __slots__ = ("box", "pixel_box", "rectangular", "rows", "rows_origin")

    box: tuple[float, float, float, float] | None
    pixel_box: tuple[int, int, int, int] | None
    rectangular: bool
    rows: tuple[RowSpans, ...] | None
    rows_origin: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "box",
        "pixel_box",
        "rectangular",
        "rows",
        "rows_origin",
    )
    __match_args__ = ("box", "pixel_box", "rectangular", "rows", "rows_origin")

    def __init__(
        self,
        box: tuple[float, float, float, float] | None,
        pixel_box: tuple[int, int, int, int] | None,
        rectangular: bool,
        rows: tuple[RowSpans, ...] | None,
        rows_origin: int = 0,
    ) -> None:
        frozen_setattr(self, "box", box)
        frozen_setattr(self, "pixel_box", pixel_box)
        frozen_setattr(self, "rectangular", rectangular)
        frozen_setattr(self, "rows", rows)
        frozen_setattr(self, "rows_origin", rows_origin)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.box == other.box
            and self.pixel_box == other.pixel_box
            and self.rectangular == other.rectangular
            and self.rows == other.rows
            and self.rows_origin == other.rows_origin
        )

    def __hash__(self) -> int:
        return hash((self.box, self.pixel_box, self.rectangular, self.rows, self.rows_origin))

    @property
    def empty(self) -> bool:
        return self.pixel_box is None


def intersect_spans(
    left: RowSpans,
    right: RowSpans,
) -> RowSpans:
    left_index = 0
    right_index = 0
    intersections: list[PixelSpan] = []
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


class ClipState:
    __slots__ = (
        "regions",
        "last_box",
        "last_region",
        "last_clipped",
        "span_arrays",
        "grid",
        "crop_x0",
        "crop_y1",
        "scale",
        "width",
        "height",
    )

    def __init__(self, grid: DeviceGrid) -> None:
        self.regions: list[ClipRegion] = []
        self.last_box: tuple[float, float, float, float] | None = None
        self.last_region: ClipRegion | None = None
        self.last_clipped: (
            tuple[tuple[float, float, float, float], tuple[int, int, int, int]] | None
        ) = None
        self.grid = grid
        self.crop_x0 = grid.crop_x0
        self.crop_y1 = grid.crop_y1
        self.scale = grid.scale
        self.width = grid.width
        self.height = grid.height
        self.span_arrays: IdentityCache[RowSpanArrays] = IdentityCache()

    def row_span_arrays(self, region: ClipRegion) -> RowSpanArrays:
        cached = self.span_arrays.get_key(region, id(region))
        if cached is not None:
            return cached
        rows = region.rows or ()
        offsets = numpy.zeros(len(rows) + 1, dtype=numpy.int64)
        numpy.cumsum([len(row) for row in rows], out=offsets[1:])
        spans = numpy.asarray(
            [value for row in rows for span in row for value in span], dtype=numpy.int64
        )
        return self.span_arrays.put_key(region, id(region), (offsets, spans))

    @property
    def depth(self) -> int:
        return len(self.regions)

    def restore(self, depth: int) -> None:
        depth = max(0, depth)
        if self.span_arrays:
            self.forget_span_arrays(self.regions[depth:])
        del self.regions[depth:]

    def pop(self) -> None:
        if self.regions:
            region = self.regions.pop()
            if self.span_arrays:
                self.forget_span_arrays((region,))

    def forget_span_arrays(self, regions: Iterable[ClipRegion]) -> None:
        discard = self.span_arrays.discard
        for region in regions:
            discard(region)

    def current_region(self) -> ClipRegion | None:
        return self.regions[-1] if self.regions else None

    def rect_row_spans(
        self,
        pixel_box: tuple[int, int, int, int] | None,
        py: int,
    ) -> RowSpans:
        if pixel_box is None:
            return ()
        ix0, iy0, ix1, iy1 = pixel_box
        return ((ix0, ix1),) if iy0 <= py < iy1 else ()

    def region_row_spans(
        self,
        region: ClipRegion | None,
        py: int,
    ) -> RowSpans:
        if region is None:
            return ((0, self.width),)
        if region.rows is None:
            return self.rect_row_spans(region.pixel_box, py)
        row = py - region.rows_origin
        return region.rows[row] if 0 <= row < len(region.rows) else ()

    def path_rows_spans(
        self,
        edges: tuple[tuple[float, float, float, float], ...],
        row_start: int,
        row_stop: int,
        fill_rule: str,
    ) -> list[RowSpans]:
        crop_y1 = self.crop_y1
        scale = self.scale
        x_span = self.grid.x_span
        descending = scale > 0
        tops: list[tuple[float, int]] = []
        lows: list[float] = []
        for index, (_, y0, _, y1) in enumerate(edges):
            low = min(y1, y0)
            lows.append(low)
            top = max(y0, y1)
            if y0 != y1 and (top == top or not descending):
                tops.append((top, index))
        if descending:
            tops.sort(key=lambda top: top[0], reverse=True)
        next_top = 0
        candidates: list[int] = []
        rows: list[RowSpans] = []
        for py in range(row_start, row_stop):
            page_y = crop_y1 - (py + 0.5) / scale
            added = False
            while next_top < len(tops) and (page_y < tops[next_top][0] or not descending):
                candidates.append(tops[next_top][1])
                next_top += 1
                added = True
            if added:
                candidates.sort()
            crossings: list[tuple[float, int]] = []
            live: list[int] = []
            for index in candidates:
                if descending and page_y < lows[index]:
                    continue
                live.append(index)
                x0, y0, x1, y1 = edges[index]
                low = lows[index]
                high = max(y0, y1)
                if low <= page_y < high:
                    offset = (page_y - y0) / (y1 - y0)
                    crossings.append((x0 + offset * (x1 - x0), 1 if y1 > y0 else -1))
            candidates = live
            spans: list[PixelSpan] = []
            for start_x, end_x in fill_path_crossing_spans(crossings, fill_rule):
                span = x_span(start_x, end_x)
                if span is not None:
                    spans.append(span)
            rows.append(tuple(spans))
        return rows

    def push(self, path: CapturedPath, fill_rule: str) -> None:
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
            box = intersect_box(parent_box, path_box)
        if box is None:
            box = EMPTY_CLIP_BOX
            pixel_box = None
        else:
            pixel_box = self.grid.page_box_to_pixels(*box)

        if rect is not None and (parent is None or parent.rectangular):
            self.regions.append(ClipRegion(box, pixel_box, True, None))
            return

        rect_pixel_box = self.grid.page_box_to_pixels(*rect) if rect is not None else None
        edges = tuple(path.fill_edges()) if rect is None else ()
        row_start, row_stop = (0, 0) if pixel_box is None else (pixel_box[1], pixel_box[3])
        path_rows = (
            None
            if rect is not None
            else self.path_rows_spans(edges, row_start, row_stop, fill_rule)
        )
        rows: list[RowSpans] = []
        for py in range(row_start, row_stop):
            path_spans = (
                self.rect_row_spans(rect_pixel_box, py)
                if path_rows is None
                else path_rows[py - row_start]
            )
            rows.append(intersect_spans(self.region_row_spans(parent, py), path_spans))
        self.regions.append(ClipRegion(box, pixel_box, False, tuple(rows), row_start))

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
        if box is self.last_box and region is self.last_region:
            return self.last_clipped
        self.last_box = box
        self.last_region = region
        result = None
        if region is None or not region.empty:
            clipped = (
                box if region is None or region.box is None else intersect_box(box, region.box)
            )
            if clipped is not None:
                pixel_box = self.grid.page_box_to_pixels(*clipped)
                if pixel_box is not None:
                    result = (clipped, pixel_box)
        self.last_clipped = result
        return result

    @staticmethod
    def path_bbox(path: Any) -> tuple[float, float, float, float] | None:
        return path.bbox() if type(path) is CapturedPath else None

    def pixel_in_clip(self, px: int, py: int) -> bool:
        spans = self.clip_row_visible_spans(py)
        index = bisect_left(spans, (px + 1, -1))
        return index > 0 and spans[index - 1][0] <= px < spans[index - 1][1]

    def clip_row_visible_spans(self, py: int) -> RowSpans:
        if py < 0 or py >= self.height:
            return ()
        return self.region_row_spans(self.current_region(), py)
