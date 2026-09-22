# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable
from typing import Any, ClassVar, Self

from core_pdf.impl.capture.records import CapturedPath
from core_pdf.impl.records import Record
from core_pdf.impl.render.paths import (
    fill_path_crossing_spans,
    intersect_box,
)

frozen_setattr = object.__setattr__


PixelSpan = tuple[int, int]
RowSpans = tuple[PixelSpan, ...]
EMPTY_CLIP_BOX = (0.0, 0.0, 0.0, 0.0)


def make_page_geometry(
    crop_x0: float, crop_y1: float, scale: float, width: int, height: int
) -> tuple[
    Callable[[float, float, float, float], tuple[int, int, int, int] | None],
    Callable[[float, float], tuple[int, int] | None],
]:

    def page_box_to_pixels(
        x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, int, int, int] | None:
        ix0 = max(0, min(width, math.floor((x0 - crop_x0) * scale)))
        ix1 = max(0, min(width, math.ceil((x1 - crop_x0) * scale)))
        iy0 = max(0, min(height, math.floor((crop_y1 - y1) * scale)))
        iy1 = max(0, min(height, math.ceil((crop_y1 - y0) * scale)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        return ix0, iy0, ix1, iy1

    def page_x_to_pixel_span(start_x: float, end_x: float) -> tuple[int, int] | None:
        if end_x <= start_x:
            return None
        start = math.ceil((start_x - crop_x0) * scale - 0.5)
        end = math.ceil((end_x - crop_x0) * scale - 0.5)
        start = max(0, min(width, start))
        end = max(0, min(width, end))
        if end <= start:
            return None
        return start, end

    return page_box_to_pixels, page_x_to_pixel_span


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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"box={self.box!r}, "
            f"pixel_box={self.pixel_box!r}, "
            f"rectangular={self.rectangular!r}, "
            f"rows={self.rows!r}, "
            f"rows_origin={self.rows_origin!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        box = changes.pop("box", self.box)
        pixel_box = changes.pop("pixel_box", self.pixel_box)
        rectangular = changes.pop("rectangular", self.rectangular)
        rows = changes.pop("rows", self.rows)
        rows_origin = changes.pop("rows_origin", self.rows_origin)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(box, pixel_box, rectangular, rows, rows_origin)

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
        self.regions: list[ClipRegion] = []
        self.crop_x0 = crop_x0
        self.crop_y1 = crop_y1
        self.scale = scale
        self.width = width
        self.height = height
        self.page_box_to_pixels, self.page_x_to_pixel_span = make_page_geometry(
            crop_x0, crop_y1, scale, width, height
        )

    @property
    def depth(self) -> int:
        return len(self.regions)

    def restore(self, depth: int) -> None:
        del self.regions[max(0, depth) :]

    def pop(self) -> None:
        if self.regions:
            self.regions.pop()

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

    def path_row_spans(
        self,
        edges: tuple[tuple[float, float, float, float], ...],
        py: int,
        fill_rule: str,
    ) -> RowSpans:
        page_y = self.crop_y1 - (py + 0.5) / self.scale
        crossings: list[tuple[float, int]] = []
        for x0, y0, x1, y1 in edges:
            if y0 == y1:
                continue
            low = min(y1, y0)
            high = max(y0, y1)
            if low <= page_y < high:
                offset = (page_y - y0) / (y1 - y0)
                crossings.append((x0 + offset * (x1 - x0), 1 if y1 > y0 else -1))
        spans: list[PixelSpan] = []
        for start_x, end_x in fill_path_crossing_spans(crossings, fill_rule):
            span = self.page_x_to_pixel_span(start_x, end_x)
            if span is not None:
                spans.append(span)
        return tuple(spans)

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
            pixel_box = self.page_box_to_pixels(*box)

        if rect is not None and (parent is None or parent.rectangular):
            self.regions.append(ClipRegion(box, pixel_box, True, None))
            return

        rect_pixel_box = self.page_box_to_pixels(*rect) if rect is not None else None
        edges = tuple(path.fill_edges()) if rect is None else ()
        row_start, row_stop = (0, 0) if pixel_box is None else (pixel_box[1], pixel_box[3])
        rows: list[RowSpans] = []
        for py in range(row_start, row_stop):
            path_spans = (
                self.rect_row_spans(rect_pixel_box, py)
                if rect is not None
                else self.path_row_spans(edges, py, fill_rule)
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
        if region is not None:
            if region.empty:
                return None
            if region.box is not None:
                clipped = intersect_box(box, region.box)
                if clipped is None:
                    return None
                box = clipped
        pixel_box = self.page_box_to_pixels(*box)
        return None if pixel_box is None else (box, pixel_box)

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
