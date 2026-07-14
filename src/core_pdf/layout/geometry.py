# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

BBox = tuple[float, float, float, float]


def bbox_tuple(rect: Any) -> BBox:
    if isinstance(rect, tuple):
        return rect
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def merge_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def bbox_intersection_lengths(a: BBox, b: BBox) -> tuple[float, float, float]:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    return x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0)


class RectBox:
    __slots__ = ("x0", "y0", "x1", "y1", "seqno", "fill", "fill_opacity")

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        seqno: int = -1,
        fill: tuple[float, ...] | None = None,
        fill_opacity: float | None = None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.seqno = seqno
        self.fill = fill
        self.fill_opacity = fill_opacity

    @classmethod
    def from_bbox(cls, bbox: tuple[float, float, float, float], **kwargs: Any) -> RectBox:
        """Create a RectBox from a (x0, y0, x1, y1) tuple."""
        return cls(bbox[0], bbox[1], bbox[2], bbox[3], **kwargs)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def is_infinite(self) -> bool:
        return False

    def get_area(self) -> float:
        return self.width * self.height

    def normalize(self) -> RectBox:
        if self.x0 <= self.x1 and self.y0 <= self.y1:
            return self
        return RectBox(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
            seqno=self.seqno,
            fill=self.fill,
            fill_opacity=self.fill_opacity,
        )

    def get_intersection_area(self, other: RectBox) -> float:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    def intersects(self, other: RectBox, occlusion_threshold: float = 0.0) -> bool:
        # Inline the intersection calculation to avoid double computation
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return False

        area = (x1 - x0) * (y1 - y0)
        area_of_bbox = (self.x1 - self.x0) * (self.y1 - self.y0)  # Inline for speed
        if area_of_bbox <= 0:
            return False

        return area / area_of_bbox > occlusion_threshold

    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> float:
        return (self.x0, self.y0, self.x1, self.y1)[index]

    def __abs__(self) -> float:
        return abs(self.get_area())

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside the rectangle."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def contains_rect(self, other: RectBox) -> bool:
        """Check if another rectangle is entirely inside this one."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def replace(self, **kwargs: Any) -> RectBox:
        """Create a new RectBox with modified fields."""
        return RectBox(
            x0=kwargs.get("x0", self.x0),
            y0=kwargs.get("y0", self.y0),
            x1=kwargs.get("x1", self.x1),
            y1=kwargs.get("y1", self.y1),
            seqno=kwargs.get("seqno", self.seqno),
            fill=kwargs.get("fill", self.fill),
            fill_opacity=kwargs.get("fill_opacity", self.fill_opacity),
        )

    def __and__(self, other: RectBox) -> RectBox:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return RectBox(0.0, 0.0, 0.0, 0.0)
        return RectBox(
            x0,
            y0,
            x1,
            y1,
            seqno=max(self.seqno, other.seqno),
            fill=self.fill if self.seqno >= other.seqno else other.fill,
            fill_opacity=(self.fill_opacity if self.seqno >= other.seqno else other.fill_opacity),
        )
