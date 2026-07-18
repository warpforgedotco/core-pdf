# SPDX-License-Identifier: AGPL-3.0-only
"""Generic page-space geometry primitives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import NamedTuple, Protocol, TypeAlias, TypedDict, Unpack, cast

RectTuple: TypeAlias = tuple[float, float, float, float]


class RectLike(Protocol):
    x0: float
    y0: float
    x1: float
    y1: float


class RectBoxInitKwargs(TypedDict, total=False):
    seqno: int
    fill: tuple[float, ...] | None
    fill_opacity: float | None


class RectBoxReplaceKwargs(RectBoxInitKwargs, total=False):
    x0: float
    y0: float
    x1: float
    y1: float


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def rect_tuple(value: object) -> RectTuple | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        rect = cast(Sequence[object], value)
        try:
            return (
                _float_value(rect[0]),
                _float_value(rect[1]),
                _float_value(rect[2]),
                _float_value(rect[3]),
            )
        except (TypeError, ValueError):
            return None
    x0 = getattr(value, "x0", None)
    y0 = getattr(value, "y0", None)
    x1 = getattr(value, "x1", None)
    y1 = getattr(value, "y1", None)
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    try:
        return (float(x0), float(y0), float(x1), float(y1))
    except (TypeError, ValueError):
        return None


class BBox(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_rect(cls, rect: RectTuple | RectLike) -> BBox:
        if isinstance(rect, BBox):
            return rect
        if isinstance(rect, tuple):
            rect_tuple = cast(RectTuple, rect)
            return cls(rect_tuple[0], rect_tuple[1], rect_tuple[2], rect_tuple[3])
        return cls(rect.x0, rect.y0, rect.x1, rect.y1)

    @classmethod
    def from_page_rect(cls, rect: BBox, page_height: float) -> BBox:
        x0, y1, x1, y0 = rect
        return cls(x0, page_height - y0, x1, page_height - y1)

    def merge(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersection_lengths(self, other: BBox) -> tuple[float, float, float]:
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0, 0.0
        return x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0)

    def intersection_area(self, other: BBox) -> float:
        return self.intersection_lengths(other)[2]


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
    def from_bbox(cls, bbox: RectTuple, **kwargs: Unpack[RectBoxInitKwargs]) -> RectBox:
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

        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return False

        area = (x1 - x0) * (y1 - y0)
        area_of_bbox = (self.x1 - self.x0) * (self.y1 - self.y0)
        if area_of_bbox <= 0:
            return False

        return area / area_of_bbox > occlusion_threshold

    def __iter__(self) -> Iterator[float]:
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
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def contains_rect(self, other: RectBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def replace(self, **kwargs: Unpack[RectBoxReplaceKwargs]) -> RectBox:
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
