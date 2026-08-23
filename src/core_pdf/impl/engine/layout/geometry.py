# SPDX-License-Identifier: AGPL-3.0-only
"""Generic page-space geometry primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Protocol, TypeAlias, TypedDict, Unpack, cast

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


def internal_float_value(value: object) -> float:
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
                internal_float_value(rect[0]),
                internal_float_value(rect[1]),
                internal_float_value(rect[2]),
                internal_float_value(rect[3]),
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


def bbox_area(bbox: Sequence[float]) -> float:
    if type(bbox) is tuple and len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)
    w = float(bbox[2]) - float(bbox[0])
    h = float(bbox[3]) - float(bbox[1])
    return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)


def bbox_intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    if type(left) is tuple and type(right) is tuple and len(left) == 4 and len(right) == 4:
        w = min(left[2], right[2]) - max(left[0], right[0])
        h = min(left[3], right[3]) - max(left[1], right[1])
        return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)
    w = min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))
    h = min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    return (w if w > 0.0 else 0.0) * (h if h > 0.0 else 0.0)


def finite_rect(box: object, *, require_positive: bool = True) -> RectTuple | None:
    """Coerce a 4-item box to finite floats, or None if it cannot represent one."""
    try:
        rect = cast("Sequence[Any]", box)
        x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    if require_positive and (x1 <= x0 or y1 <= y0):
        return None
    return (x0, y0, x1, y1)


def bbox_union(boxes: Iterable[Sequence[float]]) -> RectTuple | None:
    """Return the smallest rectangle containing every box, or None for no boxes."""
    result: RectTuple | None = None
    for box in boxes:
        x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if result is None:
            result = (x0, y0, x1, y1)
        else:
            result = (
                result[0] if result[0] < x0 else x0,
                result[1] if result[1] < y0 else y0,
                result[2] if result[2] > x1 else x1,
                result[3] if result[3] > y1 else y1,
            )
    return result


def bbox_intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    """True if ``left`` and ``right`` overlap with positive area on both axes."""
    return bbox_intersection_area(left, right) > 0.0


def bbox_contains(container: Sequence[float], subject: Sequence[float]) -> bool:
    """True if ``subject`` lies entirely within ``container``."""
    return (
        subject[0] >= container[0]
        and subject[2] <= container[2]
        and subject[1] >= container[1]
        and subject[3] <= container[3]
    )


def overlap_ratio_min(left: Sequence[float], right: Sequence[float]) -> float:
    """Intersection area relative to the smaller box (denominator floored at 1.0)."""
    intersection = bbox_intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0
    return intersection / max(1.0, min(bbox_area(left), bbox_area(right)))


def overlap_ratio_min_exact(left: Sequence[float], right: Sequence[float]) -> float:
    """Intersection area relative to the smaller positive box, without an area floor."""
    smaller_area = min(bbox_area(left), bbox_area(right))
    if smaller_area <= 0.0:
        return 0.0
    return bbox_intersection_area(left, right) / smaller_area


def overlap_ratio_of(subject: Sequence[float], container: Sequence[float]) -> float:
    """Fraction of ``subject``'s own area that ``container`` covers."""
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def flip_rect_vertical(rect: Sequence[float], page_height: float) -> RectTuple:
    """Convert between bottom-left- and top-left-origin page coordinates."""
    return (
        float(rect[0]),
        page_height - float(rect[3]),
        float(rect[2]),
        page_height - float(rect[1]),
    )


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
