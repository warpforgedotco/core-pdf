# SPDX-License-Identifier: AGPL-3.0-only
"""Generic page-space geometry primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

from core_pdf.impl.types import Rectangle


def internal_float_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def rect_tuple(value: object) -> Rectangle | None:
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


def finite_rect(box: object, *, require_positive: bool = True) -> Rectangle | None:
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


def bbox_union(boxes: Iterable[Sequence[float]]) -> Rectangle | None:
    """Return the smallest rectangle containing every box, or None for no boxes."""
    result: Rectangle | None = None
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


def horizontal_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    """Horizontal intersection relative to the narrower box (denominator floored at 1.0)."""
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return intersection / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def overlap_ratio_of(subject: Sequence[float], container: Sequence[float]) -> float:
    """Fraction of ``subject``'s own area that ``container`` covers."""
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def flip_rect_vertical(rect: Sequence[float], page_height: float) -> Rectangle:
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

    def __iter__(self) -> Iterator[float]:
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> float:
        return (self.x0, self.y0, self.x1, self.y1)[index]
