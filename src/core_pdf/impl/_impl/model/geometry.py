# SPDX-License-Identifier: AGPL-3.0-only
"""Generic page-space geometry primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.types import Rectangle

if TYPE_CHECKING:
    from core_pdf.impl._impl.model.runs import TextRun


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


def union_bbox(left: Rectangle | None, right: Rectangle | None) -> Rectangle | None:
    """Smallest rectangle containing both boxes; a missing side yields the other."""
    if left is None:
        return right
    if right is None:
        return left
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def intersect_bbox(left: Rectangle | None, right: Rectangle | None) -> Rectangle | None:
    """Overlap of two boxes; a missing side yields the other, as ``union_bbox`` does.

    The result may be empty (x0 > x1); callers that care test it themselves.
    """
    if left is None:
        return right
    if right is None:
        return left
    return (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )


def points_bbox(points: Iterable[tuple[float, float]]) -> Rectangle | None:
    """Axis-aligned bounds of a point sequence, or None for no points."""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for x, y in points:
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    if x0 > x1:
        return None
    return (x0, y0, x1, y1)


def transform_bbox(bbox: Rectangle, matrix: Sequence[float]) -> Rectangle:
    """Axis-aligned bounds of ``bbox`` after the PDF matrix ``(a, b, c, d, e, f)``."""
    x0, y0, x1, y1 = bbox
    a, b, c, d, e, f = matrix
    xs = (x0 * a + y0 * c + e, x1 * a + y0 * c + e, x0 * a + y1 * c + e, x1 * a + y1 * c + e)
    ys = (x0 * b + y0 * d + f, x1 * b + y0 * d + f, x0 * b + y1 * d + f, x1 * b + y1 * d + f)
    return (min(xs), min(ys), max(xs), max(ys))


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap between the intervals ``[a0, a1]`` and ``[b0, b1]``."""
    overlap = min(a1, b1) - max(a0, b0)
    return overlap if overlap > 0.0 else 0.0


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
    intersection = interval_overlap(left[0], left[2], right[0], right[2])
    return intersection / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def overlap_ratio_of(subject: Sequence[float], container: Sequence[float]) -> float:
    """Fraction of ``subject``'s own area that ``container`` covers."""
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def page_rotation_matrix(
    rotate: int, page_width: float, page_height: float
) -> tuple[float, float, float, float, float, float]:
    """PDF matrix mapping page space into the frame displayed at ``rotate`` degrees."""
    match rotate % 360:
        case 90:
            return (0.0, -1.0, 1.0, 0.0, 0.0, page_width)
        case 180:
            return (-1.0, 0.0, 0.0, -1.0, page_width, page_height)
        case 270:
            return (0.0, 1.0, -1.0, 0.0, page_height, 0.0)
        case _:
            return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def rotate_page_runs(
    runs: list[TextRun],
    *,
    rotate: int,
    page_width: float,
    page_height: float,
) -> list[TextRun]:
    """Text runs mapped into the frame the page displays at ``rotate`` degrees."""
    rotate %= 360
    if rotate == 0:
        return runs

    matrix = page_rotation_matrix(rotate, page_width, page_height)
    a, b, c, d, e, f = matrix
    transformed: list[TextRun] = []
    for run in runs:
        x0, y0, x1, y1 = transform_bbox((run.x0, run.y0, run.x1, run.y1), matrix)
        transformed.append(
            run.replace(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                tx=run.tx * a + run.ty * c + e,
                ty=run.tx * b + run.ty * d + f,
                rotation_angle=(run.rotation_angle - rotate) % 360,
            )
        )
    return transformed


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
