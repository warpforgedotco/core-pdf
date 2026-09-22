# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, cast

from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_08_graphics.geometry import points_bbox, transform_bbox


def float_value(value: object) -> float:
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def rect_tuple(value: object) -> Rectangle | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (
                float_value(value[0]),
                float_value(value[1]),
                float_value(value[2]),
                float_value(value[3]),
            )
        except TypeError, ValueError:
            return None
    x0 = getattr(value, "x0", None)
    y0 = getattr(value, "y0", None)
    x1 = getattr(value, "x1", None)
    y1 = getattr(value, "y1", None)
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None
    try:
        return (float(x0), float(y0), float(x1), float(y1))
    except TypeError, ValueError:
        return None


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = min(a1, b1) - max(a0, b0)
    return max(0.0, overlap)


def bbox_area(bbox: Sequence[float]) -> float:
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return max(0.0, width) * max(0.0, height)


def bbox_intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    width = interval_overlap(float(left[0]), float(left[2]), float(right[0]), float(right[2]))
    height = interval_overlap(float(left[1]), float(left[3]), float(right[1]), float(right[3]))
    return width * height


def finite_rect(box: object, *, require_positive: bool = True) -> Rectangle | None:
    try:
        rect = cast("Sequence[Any]", box)
        x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    except IndexError, KeyError, TypeError, ValueError:
        return None
    if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
        return None
    if require_positive and (x1 <= x0 or y1 <= y0):
        return None
    return (x0, y0, x1, y1)


def union_bbox(left: Rectangle | None, right: Rectangle | None) -> Rectangle | None:
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


def bbox_union(boxes: Iterable[Sequence[float]]) -> Rectangle | None:
    result: Rectangle | None = None
    for box in boxes:
        result = union_bbox(result, (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    return result


def bbox_intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    return bbox_intersection_area(left, right) > 0.0


def bbox_contains(container: Sequence[float], subject: Sequence[float]) -> bool:
    return (
        subject[0] >= container[0]
        and subject[2] <= container[2]
        and subject[1] >= container[1]
        and subject[3] <= container[3]
    )


def overlap_ratio_min(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = bbox_intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0
    return intersection / max(1.0, min(bbox_area(left), bbox_area(right)))


def overlap_ratio_min_exact(left: Sequence[float], right: Sequence[float]) -> float:
    smaller_area = min(bbox_area(left), bbox_area(right))
    if smaller_area <= 0.0:
        return 0.0
    return bbox_intersection_area(left, right) / smaller_area


def horizontal_overlap_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = interval_overlap(left[0], left[2], right[0], right[2])
    return intersection / max(1.0, min(left[2] - left[0], right[2] - right[0]))


def overlap_ratio_of(subject: Sequence[float], container: Sequence[float]) -> float:
    subject_area = bbox_area(subject)
    if subject_area <= 0.0:
        return 0.0
    return bbox_intersection_area(subject, container) / subject_area


def normalize_rect(rect: Sequence[float]) -> Rectangle:
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def flip_rect_vertical(rect: Sequence[float], page_height: float) -> Rectangle:
    return (
        float(rect[0]),
        page_height - float(rect[3]),
        float(rect[2]),
        page_height - float(rect[1]),
    )


__all__ = (
    "bbox_area",
    "bbox_contains",
    "bbox_intersection_area",
    "bbox_intersects",
    "bbox_union",
    "finite_rect",
    "flip_rect_vertical",
    "horizontal_overlap_ratio",
    "float_value",
    "intersect_bbox",
    "interval_overlap",
    "normalize_rect",
    "overlap_ratio_min",
    "overlap_ratio_min_exact",
    "overlap_ratio_of",
    "points_bbox",
    "rect_tuple",
    "transform_bbox",
    "union_bbox",
)
