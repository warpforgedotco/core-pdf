"""Geometry-only detection of vector chart regions.

This module intentionally does not inspect chart text.  Text is often the only
thing that makes a PDF look document-specific, while repeated vector marks are
the reusable signal shared by bars, points, and chart grid lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChartRegion:
    """A page-space region containing a cluster of chart-like vector marks."""

    bbox: tuple[float, float, float, float]
    geometry_count: int


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if callable(value):
        value = value()
    if value is None:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _drawing_bbox(drawing: Any) -> tuple[float, float, float, float] | None:
    kind = str(getattr(drawing, "kind", ""))
    if kind in {"image", "inline-image", "clip", "state-push", "state-pop"}:
        return None
    path = getattr(drawing, "path", None)
    path_box = _bbox(getattr(path, "bbox", None))
    return path_box or _bbox(getattr(drawing, "rect", None))


def _geometry_boxes(
    lines: Any,
    drawings: Any,
    minimum_line_length: float,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for line in lines or ():
        if max(abs(float(line.x1) - float(line.x0)), abs(float(line.y1) - float(line.y0))) < (
            minimum_line_length
        ):
            continue
        box = _bbox((line.x0, line.y0, line.x1, line.y1))
        if box is not None and box not in seen:
            seen.add(box)
            boxes.append(box)
    for drawing in drawings or ():
        box = _drawing_bbox(drawing)
        if box is not None and box not in seen:
            seen.add(box)
            boxes.append(box)
    return boxes


def _touches(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    gap: float,
) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    horizontal_gap = max(left_x0 - right_x1, right_x0 - left_x1, 0.0)
    vertical_gap = max(left_y0 - right_y1, right_y0 - left_y1, 0.0)
    horizontal_overlap = min(left_x1, right_x1) - max(left_x0, right_x0)
    vertical_overlap = min(left_y1, right_y1) - max(left_y0, right_y0)
    return (
        horizontal_gap <= gap
        and vertical_overlap >= -gap
        or vertical_gap <= gap
        and horizontal_overlap >= -gap
    )


def detect_chart_regions(
    *,
    lines: Any,
    drawings: Any,
    figures: Any = (),
    page_width: float,
    page_height: float,
) -> tuple[ChartRegion, ...]:
    """Return clusters of repeated page geometry that resemble chart regions.

    The thresholds scale with the page's shorter edge.  A candidate needs at
    least six connected marks, span a meaningful portion of the page, and not
    occupy almost the entire page.  Those constraints reject isolated rules,
    logos, and ordinary paragraph decoration without knowing the document.
    """

    if page_width <= 0 or page_height <= 0:
        return ()
    figure_values = tuple(figures or ())
    minimum_line_length = max(2.0, min(page_width, page_height) * 0.01)
    boxes = _geometry_boxes(lines, drawings, minimum_line_length)
    if len(boxes) < 6 and not figure_values:
        return ()
    gap = max(2.0, min(page_width, page_height) * 0.018)
    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, left_box in enumerate(boxes):
        for right in range(left):
            if _touches(left_box, boxes[right], gap):
                union(left, right)

    components: dict[int, list[tuple[float, float, float, float]]] = {}
    for index, box in enumerate(boxes):
        components.setdefault(find(index), []).append(box)

    page_area = page_width * page_height
    candidates: list[ChartRegion] = []
    for component in components.values():
        if len(component) < 6:
            continue
        x0 = max(0.0, min(box[0] for box in component))
        y0 = max(0.0, min(box[1] for box in component))
        x1 = min(page_width, max(box[2] for box in component))
        y1 = min(page_height, max(box[3] for box in component))
        width = x1 - x0
        height = y1 - y0
        if width < page_width * 0.15 or height < page_height * 0.08:
            continue
        if width * height > page_area * 0.82:
            continue
        candidates.append(ChartRegion((x0, y0, x1, y1), len(component)))

    if not candidates:
        for figure in figure_values:
            figure_box = _bbox(getattr(figure, "bbox", None))
            if figure_box is None:
                continue
            x0, y0, x1, y1 = figure_box
            if (
                x1 - x0 >= page_width * 0.15
                and y1 - y0 >= page_height * 0.08
                and (x1 - x0) * (y1 - y0) <= page_area * 0.82
            ):
                candidates.append(ChartRegion(figure_box, 0))

    # Nested components can arise when a chart has disconnected bars and a
    # surrounding frame. Keep the smallest region that contains each cluster.
    candidates.sort(key=lambda region: (region.bbox[0], region.bbox[1], -region.geometry_count))
    result: list[ChartRegion] = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate.bbox
        if any(
            previous.bbox[0] <= x0
            and previous.bbox[1] <= y0
            and previous.bbox[2] >= x1
            and previous.bbox[3] >= y1
            and previous.geometry_count >= candidate.geometry_count
            for previous in result
        ):
            continue
        result = [
            previous
            for previous in result
            if not (
                x0 <= previous.bbox[0]
                and y0 <= previous.bbox[1]
                and x1 >= previous.bbox[2]
                and y1 >= previous.bbox[3]
                and candidate.geometry_count >= previous.geometry_count
            )
        ]
        result.append(candidate)
    return tuple(result)


__all__ = ("ChartRegion", "detect_chart_regions")
