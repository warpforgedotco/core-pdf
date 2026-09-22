from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy

from core_pdf.api.compat.pdfminer._layout import LTChar, LTFigure, LTTextBox
from core_pdf.impl._impl.model.geometry import flip_rect_vertical

from ._classification import (
    internal_BULLET,
    internal_element_classes,
)
from ._elements import (
    Element,
    ListItem,
)


@dataclass(frozen=True, slots=True)
class internal_TextRegion:
    text: str
    bbox: tuple[float, float, float, float]
    element_class: type[Element] | None = None


def internal_clean_text(text: str) -> str:
    cleaned = text.translate(
        {
            ord("\n"): ord(" "),
            ord("\xa0"): ord(" "),
        }
    )
    return re.sub(r" {2,}", " ", cleaned).strip()


def internal_layout_regions(items: list[LTTextBox]) -> list[internal_TextRegion]:
    return [
        internal_TextRegion(text, item.bbox)
        for item in items
        if (text := internal_clean_text(internal_deduplicated_box_text(item)))
    ]


def internal_duplicate_character(first: LTChar, second: LTChar, threshold: float = 2.0) -> bool:
    if first.get_text() != second.get_text():
        return False
    if abs(first.x0 - second.x0) >= threshold or abs(first.y0 - second.y0) >= threshold:
        return False
    first_width = first.x1 - first.x0
    second_width = second.x1 - second.x0
    average_width = (first_width + second_width) / 2.0
    if average_width <= 0:
        return False
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    return overlap / average_width > 0.5


def internal_deduplicated_box_text(box: LTTextBox) -> str:
    parts: list[str] = []
    for line in box:
        previous: LTChar | None = None
        for item in line:
            if isinstance(item, LTChar):
                if previous is not None and internal_duplicate_character(previous, item):
                    continue
                previous = item
            parts.append(item.get_text())
    return "".join(parts)


def internal_figure_text_snippets(figure: LTFigure) -> list[str]:
    snippets = list(figure.text_snippets)
    for child in figure:
        if not isinstance(child, LTFigure):
            continue
        child_snippets = internal_figure_text_snippets(child)
        if snippets and child_snippets:
            snippets[-1] += child_snippets.pop(0)
        snippets.extend(child_snippets)
    return snippets


def internal_projection_segments(
    boxes: numpy.ndarray[Any, Any], axis: int
) -> list[tuple[int, int]]:
    if not len(boxes):
        return []
    length = max(0, int(numpy.max(boxes[:, axis::2])))
    starts = numpy.minimum(boxes[:, axis].astype(numpy.int64), length)
    ends = numpy.minimum(boxes[:, axis + 2].astype(numpy.int64), length)
    keep = starts < ends
    intervals = sorted(zip(starts[keep].tolist(), ends[keep].tolist(), strict=True))
    segments: list[tuple[int, int]] = []
    for start, end in intervals:
        if segments and start <= segments[-1][1]:
            previous_start, previous_end = segments[-1]
            segments[-1] = (previous_start, max(previous_end, end))
        else:
            segments.append((start, end))
    return segments


def internal_recursive_xy_cut(
    boxes: numpy.ndarray[Any, Any],
    indices: numpy.ndarray[Any, Any],
    result: list[int],
) -> None:
    x_order = boxes[:, 0].argsort()
    x_boxes = boxes[x_order]
    x_indices = indices[x_order]
    x_segments = internal_projection_segments(x_boxes, 0)
    for start, end in numpy.searchsorted(x_boxes[:, 0], x_segments):
        chunk = x_boxes[start:end]
        chunk_indices = x_indices[start:end]
        y_order = chunk[:, 1].argsort()
        y_boxes = chunk[y_order]
        y_indices = chunk_indices[y_order]
        y_segments = internal_projection_segments(y_boxes, 1)
        if len(y_segments) == 1:
            result.extend(int(index) for index in y_indices)
            continue
        for start, end in numpy.searchsorted(y_boxes[:, 1], y_segments):
            internal_recursive_xy_cut(y_boxes[start:end], y_indices[start:end], result)


def internal_region_order(
    regions: list[internal_TextRegion],
    page_height: float,
) -> list[int]:
    basic_order = sorted(
        range(len(regions)),
        key=lambda index: (page_height - regions[index].bbox[3], regions[index].bbox[0]),
    )
    boxes = []
    for index in basic_order:
        left, top, right, bottom = flip_rect_vertical(regions[index].bbox, page_height)
        coordinate_limit = max(1.0, page_height) * 64.0
        if any(
            coordinate < 0 or coordinate > coordinate_limit
            for coordinate in (left, top, right, bottom)
        ):
            return basic_order
        left_int = int(left)
        top_int = int(top)
        right_int = int(right)
        bottom_int = int(bottom)
        boxes.append(
            (
                left_int,
                top_int,
                int(right_int - (right_int - left_int) * 0.1),
                int(bottom_int - (bottom_int - top_int) * 0.1),
            )
        )
    if not boxes:
        return []
    result: list[int] = []
    internal_recursive_xy_cut(
        numpy.asarray(boxes, dtype=numpy.int64),
        numpy.asarray(basic_order, dtype=numpy.int64),
        result,
    )
    return result


def internal_combine_list_regions(
    regions: list[internal_TextRegion],
    page_height: float,
) -> list[internal_TextRegion]:
    combined: list[internal_TextRegion] = []
    anchor_text: str | None = None
    anchor_bbox: tuple[float, float, float, float] | None = None
    active_bbox: tuple[float, float, float, float] | None = None
    anchor_position: int | None = None
    element_classes = internal_element_classes(
        ((region.text, region.bbox) for region in regions), page_height
    )
    for region, element_class in zip(regions, element_classes, strict=True):
        text, bbox = region.text, region.bbox
        if element_class is ListItem:
            anchor_text = internal_BULLET.sub("", text, count=1).strip()
            anchor_bbox = bbox
            active_bbox = bbox
            combined.append(internal_TextRegion(anchor_text, bbox, ListItem))
            anchor_position = len(combined) - 1
            continue
        if anchor_text is not None and anchor_bbox is not None and active_bbox is not None:
            left, bottom, right, top = anchor_bbox
            width = right - left
            height = top - bottom
            current_left, current_bottom, current_right, current_top = bbox
            within_x = (
                current_left > left - 0.2 * width
                and current_right < right + 0.2 * width
                and current_left >= left
            )
            within_y = current_top > bottom - 0.3 * height and current_top < top + 0.3 * height
            if within_x and within_y:
                active_left, active_bottom, active_right, active_top = active_bbox
                merged_bbox = (
                    min(active_left, current_left),
                    min(active_bottom, current_bottom),
                    max(active_right, current_right),
                    max(active_top, current_top),
                )
                merged_region = internal_TextRegion(f"{anchor_text} {text}", merged_bbox, ListItem)
                if anchor_position is not None:
                    combined[anchor_position] = merged_region
                if combined:
                    if anchor_position == len(combined) - 1:
                        anchor_position = None
                    combined.pop()
                combined.append(merged_region)
                active_bbox = merged_bbox
                continue
        combined.append(internal_TextRegion(text, bbox, element_class))
    return combined
