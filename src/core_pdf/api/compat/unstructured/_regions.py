"""Unstructured text regions, list continuation, and reading order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy

from core_pdf.api.compat.pdfminer._layout import LTChar, LTFigure, LTTextBox
from core_pdf.impl._impl.model.geometry import flip_rect_vertical

from ._classification import (
    internal_BULLET,
    internal_element_class,
)
from ._elements import (
    Element,
    ListItem,
)


@dataclass(frozen=True, slots=True)
class internal_LayoutRegion:
    text: str
    bbox: tuple[float, float, float, float]
    element_class: type[Element] | None = None


def internal_clean_text(text: str) -> str:
    # Match Unstructured's whitespace cleanup. Tabs and Unicode spacing
    # characters other than NBSP remain meaningful.
    cleaned = text.translate(
        {
            ord("\n"): ord(" "),
            ord("\xa0"): ord(" "),
        }
    )
    return re.sub(r" {2,}", " ", cleaned).strip()


def internal_layout_regions(items: list[LTTextBox]) -> list[internal_LayoutRegion]:
    """Project each canonical pdfminer text box to one Unstructured region."""
    return [
        internal_LayoutRegion(text, item.bbox)
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
    """Return drawing-delimited text while preserving nested figure concatenation."""
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
    length = int(numpy.max(boxes[:, axis::2]))
    projection = numpy.zeros(max(0, length), dtype=numpy.int64)
    for box in boxes:
        projection[int(box[axis]) : int(box[axis + 2])] += 1
    occupied = numpy.where(projection > 0)[0]
    if not len(occupied):
        return []
    gaps = numpy.where(occupied[1:] - occupied[:-1] > 1)[0]
    starts = [int(occupied[0]), *(int(occupied[index + 1]) for index in gaps)]
    ends = [*(int(occupied[index]) + 1 for index in gaps), int(occupied[-1]) + 1]
    return list(zip(starts, ends))


def internal_recursive_xy_cut(
    boxes: numpy.ndarray[Any, Any],
    indices: numpy.ndarray[Any, Any],
    result: list[int],
) -> None:
    x_order = boxes[:, 0].argsort()
    x_boxes = boxes[x_order]
    x_indices = indices[x_order]
    for x0, x1 in internal_projection_segments(x_boxes, 0):
        x_mask = (x0 <= x_boxes[:, 0]) & (x_boxes[:, 0] < x1)
        chunk = x_boxes[x_mask]
        chunk_indices = x_indices[x_mask]
        y_order = chunk[:, 1].argsort()
        y_boxes = chunk[y_order]
        y_indices = chunk_indices[y_order]
        y_segments = internal_projection_segments(y_boxes, 1)
        if len(y_segments) == 1:
            result.extend(int(index) for index in y_indices)
            continue
        for y0, y1 in y_segments:
            y_mask = (y0 <= y_boxes[:, 1]) & (y_boxes[:, 1] < y1)
            internal_recursive_xy_cut(y_boxes[y_mask], y_indices[y_mask], result)


def internal_region_order(
    regions: list[internal_LayoutRegion],
    page_height: float,
) -> list[int]:
    # The fast parser applies a stable basic top/left sort before XY-cut to
    # make tie behavior deterministic across Python versions.
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
            # Unstructured validates the floating-point coordinates before
            # casting them for XY-cut. On invalid geometry it preserves the
            # PDFMiner layout iteration order verbatim.
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
    regions: list[internal_LayoutRegion],
    page_height: float,
) -> list[internal_LayoutRegion]:
    """Apply Unstructured's pre-sort continuation merge for list elements."""
    combined: list[internal_LayoutRegion] = []
    anchor_text: str | None = None
    anchor_bbox: tuple[float, float, float, float] | None = None
    active_bbox: tuple[float, float, float, float] | None = None
    anchor_position: int | None = None
    for region in regions:
        text, bbox = region.text, region.bbox
        element_class = internal_element_class(text, bbox, page_height)
        if element_class is ListItem:
            anchor_text = internal_BULLET.sub("", text, count=1).strip()
            anchor_bbox = bbox
            active_bbox = bbox
            combined.append(internal_LayoutRegion(anchor_text, bbox, ListItem))
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
                merged_region = internal_LayoutRegion(
                    f"{anchor_text} {text}", merged_bbox, ListItem
                )
                if anchor_position is not None:
                    # ``_combine_list_elements`` mutates its retained
                    # ``tmp_element`` before deep-copying it. If another
                    # element was emitted since the list anchor, that earlier
                    # list entry therefore changes as well.
                    combined[anchor_position] = merged_region
                # Mirror the reference continuation loop exactly: it removes
                # the most recently emitted element, which need not be the
                # active list item when intervening columns were encountered,
                # then appends a merged copy of that list item.
                if combined:
                    if anchor_position == len(combined) - 1:
                        anchor_position = None
                    combined.pop()
                combined.append(merged_region)
                active_bbox = merged_bbox
                continue
        combined.append(internal_LayoutRegion(text, bbox, element_class))
    return combined
