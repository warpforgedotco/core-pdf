from __future__ import annotations

import re
from typing import Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl.geometry import flip_rect_vertical, interval_overlap
from core_pdf_compat.pdfminer._layout import LTChar, LTFigure, LTTextBox

from ._classification import (
    BULLET,
    classify_elements,
)
from ._elements import (
    Element,
    ListItem,
)

frozen_setattr = object.__setattr__


class TextRegion:
    __slots__ = ("text", "bbox", "element_class")

    text: str
    bbox: tuple[float, float, float, float]
    element_class: type[Element] | None

    __fields__: ClassVar[tuple[str, ...]] = ("text", "bbox", "element_class")
    __match_args__ = ("text", "bbox", "element_class")

    def __init__(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        element_class: type[Element] | None = None,
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "element_class", element_class)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"element_class={self.element_class!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.element_class == other.element_class
        )

    def __hash__(self) -> int:
        return hash((self.text, self.bbox, self.element_class))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        element_class = changes.pop("element_class", self.element_class)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(text, bbox, element_class)


def clean_text(text: str) -> str:
    cleaned = text.translate(
        {
            ord("\n"): ord(" "),
            ord("\xa0"): ord(" "),
        }
    )
    return re.sub(r" {2,}", " ", cleaned).strip()


def layout_regions(items: list[LTTextBox]) -> list[TextRegion]:
    return [
        TextRegion(text, item.bbox)
        for item in items
        if (text := clean_text(deduplicated_box_text(item)))
    ]


def duplicate_character(first: LTChar, second: LTChar, threshold: float = 2.0) -> bool:
    if first.get_text() != second.get_text():
        return False
    if abs(first.x0 - second.x0) >= threshold or abs(first.y0 - second.y0) >= threshold:
        return False
    first_width = first.x1 - first.x0
    second_width = second.x1 - second.x0
    average_width = (first_width + second_width) / 2.0
    if average_width <= 0:
        return False
    overlap = interval_overlap(first.x0, first.x1, second.x0, second.x1)
    return overlap / average_width > 0.5


def deduplicated_box_text(box: LTTextBox) -> str:
    parts: list[str] = []
    for line in box:
        previous: LTChar | None = None
        for item in line:
            if isinstance(item, LTChar):
                if previous is not None and duplicate_character(previous, item):
                    continue
                previous = item
            parts.append(item.get_text())
    return "".join(parts)


def figure_text_snippets(figure: LTFigure) -> list[str]:
    snippets = list(figure.text_snippets)
    for child in figure:
        if not isinstance(child, LTFigure):
            continue
        child_snippets = figure_text_snippets(child)
        if snippets and child_snippets:
            snippets[-1] += child_snippets.pop(0)
        snippets.extend(child_snippets)
    return snippets


def projection_segments(boxes: numpy.ndarray[Any, Any], axis: int) -> list[tuple[int, int]]:
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


def recursive_xy_cut(
    boxes: numpy.ndarray[Any, Any],
    indices: numpy.ndarray[Any, Any],
    result: list[int],
) -> None:
    x_order = boxes[:, 0].argsort()
    x_boxes = boxes[x_order]
    x_indices = indices[x_order]
    x_segments = projection_segments(x_boxes, 0)
    for start, end in numpy.searchsorted(x_boxes[:, 0], x_segments):
        chunk = x_boxes[start:end]
        chunk_indices = x_indices[start:end]
        y_order = chunk[:, 1].argsort()
        y_boxes = chunk[y_order]
        y_indices = chunk_indices[y_order]
        y_segments = projection_segments(y_boxes, 1)
        if len(y_segments) == 1:
            result.extend(int(index) for index in y_indices)
            continue
        for start, end in numpy.searchsorted(y_boxes[:, 1], y_segments):
            recursive_xy_cut(y_boxes[start:end], y_indices[start:end], result)


def region_order(
    regions: list[TextRegion],
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
    recursive_xy_cut(
        numpy.asarray(boxes, dtype=numpy.int64),
        numpy.asarray(basic_order, dtype=numpy.int64),
        result,
    )
    return result


def combine_list_regions(
    regions: list[TextRegion],
    page_height: float,
) -> list[TextRegion]:
    combined: list[TextRegion] = []
    anchor_text: str | None = None
    anchor_bbox: tuple[float, float, float, float] | None = None
    active_bbox: tuple[float, float, float, float] | None = None
    anchor_position: int | None = None
    element_classes = classify_elements(
        ((region.text, region.bbox) for region in regions), page_height
    )
    for region, element_class in zip(regions, element_classes, strict=True):
        text, bbox = region.text, region.bbox
        if element_class is ListItem:
            anchor_text = BULLET.sub("", text, count=1).strip()
            anchor_bbox = bbox
            active_bbox = bbox
            combined.append(TextRegion(anchor_text, bbox, ListItem))
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
                merged_region = TextRegion(f"{anchor_text} {text}", merged_bbox, ListItem)
                if anchor_position is not None:
                    combined[anchor_position] = merged_region
                if combined:
                    if anchor_position == len(combined) - 1:
                        anchor_position = None
                    combined.pop()
                combined.append(merged_region)
                active_bbox = merged_bbox
                continue
        combined.append(TextRegion(text, bbox, element_class))
    return combined
