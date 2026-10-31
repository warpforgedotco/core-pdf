from __future__ import annotations

import contextlib
import math
import re
from typing import Any, NamedTuple

from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.geometry import flip_rect_vertical, interval_overlap
from core_pdf_compat.pdfminer._layout import LTChar, LTTextBox

# unstructured's default PDF_ANNOTATION_THRESHOLD.
ANNOTATION_THRESHOLD = 0.9

type Box = tuple[float, float, float, float]
type ElementLink = dict[str, Any]


class PageLink(NamedTuple):
    bbox: Box
    url: str | None


class Word(NamedTuple):
    text: str
    bbox: Box
    start_index: int


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


def page_links(source_page: Any, page_height: float) -> list[PageLink]:
    # unstructured reads only annotations with an action, so a bare /Dest has no link.
    try:
        links = source_page.get_links()
    except PdfError, ValueError:
        return []
    return [
        PageLink(flip_rect_vertical(link.bbox, page_height), link.url)
        for link in links
        if link.link_type is not None
    ]


def box_links(
    box: LTTextBox, raw_text: str, links: list[PageLink], page_height: float
) -> list[ElementLink]:
    box_bbox = flip_rect_vertical(box.bbox, page_height)
    within = [link for link in links if link_within_box(link.bbox, box_bbox)]
    if not within:
        return []
    words = box_words(box, page_height)
    shifts = whitespace_shifts(raw_text)
    result: list[ElementLink] = []
    for link in within:
        text, start_index = link_span(words, link.bbox)
        # A span past the cleaned text has no position in it, so it is left out.
        with contextlib.suppress(IndexError):
            result.append(
                {"text": text, "url": link.url, "start_index": start_index - shifts[start_index]}
            )
    return result


def link_within_box(link_bbox: Box, box_bbox: Box) -> bool:
    link_area = (link_bbox[2] - link_bbox[0]) * (link_bbox[3] - link_bbox[1])
    if not link_area:
        return False
    left = max(link_bbox[0], box_bbox[0])
    top = max(link_bbox[1], box_bbox[1])
    right = min(link_bbox[2], box_bbox[2])
    bottom = min(link_bbox[3], box_bbox[3])
    overlap = (right - left) * (bottom - top) if left < right and top < bottom else 0.0
    return overlap / link_area > ANNOTATION_THRESHOLD


def box_words(box: LTTextBox, page_height: float) -> list[Word]:
    # start_index counts every item of the line, deduplicated or not, so that a link's
    # position in the box's raw text comes out the same as unstructured's.
    words: list[Word] = []
    text_length = 0
    for line in box:
        items = list(line)
        word = ""
        word_bbox: Box | None = None
        start_index = 0
        previous: LTChar | None = None
        alphanumeric = False
        for index, item in enumerate(items):
            if not isinstance(item, LTChar):
                # A line that opens with an annotation has no word for it to end.
                if word_bbox is not None:
                    words.append(Word(word, word_bbox, start_index))
                word = ""
                continue
            if previous is not None and duplicate_character(previous, item):
                continue
            previous = item
            character = item.get_text()
            if word and word_bbox is not None and not character.strip():
                words.append(Word(word, word_bbox, start_index))
                word = ""
                continue
            if not word:
                alphanumeric = character.isalnum()
            if word and word_bbox is not None and character.isalnum() != alphanumeric:
                alphanumeric = character.isalnum()
                words.append(Word(word, word_bbox, start_index))
                word = ""
            if word and word_bbox is not None:
                word_bbox = (word_bbox[0], word_bbox[1], item.x1, page_height - item.y0)
            else:
                start_index = text_length + index
                word_bbox = (item.x0, page_height - item.y1, item.x1, page_height - item.y0)
            word += character
        text_length += len(items)
    return words


def link_span(words: list[Word], link_bbox: Box) -> tuple[str, int]:
    if not words:
        return "", -1
    start = min(
        range(len(words)),
        key=lambda index: math.hypot(
            link_bbox[0] - words[index].bbox[0], link_bbox[1] - words[index].bbox[1]
        ),
    )
    end = min(
        range(len(words)),
        key=lambda index: math.hypot(
            link_bbox[2] - words[index].bbox[2], link_bbox[3] - words[index].bbox[3]
        ),
    )
    spanned = words[start : end + 1] if end >= start else words[start : start + 1]
    return " ".join(word.text for word in spanned).strip(), words[start].start_index


def whitespace_shifts(text: str) -> list[int]:
    # How far each character of the cleaned text moved left of its place in `text`.
    cleaned = re.sub(r" {2,}", " ", text.translate({0xA0: " ", 0x0A: " "})).strip()
    shifts = [0] * len(text)
    distance = 0
    source_index = 0
    for cleaned_index, cleaned_character in enumerate(cleaned):
        while text[source_index] != cleaned_character and not (
            text[source_index] in "\xa0\n" and cleaned_character == " "
        ):
            distance += 1
            source_index += 1
        shifts[cleaned_index] = distance
        source_index += 1
    shifts[len(cleaned) :] = [distance] * (len(text) - len(cleaned))
    return shifts
