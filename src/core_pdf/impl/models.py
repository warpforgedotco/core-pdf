# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterable
from math import hypot
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, cast

from core_pdf.impl.engine.layout.geometry import BBox
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfArray, PdfDict, PdfObject, Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.engine.layout.geometry import RectBox


class TextSpan(TypedDict):
    seqno: int
    color: tuple[float, ...] | None
    bbox: RectBox
    chars: list[tuple[int, int, int, RectBox]]


class LinkTextWordRecord(TypedDict):
    bbox: Rectangle
    text: str
    start_index: int


class LinkTextWordObject(Protocol):
    bbox: Rectangle
    text: str
    start_index: int


LinkTextWord: TypeAlias = LinkTextWordRecord | LinkTextWordObject


class OutlineItem:
    """Resolved outline tree item from the document catalog."""

    __slots__ = ("title", "level", "dest", "page_index", "count")

    def __init__(
        self,
        title: str,
        level: int,
        dest: PdfObject | str | None,
        page_index: int | None,
        count: int,
    ) -> None:
        self.title = title
        self.level = level
        self.dest = dest
        self.page_index = page_index
        self.count = count


class NamedDestination:
    """Named destination record from a name tree or destination dictionary."""

    __slots__ = ("page_index", "type", "args", "raw")

    def __init__(
        self,
        page_index: int | None,
        type: str | None,
        args: PdfArray,
        raw: PdfObject | str,
    ) -> None:
        self.page_index = page_index
        self.type = type
        self.args = args
        self.raw = raw


class EmbeddedFileRecord:
    """Embedded file specification and decoded file stream."""

    __slots__ = ("name", "filename", "filespec", "stream", "data")

    def __init__(
        self,
        name: str,
        filename: str,
        filespec: PdfDict,
        stream: PdfStream,
        data: bytes,
    ) -> None:
        self.name = name
        self.filename = filename
        self.filespec = filespec
        self.stream = stream
        self.data = data


class AnnotationRecord:
    """Page annotation record resolved from an annotation dictionary."""

    __slots__ = ("subtype", "rect", "contents", "dest", "action", "dict")

    def __init__(
        self,
        subtype: str | None,
        rect: Rectangle | None,
        contents: str,
        dict_: PdfDict,
        dest: PdfObject | None = None,
        action: PdfDict | None = None,
    ) -> None:
        self.subtype = subtype
        self.rect = rect
        self.contents = contents
        self.dest = dest
        self.action = action
        self.dict = dict_


class LinkRecord:
    """Link annotation projected into extracted page coordinates."""

    __slots__ = ("bbox", "url", "link_type", "page_number", "dict")

    def __init__(
        self,
        bbox: Rectangle,
        page_number: int,
        url: str | None = None,
        link_type: str | None = None,
        dict_: PdfDict | None = None,
    ) -> None:
        self.bbox = bbox
        self.url = url
        self.link_type = link_type
        self.page_number = page_number
        self.dict = dict_

    def page_bbox(self, page_height: float) -> Rectangle:
        return BBox.from_page_rect(BBox.from_rect(self.bbox), page_height)

    def overlaps_page_bbox(
        self,
        bbox: Rectangle,
        page_height: float,
        threshold: float = 0.5,
    ) -> bool:
        link_bbox = self.page_bbox(page_height)
        link_box = BBox.from_rect(link_bbox)
        link_area = link_box.area()
        return bool(
            link_area and BBox.from_rect(bbox).intersection_area(link_box) / link_area > threshold
        )

    def text_span(
        self, words: Iterable[LinkTextWord], page_height: float | None = None
    ) -> tuple[str, int]:
        word_list = list(words)
        if not word_list:
            return "", -1

        bbox = self.page_bbox(page_height) if page_height is not None else self.bbox
        bboxes = [
            cast(LinkTextWordRecord, word)["bbox"]
            if isinstance(word, dict)
            else cast(LinkTextWordObject, word).bbox
            for word in word_list
        ]
        start_index = min(
            range(len(bboxes)),
            key=lambda i: hypot(bbox[0] - bboxes[i][0], bbox[1] - bboxes[i][1]),
        )
        end_index = min(
            range(len(bboxes)),
            key=lambda i: hypot(bbox[2] - bboxes[i][2], bbox[3] - bboxes[i][3]),
        )
        if end_index >= start_index:
            selected_words = word_list[start_index : end_index + 1]
        else:
            selected_words = [word_list[start_index]]
        text = " ".join(
            cast(LinkTextWordRecord, word)["text"]
            if isinstance(word, dict)
            else cast(LinkTextWordObject, word).text
            for word in selected_words
        )
        first_word = word_list[start_index]
        return (
            text.strip(),
            cast(LinkTextWordRecord, first_word)["start_index"]
            if isinstance(first_word, dict)
            else cast(LinkTextWordObject, first_word).start_index,
        )

    def text_metadata(
        self, words: Iterable[LinkTextWord], page_height: float | None = None
    ) -> dict[str, object]:
        text, start_index = self.text_span(words, page_height)
        bbox = self.page_bbox(page_height) if page_height is not None else self.bbox
        return {
            "bbox": bbox,
            "text": text,
            "uri": self.url,
            "url": self.url,
            "start_index": start_index,
        }


class FieldRecord:
    """Interactive form field with resolved value and widget metadata."""

    __slots__ = (
        "name",
        "type",
        "value",
        "value_text",
        "rect",
        "dict",
        "kids",
        "widget",
    )

    def __init__(
        self,
        name: str,
        type: str,
        value: PdfObject,
        value_text: str,
        rect: Rectangle | None,
        dict_: PdfDict,
        kids: PdfArray | None = None,
        widget: PdfDict | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.value = value
        self.value_text = value_text
        self.rect = rect
        self.dict = dict_
        self.kids = kids if kids is not None else []
        self.widget = widget

    def value_words(self) -> list[tuple[str, int]]:
        words = []
        start = None
        for index, char in enumerate(self.value_text):
            if char.isspace():
                if start is not None:
                    words.append((self.value_text[start:index], start))
                    start = None
            elif start is None:
                start = index
        if start is not None:
            words.append((self.value_text[start:], start))
        return words
