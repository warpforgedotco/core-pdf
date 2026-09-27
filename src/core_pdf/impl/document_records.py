# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf.impl.recovery_text_strings import decode_pdf_text_string
from core_pdf.impl.types import PdfName, PdfString, RecordType, Rectangle, ReplaceFields
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfArray, PdfDict, PdfObject


class RawOutlineItem(ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
    title: str
    level: int
    dest: PdfObject | str | None
    page_index: int | None
    count: int


class RawNamedDestination(ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
    page_index: int | None
    type: str | None
    args: PdfArray
    raw: PdfObject | str


class RawEmbeddedFile(ReplaceFields, metaclass=RecordType, frozen=False, eq=False):
    name: str
    filename: str
    filespec: PdfDict
    stream: PdfStream
    data: bytes


class RawAnnotation:
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


class RawLink:
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


class RawFormField:
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
        value: object,
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

    @property
    def flags(self) -> int:
        value = self.dict.get(PdfName.of("Ff"), 0)
        return int(value) if isinstance(value, int) else 0

    @property
    def is_read_only(self) -> bool:
        return bool(self.flags & 1)

    @property
    def is_required(self) -> bool:
        return bool(self.flags & 2)

    @property
    def no_export(self) -> bool:
        return bool(self.flags & 4)

    @property
    def options(self) -> tuple[str, ...]:
        value = self.dict.get(PdfName.of("Opt"), ())
        if not isinstance(value, list):
            return ()
        result: list[str] = []
        for item in value:
            if isinstance(item, PdfString):
                try:
                    result.append(decode_pdf_text_string(item.data))
                except ValueError:
                    result.append(item.data.decode("utf-8", errors="replace"))
            elif isinstance(item, str):
                result.append(item)
        return tuple(result)


__all__ = (
    "RawAnnotation",
    "RawEmbeddedFile",
    "RawFormField",
    "RawLink",
    "RawNamedDestination",
    "RawOutlineItem",
)
