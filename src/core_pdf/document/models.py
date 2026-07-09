from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from core_pdf.layout.geometry import RectBox
    from core_pdf.syntax.primitives import PdfReference


class TextTraceSpan(TypedDict):
    seqno: int
    color: tuple[float, ...] | None
    bbox: RectBox
    chars: list[tuple[int, int, int, RectBox]]


class OutlineItem:
    __slots__ = ("title", "level", "dest", "page_index", "count")

    def __init__(
        self,
        title: str,
        level: int,
        dest: PdfReference | list[Any] | str | None,
        page_index: int | None,
        count: int,
    ) -> None:
        self.title = title
        self.level = level
        self.dest = dest
        self.page_index = page_index
        self.count = count


class NamedDestination:
    __slots__ = ("page_index", "type", "args", "raw")

    def __init__(
        self,
        page_index: int | None,
        type: str | None,
        args: list[Any],
        raw: Any,
    ) -> None:
        self.page_index = page_index
        self.type = type
        self.args = args
        self.raw = raw


class AnnotationRecord:
    __slots__ = ("subtype", "rect", "contents", "dest", "action", "dict")

    def __init__(
        self,
        subtype: str | None,
        rect: tuple[float, float, float, float] | None,
        contents: str,
        dict_: dict[str, Any],
        dest: Any = None,
        action: dict[str, Any] | None = None,
    ) -> None:
        self.subtype = subtype
        self.rect = rect
        self.contents = contents
        self.dest = dest
        self.action = action
        self.dict = dict_


class FieldRecord:
    __slots__ = ("name", "type", "value", "dict", "kids", "widget")

    def __init__(
        self,
        name: str,
        type: str,
        value: Any,
        dict_: dict[str, Any],
        kids: list[Any] | None = None,
        widget: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.value = value
        self.dict = dict_
        self.kids = kids if kids is not None else []
        self.widget = widget
