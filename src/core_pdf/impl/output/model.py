# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterator, Mapping
from collections.abc import Mapping as MappingABC
from copy import replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, NoReturn, Self, TypeAlias

from core_pdf.impl.model.geometry import bbox_union
from core_pdf.impl.model.page_selection import PageSelection
from core_pdf.impl.model.text import internal_reconcile_text_words
from core_pdf.impl.types import Rectangle, TextWord

internal_frozen_setattr = object.__setattr__


SCHEMA_VERSION = "5.0"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def internal_freeze(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: internal_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(internal_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(internal_freeze(item) for item in value)
    return value


UNKNOWN = "unknown"


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    QUOTE = "quote"
    CODE = "code"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


class TableCell:
    __slots__ = ("row", "column", "text", "row_span", "column_span", "bbox")

    row: int
    column: int
    text: str
    row_span: int
    column_span: int
    bbox: Rectangle | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "row",
        "column",
        "text",
        "row_span",
        "column_span",
        "bbox",
    )
    __match_args__ = ("row", "column", "text", "row_span", "column_span", "bbox")

    def __init__(
        self,
        row: int,
        column: int,
        text: str,
        row_span: int = 1,
        column_span: int = 1,
        bbox: Rectangle | None = None,
    ) -> None:
        internal_frozen_setattr(self, "row", row)
        internal_frozen_setattr(self, "column", column)
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "row_span", row_span)
        internal_frozen_setattr(self, "column_span", column_span)
        internal_frozen_setattr(self, "bbox", bbox)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"row={self.row!r}, "
            f"column={self.column!r}, "
            f"text={self.text!r}, "
            f"row_span={self.row_span!r}, "
            f"column_span={self.column_span!r}, "
            f"bbox={self.bbox!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.row == other.row
            and self.column == other.column
            and self.text == other.text
            and self.row_span == other.row_span
            and self.column_span == other.column_span
            and self.bbox == other.bbox
        )

    def __hash__(self) -> int:
        return hash((self.row, self.column, self.text, self.row_span, self.column_span, self.bbox))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        row = changes.pop("row", self.row)
        column = changes.pop("column", self.column)
        text = changes.pop("text", self.text)
        row_span = changes.pop("row_span", self.row_span)
        column_span = changes.pop("column_span", self.column_span)
        bbox = changes.pop("bbox", self.bbox)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(row, column, text, row_span, column_span, bbox)


class TableRowBand:
    __slots__ = ("index", "bbox", "kind", "confidence")

    index: int
    bbox: Rectangle | None
    kind: str
    confidence: float | None

    __fields__: ClassVar[tuple[str, ...]] = ("index", "bbox", "kind", "confidence")
    __match_args__ = ("index", "bbox", "kind", "confidence")

    def __init__(
        self,
        index: int,
        bbox: Rectangle | None = None,
        kind: str = "body",
        confidence: float | None = None,
    ) -> None:
        internal_frozen_setattr(self, "index", index)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "confidence", confidence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"index={self.index!r}, "
            f"bbox={self.bbox!r}, "
            f"kind={self.kind!r}, "
            f"confidence={self.confidence!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.index == other.index
            and self.bbox == other.bbox
            and self.kind == other.kind
            and self.confidence == other.confidence
        )

    def __hash__(self) -> int:
        return hash((self.index, self.bbox, self.kind, self.confidence))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        index = changes.pop("index", self.index)
        bbox = changes.pop("bbox", self.bbox)
        kind = changes.pop("kind", self.kind)
        confidence = changes.pop("confidence", self.confidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(index, bbox, kind, confidence)


class TableColumnBand:
    __slots__ = ("index", "bbox", "confidence")

    index: int
    bbox: Rectangle | None
    confidence: float | None

    __fields__: ClassVar[tuple[str, ...]] = ("index", "bbox", "confidence")
    __match_args__ = ("index", "bbox", "confidence")

    def __init__(
        self,
        index: int,
        bbox: Rectangle | None = None,
        confidence: float | None = None,
    ) -> None:
        internal_frozen_setattr(self, "index", index)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "confidence", confidence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"index={self.index!r}, "
            f"bbox={self.bbox!r}, "
            f"confidence={self.confidence!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.index == other.index
            and self.bbox == other.bbox
            and self.confidence == other.confidence
        )

    def __hash__(self) -> int:
        return hash((self.index, self.bbox, self.confidence))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        index = changes.pop("index", self.index)
        bbox = changes.pop("bbox", self.bbox)
        confidence = changes.pop("confidence", self.confidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(index, bbox, confidence)


class TableAssociatedText:
    __slots__ = ("text", "bbox", "kind", "confidence")

    text: str
    bbox: Rectangle | None
    kind: str
    confidence: float | None

    __fields__: ClassVar[tuple[str, ...]] = ("text", "bbox", "kind", "confidence")
    __match_args__ = ("text", "bbox", "kind", "confidence")

    def __init__(
        self,
        text: str,
        bbox: Rectangle | None = None,
        kind: str = "caption",
        confidence: float | None = None,
    ) -> None:
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "confidence", confidence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"kind={self.kind!r}, "
            f"confidence={self.confidence!r}"
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
            and self.kind == other.kind
            and self.confidence == other.confidence
        )

    def __hash__(self) -> int:
        return hash((self.text, self.bbox, self.kind, self.confidence))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        kind = changes.pop("kind", self.kind)
        confidence = changes.pop("confidence", self.confidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(text, bbox, kind, confidence)


class Table:
    __slots__ = (
        "order",
        "rows",
        "bbox",
        "confidence",
        "title",
        "caption",
        "row_bands",
        "column_bands",
        "metadata",
    )

    order: int
    rows: tuple[tuple[TableCell, ...], ...]
    bbox: Rectangle | None
    confidence: float | None
    title: TableAssociatedText | None
    caption: TableAssociatedText | None
    row_bands: tuple[TableRowBand, ...]
    column_bands: tuple[TableColumnBand, ...]
    metadata: Mapping[str, Any]

    __fields__: ClassVar[tuple[str, ...]] = (
        "order",
        "rows",
        "bbox",
        "confidence",
        "title",
        "caption",
        "row_bands",
        "column_bands",
        "metadata",
    )
    __match_args__ = (
        "order",
        "rows",
        "bbox",
        "confidence",
        "title",
        "caption",
        "row_bands",
        "column_bands",
        "metadata",
    )

    def __init__(
        self,
        order: int,
        rows: tuple[tuple[TableCell, ...], ...] = (),
        bbox: Rectangle | None = None,
        confidence: float | None = None,
        title: TableAssociatedText | None = None,
        caption: TableAssociatedText | None = None,
        row_bands: tuple[TableRowBand, ...] = (),
        column_bands: tuple[TableColumnBand, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        internal_frozen_setattr(self, "order", order)
        internal_frozen_setattr(self, "rows", rows)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "title", title)
        internal_frozen_setattr(self, "caption", caption)
        internal_frozen_setattr(self, "row_bands", row_bands)
        internal_frozen_setattr(self, "column_bands", column_bands)
        internal_frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"order={self.order!r}, "
            f"rows={self.rows!r}, "
            f"bbox={self.bbox!r}, "
            f"confidence={self.confidence!r}, "
            f"title={self.title!r}, "
            f"caption={self.caption!r}, "
            f"row_bands={self.row_bands!r}, "
            f"column_bands={self.column_bands!r}, "
            f"metadata={self.metadata!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.order == other.order
            and self.rows == other.rows
            and self.bbox == other.bbox
            and self.confidence == other.confidence
            and self.title == other.title
            and self.caption == other.caption
            and self.row_bands == other.row_bands
            and self.column_bands == other.column_bands
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.order,
                self.rows,
                self.bbox,
                self.confidence,
                self.title,
                self.caption,
                self.row_bands,
                self.column_bands,
                self.metadata,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        order = changes.pop("order", self.order)
        rows = changes.pop("rows", self.rows)
        bbox = changes.pop("bbox", self.bbox)
        confidence = changes.pop("confidence", self.confidence)
        title = changes.pop("title", self.title)
        caption = changes.pop("caption", self.caption)
        row_bands = changes.pop("row_bands", self.row_bands)
        column_bands = changes.pop("column_bands", self.column_bands)
        metadata = changes.pop("metadata", self.metadata)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            order,
            rows,
            bbox,
            confidence,
            title,
            caption,
            row_bands,
            column_bands,
            metadata,
        )

    def _post_init(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))

    @property
    def layout_bbox(self) -> Rectangle | None:
        boxes = [box for box in (self.bbox, self.title_bbox, self.caption_bbox) if box is not None]
        return bbox_union(boxes)

    @property
    def title_bbox(self) -> Rectangle | None:
        return self.title.bbox if self.title is not None else None

    @property
    def caption_bbox(self) -> Rectangle | None:
        return self.caption.bbox if self.caption is not None else None

    @property
    def content_bbox(self) -> Rectangle | None:
        boxes = [cell.bbox for row in self.rows for cell in row if cell.bbox is not None]
        return bbox_union(boxes) if boxes else self.bbox


class Figure:
    __slots__ = ("order", "bbox", "kind", "metadata")

    order: int
    bbox: Rectangle | None
    kind: str
    metadata: Mapping[str, Any]

    __fields__: ClassVar[tuple[str, ...]] = ("order", "bbox", "kind", "metadata")
    __match_args__ = ("order", "bbox", "kind", "metadata")

    def __init__(
        self,
        order: int,
        bbox: Rectangle | None = None,
        kind: str = "figure",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        internal_frozen_setattr(self, "order", order)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"order={self.order!r}, "
            f"bbox={self.bbox!r}, "
            f"kind={self.kind!r}, "
            f"metadata={self.metadata!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.order == other.order
            and self.bbox == other.bbox
            and self.kind == other.kind
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        return hash((self.order, self.bbox, self.kind, self.metadata))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        order = changes.pop("order", self.order)
        bbox = changes.pop("bbox", self.bbox)
        kind = changes.pop("kind", self.kind)
        metadata = changes.pop("metadata", self.metadata)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(order, bbox, kind, metadata)

    def _post_init(self) -> None:
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))


class Link:
    __slots__ = ("bbox", "url", "link_type", "text")

    bbox: Rectangle | None
    url: str | None
    link_type: str | None
    text: str

    __fields__: ClassVar[tuple[str, ...]] = ("bbox", "url", "link_type", "text")
    __match_args__ = ("bbox", "url", "link_type", "text")

    def __init__(
        self,
        bbox: Rectangle | None = None,
        url: str | None = None,
        link_type: str | None = None,
        text: str = "",
    ) -> None:
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "url", url)
        internal_frozen_setattr(self, "link_type", link_type)
        internal_frozen_setattr(self, "text", text)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"bbox={self.bbox!r}, "
            f"url={self.url!r}, "
            f"link_type={self.link_type!r}, "
            f"text={self.text!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.url == other.url
            and self.link_type == other.link_type
            and self.text == other.text
        )

    def __hash__(self) -> int:
        return hash((self.bbox, self.url, self.link_type, self.text))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        bbox = changes.pop("bbox", self.bbox)
        url = changes.pop("url", self.url)
        link_type = changes.pop("link_type", self.link_type)
        text = changes.pop("text", self.text)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(bbox, url, link_type, text)


class Annotation:
    __slots__ = ("subtype", "bbox", "contents", "destination")

    subtype: str | None
    bbox: Rectangle | None
    contents: str
    destination: Any

    __fields__: ClassVar[tuple[str, ...]] = ("subtype", "bbox", "contents", "destination")
    __match_args__ = ("subtype", "bbox", "contents", "destination")

    def __init__(
        self,
        subtype: str | None = None,
        bbox: Rectangle | None = None,
        contents: str = "",
        destination: Any = None,
    ) -> None:
        internal_frozen_setattr(self, "subtype", subtype)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "contents", contents)
        internal_frozen_setattr(self, "destination", destination)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"subtype={self.subtype!r}, "
            f"bbox={self.bbox!r}, "
            f"contents={self.contents!r}, "
            f"destination={self.destination!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.subtype == other.subtype
            and self.bbox == other.bbox
            and self.contents == other.contents
            and self.destination == other.destination
        )

    def __hash__(self) -> int:
        return hash((self.subtype, self.bbox, self.contents, self.destination))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        subtype = changes.pop("subtype", self.subtype)
        bbox = changes.pop("bbox", self.bbox)
        contents = changes.pop("contents", self.contents)
        destination = changes.pop("destination", self.destination)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(subtype, bbox, contents, destination)

    def _post_init(self) -> None:
        object.__setattr__(self, "destination", internal_freeze(self.destination))


class FormField:
    __slots__ = (
        "name",
        "field_type",
        "value_text",
        "bbox",
        "field_index",
        "required",
        "read_only",
        "no_export",
        "options",
    )

    name: str
    field_type: str
    value_text: str
    bbox: Rectangle | None
    field_index: int | None
    required: bool
    read_only: bool
    no_export: bool
    options: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "name",
        "field_type",
        "value_text",
        "bbox",
        "field_index",
        "required",
        "read_only",
        "no_export",
        "options",
    )
    __match_args__ = (
        "name",
        "field_type",
        "value_text",
        "bbox",
        "field_index",
        "required",
        "read_only",
        "no_export",
        "options",
    )

    def __init__(
        self,
        name: str,
        field_type: str,
        value_text: str = "",
        bbox: Rectangle | None = None,
        field_index: int | None = None,
        required: bool = False,
        read_only: bool = False,
        no_export: bool = False,
        options: tuple[str, ...] = (),
    ) -> None:
        internal_frozen_setattr(self, "name", name)
        internal_frozen_setattr(self, "field_type", field_type)
        internal_frozen_setattr(self, "value_text", value_text)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "field_index", field_index)
        internal_frozen_setattr(self, "required", required)
        internal_frozen_setattr(self, "read_only", read_only)
        internal_frozen_setattr(self, "no_export", no_export)
        internal_frozen_setattr(self, "options", options)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self.name!r}, "
            f"field_type={self.field_type!r}, "
            f"value_text={self.value_text!r}, "
            f"bbox={self.bbox!r}, "
            f"field_index={self.field_index!r}, "
            f"required={self.required!r}, "
            f"read_only={self.read_only!r}, "
            f"no_export={self.no_export!r}, "
            f"options={self.options!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.field_type == other.field_type
            and self.value_text == other.value_text
            and self.bbox == other.bbox
            and self.field_index == other.field_index
            and self.required == other.required
            and self.read_only == other.read_only
            and self.no_export == other.no_export
            and self.options == other.options
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.field_type,
                self.value_text,
                self.bbox,
                self.field_index,
                self.required,
                self.read_only,
                self.no_export,
                self.options,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        field_type = changes.pop("field_type", self.field_type)
        value_text = changes.pop("value_text", self.value_text)
        bbox = changes.pop("bbox", self.bbox)
        field_index = changes.pop("field_index", self.field_index)
        required = changes.pop("required", self.required)
        read_only = changes.pop("read_only", self.read_only)
        no_export = changes.pop("no_export", self.no_export)
        options = changes.pop("options", self.options)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            name,
            field_type,
            value_text,
            bbox,
            field_index,
            required,
            read_only,
            no_export,
            options,
        )


class TextSpan:
    __slots__ = (
        "text",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
    )

    text: str
    bold: bool
    italic: bool
    underline: bool
    strikeout: bool
    mark: bool
    superscript: bool
    subscript: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
    )
    __match_args__ = (
        "text",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
    )

    def __init__(
        self,
        text: str,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        strikeout: bool = False,
        mark: bool = False,
        superscript: bool = False,
        subscript: bool = False,
    ) -> None:
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "bold", bold)
        internal_frozen_setattr(self, "italic", italic)
        internal_frozen_setattr(self, "underline", underline)
        internal_frozen_setattr(self, "strikeout", strikeout)
        internal_frozen_setattr(self, "mark", mark)
        internal_frozen_setattr(self, "superscript", superscript)
        internal_frozen_setattr(self, "subscript", subscript)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bold={self.bold!r}, "
            f"italic={self.italic!r}, "
            f"underline={self.underline!r}, "
            f"strikeout={self.strikeout!r}, "
            f"mark={self.mark!r}, "
            f"superscript={self.superscript!r}, "
            f"subscript={self.subscript!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bold == other.bold
            and self.italic == other.italic
            and self.underline == other.underline
            and self.strikeout == other.strikeout
            and self.mark == other.mark
            and self.superscript == other.superscript
            and self.subscript == other.subscript
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.bold,
                self.italic,
                self.underline,
                self.strikeout,
                self.mark,
                self.superscript,
                self.subscript,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bold = changes.pop("bold", self.bold)
        italic = changes.pop("italic", self.italic)
        underline = changes.pop("underline", self.underline)
        strikeout = changes.pop("strikeout", self.strikeout)
        mark = changes.pop("mark", self.mark)
        superscript = changes.pop("superscript", self.superscript)
        subscript = changes.pop("subscript", self.subscript)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            text,
            bold,
            italic,
            underline,
            strikeout,
            mark,
            superscript,
            subscript,
        )


class TextLine:
    __slots__ = (
        "text",
        "break_before",
        "bbox",
        "advance_bbox",
        "ink_bbox",
        "kind",
        "source",
        "confidence",
        "baseline",
        "contributing_sources",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
        "spans",
        "words",
    )

    text: str
    break_before: int
    bbox: Rectangle | None
    advance_bbox: Rectangle | None
    ink_bbox: Rectangle | None
    kind: str
    source: str
    confidence: float | None
    baseline: Rectangle | None
    contributing_sources: tuple[str, ...]
    bold: bool
    italic: bool
    underline: bool
    strikeout: bool
    mark: bool
    superscript: bool
    subscript: bool
    spans: tuple[TextSpan, ...]
    words: tuple[TextWord, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "break_before",
        "bbox",
        "advance_bbox",
        "ink_bbox",
        "kind",
        "source",
        "confidence",
        "baseline",
        "contributing_sources",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
        "spans",
        "words",
    )
    __match_args__ = (
        "text",
        "break_before",
        "bbox",
        "advance_bbox",
        "ink_bbox",
        "kind",
        "source",
        "confidence",
        "baseline",
        "contributing_sources",
        "bold",
        "italic",
        "underline",
        "strikeout",
        "mark",
        "superscript",
        "subscript",
        "spans",
        "words",
    )

    def __init__(
        self,
        text: str,
        break_before: int = 1,
        bbox: Rectangle | None = None,
        advance_bbox: Rectangle | None = None,
        ink_bbox: Rectangle | None = None,
        kind: str = "text-line",
        source: str = UNKNOWN,
        confidence: float | None = None,
        baseline: Rectangle | None = None,
        contributing_sources: tuple[str, ...] = (),
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        strikeout: bool = False,
        mark: bool = False,
        superscript: bool = False,
        subscript: bool = False,
        spans: tuple[TextSpan, ...] = (),
        words: tuple[TextWord, ...] = (),
    ) -> None:
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "break_before", break_before)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "advance_bbox", advance_bbox)
        internal_frozen_setattr(self, "ink_bbox", ink_bbox)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "source", source)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "baseline", baseline)
        internal_frozen_setattr(self, "contributing_sources", contributing_sources)
        internal_frozen_setattr(self, "bold", bold)
        internal_frozen_setattr(self, "italic", italic)
        internal_frozen_setattr(self, "underline", underline)
        internal_frozen_setattr(self, "strikeout", strikeout)
        internal_frozen_setattr(self, "mark", mark)
        internal_frozen_setattr(self, "superscript", superscript)
        internal_frozen_setattr(self, "subscript", subscript)
        internal_frozen_setattr(self, "spans", spans)
        internal_frozen_setattr(self, "words", words)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"break_before={self.break_before!r}, "
            f"bbox={self.bbox!r}, "
            f"advance_bbox={self.advance_bbox!r}, "
            f"ink_bbox={self.ink_bbox!r}, "
            f"kind={self.kind!r}, "
            f"source={self.source!r}, "
            f"confidence={self.confidence!r}, "
            f"baseline={self.baseline!r}, "
            f"contributing_sources={self.contributing_sources!r}, "
            f"bold={self.bold!r}, "
            f"italic={self.italic!r}, "
            f"underline={self.underline!r}, "
            f"strikeout={self.strikeout!r}, "
            f"mark={self.mark!r}, "
            f"superscript={self.superscript!r}, "
            f"subscript={self.subscript!r}, "
            f"spans={self.spans!r}, "
            f"words={self.words!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.break_before == other.break_before
            and self.bbox == other.bbox
            and self.advance_bbox == other.advance_bbox
            and self.ink_bbox == other.ink_bbox
            and self.kind == other.kind
            and self.source == other.source
            and self.confidence == other.confidence
            and self.baseline == other.baseline
            and self.contributing_sources == other.contributing_sources
            and self.bold == other.bold
            and self.italic == other.italic
            and self.underline == other.underline
            and self.strikeout == other.strikeout
            and self.mark == other.mark
            and self.superscript == other.superscript
            and self.subscript == other.subscript
            and self.spans == other.spans
            and self.words == other.words
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.break_before,
                self.bbox,
                self.advance_bbox,
                self.ink_bbox,
                self.kind,
                self.source,
                self.confidence,
                self.baseline,
                self.contributing_sources,
                self.bold,
                self.italic,
                self.underline,
                self.strikeout,
                self.mark,
                self.superscript,
                self.subscript,
                self.spans,
                self.words,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        break_before = changes.pop("break_before", self.break_before)
        bbox = changes.pop("bbox", self.bbox)
        advance_bbox = changes.pop("advance_bbox", self.advance_bbox)
        ink_bbox = changes.pop("ink_bbox", self.ink_bbox)
        kind = changes.pop("kind", self.kind)
        source = changes.pop("source", self.source)
        confidence = changes.pop("confidence", self.confidence)
        baseline = changes.pop("baseline", self.baseline)
        contributing_sources = changes.pop("contributing_sources", self.contributing_sources)
        bold = changes.pop("bold", self.bold)
        italic = changes.pop("italic", self.italic)
        underline = changes.pop("underline", self.underline)
        strikeout = changes.pop("strikeout", self.strikeout)
        mark = changes.pop("mark", self.mark)
        superscript = changes.pop("superscript", self.superscript)
        subscript = changes.pop("subscript", self.subscript)
        spans = changes.pop("spans", self.spans)
        words = changes.pop("words", self.words)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            text,
            break_before,
            bbox,
            advance_bbox,
            ink_bbox,
            kind,
            source,
            confidence,
            baseline,
            contributing_sources,
            bold,
            italic,
            underline,
            strikeout,
            mark,
            superscript,
            subscript,
            spans,
            words,
        )

    def _post_init(self) -> None:
        object.__setattr__(self, "words", internal_reconcile_text_words(self.text, self.words))
        if self.spans and "".join(span.text for span in self.spans) != self.text:
            object.__setattr__(self, "spans", ())

    def styled_spans(self) -> tuple[TextSpan, ...]:
        if self.spans:
            if self.underline or self.strikeout:
                return tuple(
                    replace(
                        span,
                        underline=span.underline or self.underline,
                        strikeout=span.strikeout or self.strikeout,
                    )
                    for span in self.spans
                )
            return self.spans
        return (
            TextSpan(
                text=self.text,
                bold=self.bold,
                italic=self.italic,
                underline=self.underline,
                strikeout=self.strikeout,
                mark=self.mark,
                superscript=self.superscript,
                subscript=self.subscript,
            ),
        )


class Block:
    __slots__ = (
        "order",
        "kind",
        "lines",
        "bbox",
        "column_index",
        "rotation",
        "confidence",
        "level",
        "provenance",
    )

    order: int
    kind: BlockKind
    lines: tuple[TextLine, ...]
    bbox: Rectangle | None
    column_index: int | None
    rotation: int
    confidence: float | None
    level: int | None
    provenance: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "order",
        "kind",
        "lines",
        "bbox",
        "column_index",
        "rotation",
        "confidence",
        "level",
        "provenance",
    )
    __match_args__ = (
        "order",
        "kind",
        "lines",
        "bbox",
        "column_index",
        "rotation",
        "confidence",
        "level",
        "provenance",
    )

    def __init__(
        self,
        order: int,
        kind: BlockKind,
        lines: tuple[TextLine, ...] = (),
        bbox: Rectangle | None = None,
        column_index: int | None = None,
        rotation: int = 0,
        confidence: float | None = None,
        level: int | None = None,
        provenance: tuple[str, ...] = (),
    ) -> None:
        internal_frozen_setattr(self, "order", order)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "lines", lines)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "column_index", column_index)
        internal_frozen_setattr(self, "rotation", rotation)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "level", level)
        internal_frozen_setattr(self, "provenance", provenance)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"order={self.order!r}, "
            f"kind={self.kind!r}, "
            f"lines={self.lines!r}, "
            f"bbox={self.bbox!r}, "
            f"column_index={self.column_index!r}, "
            f"rotation={self.rotation!r}, "
            f"confidence={self.confidence!r}, "
            f"level={self.level!r}, "
            f"provenance={self.provenance!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.order == other.order
            and self.kind == other.kind
            and self.lines == other.lines
            and self.bbox == other.bbox
            and self.column_index == other.column_index
            and self.rotation == other.rotation
            and self.confidence == other.confidence
            and self.level == other.level
            and self.provenance == other.provenance
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.order,
                self.kind,
                self.lines,
                self.bbox,
                self.column_index,
                self.rotation,
                self.confidence,
                self.level,
                self.provenance,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        order = changes.pop("order", self.order)
        kind = changes.pop("kind", self.kind)
        lines = changes.pop("lines", self.lines)
        bbox = changes.pop("bbox", self.bbox)
        column_index = changes.pop("column_index", self.column_index)
        rotation = changes.pop("rotation", self.rotation)
        confidence = changes.pop("confidence", self.confidence)
        level = changes.pop("level", self.level)
        provenance = changes.pop("provenance", self.provenance)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            order,
            kind,
            lines,
            bbox,
            column_index,
            rotation,
            confidence,
            level,
            provenance,
        )

    @property
    def text(self) -> str:
        parts: list[str] = []
        for line in self.lines:
            if parts:
                parts.append("\n" * max(1, line.break_before))
            parts.append(line.text)
        return "".join(parts)


PageElement: TypeAlias = Block | Table | Figure


class ContentNode:
    __slots__ = ("node_id", "kind", "payload", "page_number")

    node_id: int
    kind: str
    payload: PageElement
    page_number: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("node_id", "kind", "payload", "page_number")
    __match_args__ = ("node_id", "kind", "payload", "page_number")

    def __init__(
        self,
        node_id: int,
        kind: str,
        payload: PageElement,
        page_number: int | None = None,
    ) -> None:
        internal_frozen_setattr(self, "node_id", node_id)
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "payload", payload)
        internal_frozen_setattr(self, "page_number", page_number)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"node_id={self.node_id!r}, "
            f"kind={self.kind!r}, "
            f"payload={self.payload!r}, "
            f"page_number={self.page_number!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.node_id == other.node_id
            and self.kind == other.kind
            and self.payload == other.payload
            and self.page_number == other.page_number
        )

    def __hash__(self) -> int:
        return hash((self.node_id, self.kind, self.payload, self.page_number))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        node_id = changes.pop("node_id", self.node_id)
        kind = changes.pop("kind", self.kind)
        payload = changes.pop("payload", self.payload)
        page_number = changes.pop("page_number", self.page_number)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(node_id, kind, payload, page_number)

    @property
    def bbox(self) -> Rectangle | None:
        return self.payload.bbox

    @property
    def provenance(self) -> tuple[str, ...]:
        value = getattr(self.payload, "provenance", ())
        if value:
            return tuple(value)
        metadata = getattr(self.payload, "metadata", {})
        source = metadata.get("source") if isinstance(metadata, MappingABC) else None
        return (str(source),) if source else ()


class TextView:
    __slots__ = ("elements", "page_number")

    elements: tuple[PageElement, ...]
    page_number: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("elements", "page_number")
    __match_args__ = ("elements", "page_number")

    def __init__(self, elements: tuple[PageElement, ...], page_number: int | None = None) -> None:
        internal_frozen_setattr(self, "elements", elements)
        internal_frozen_setattr(self, "page_number", page_number)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"elements={self.elements!r}, "
            f"page_number={self.page_number!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.elements == other.elements and self.page_number == other.page_number

    def __hash__(self) -> int:
        return hash((self.elements, self.page_number))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        elements = changes.pop("elements", self.elements)
        page_number = changes.pop("page_number", self.page_number)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(elements, page_number)

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return tuple(line for block in self.blocks for line in block.lines)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(element for element in self.elements if isinstance(element, Block))

    @property
    def words(self) -> tuple[TextWord, ...]:
        words: list[TextWord] = []
        line_index = 0
        for block_index, block in enumerate(self.blocks):
            for line in block.lines:
                for word_index, word in enumerate(line.words):
                    words.append(
                        replace(
                            word,
                            line_index=line_index,
                            word_index=word_index,
                            block_index=block_index,
                            page_number=self.page_number,
                            source=line.source if word.source == UNKNOWN else word.source,
                        )
                    )
                line_index += 1
        return tuple(words)

    @property
    def text(self) -> str:
        parts: list[str] = []
        for element in self.elements:
            if isinstance(element, Block):
                text = element.text
            elif isinstance(element, Table):
                text = "\n".join("\t".join(cell.text for cell in row) for row in element.rows)
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n\n".join(parts)


class TextLineReference:
    __slots__ = ("page_number", "line_index", "line")

    page_number: int
    line_index: int
    line: TextLine

    __fields__: ClassVar[tuple[str, ...]] = ("page_number", "line_index", "line")
    __match_args__ = ("page_number", "line_index", "line")

    def __init__(self, page_number: int, line_index: int, line: TextLine) -> None:
        internal_frozen_setattr(self, "page_number", page_number)
        internal_frozen_setattr(self, "line_index", line_index)
        internal_frozen_setattr(self, "line", line)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_number={self.page_number!r}, "
            f"line_index={self.line_index!r}, "
            f"line={self.line!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_number == other.page_number
            and self.line_index == other.line_index
            and self.line == other.line
        )

    def __hash__(self) -> int:
        return hash((self.page_number, self.line_index, self.line))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        page_number = changes.pop("page_number", self.page_number)
        line_index = changes.pop("line_index", self.line_index)
        line = changes.pop("line", self.line)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(page_number, line_index, line)


class TableView:
    __slots__ = ("tables", "page_number")

    tables: tuple[Table, ...]
    page_number: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("tables", "page_number")
    __match_args__ = ("tables", "page_number")

    def __init__(self, tables: tuple[Table, ...], page_number: int | None = None) -> None:
        internal_frozen_setattr(self, "tables", tables)
        internal_frozen_setattr(self, "page_number", page_number)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"tables={self.tables!r}, "
            f"page_number={self.page_number!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.tables == other.tables and self.page_number == other.page_number

    def __hash__(self) -> int:
        return hash((self.tables, self.page_number))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        tables = changes.pop("tables", self.tables)
        page_number = changes.pop("page_number", self.page_number)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(tables, page_number)


class TableReference:
    __slots__ = ("page_number", "table_index", "table")

    page_number: int
    table_index: int
    table: Table

    __fields__: ClassVar[tuple[str, ...]] = ("page_number", "table_index", "table")
    __match_args__ = ("page_number", "table_index", "table")

    def __init__(self, page_number: int, table_index: int, table: Table) -> None:
        internal_frozen_setattr(self, "page_number", page_number)
        internal_frozen_setattr(self, "table_index", table_index)
        internal_frozen_setattr(self, "table", table)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_number={self.page_number!r}, "
            f"table_index={self.table_index!r}, "
            f"table={self.table!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_number == other.page_number
            and self.table_index == other.table_index
            and self.table == other.table
        )

    def __hash__(self) -> int:
        return hash((self.page_number, self.table_index, self.table))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        page_number = changes.pop("page_number", self.page_number)
        table_index = changes.pop("table_index", self.table_index)
        table = changes.pop("table", self.table)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(page_number, table_index, table)


class DocumentTextView:
    __slots__ = ("pages",)

    pages: tuple[TextView, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("pages",)
    __match_args__ = ("pages",)

    def __init__(self, pages: tuple[TextView, ...]) -> None:
        internal_frozen_setattr(self, "pages", pages)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(pages={self.pages!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.pages == other.pages

    def __hash__(self) -> int:
        return hash((self.pages,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        pages = changes.pop("pages", self.pages)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(pages)

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return tuple(line for page in self.pages for line in page.lines)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(block for page in self.pages for block in page.blocks)

    @property
    def line_references(self) -> tuple[TextLineReference, ...]:
        return tuple(
            TextLineReference(
                page_number=(page.page_number if page.page_number is not None else page_number),
                line_index=index,
                line=line,
            )
            for page_number, page in enumerate(self.pages, start=1)
            for index, line in enumerate(page.lines)
        )

    @property
    def words(self) -> tuple[TextWord, ...]:
        words: list[TextWord] = []
        for page in self.pages:
            words.extend(page.words)
        return tuple(words)

    @property
    def text(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"


class DocumentTableView:
    __slots__ = ("pages",)

    pages: tuple[TableView, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("pages",)
    __match_args__ = ("pages",)

    def __init__(self, pages: tuple[TableView, ...]) -> None:
        internal_frozen_setattr(self, "pages", pages)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(pages={self.pages!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.pages == other.pages

    def __hash__(self) -> int:
        return hash((self.pages,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        pages = changes.pop("pages", self.pages)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(pages)

    @property
    def tables(self) -> tuple[Table, ...]:
        return tuple(table for page in self.pages for table in page.tables)

    @property
    def references(self) -> tuple[TableReference, ...]:
        return tuple(
            TableReference(
                page_number=(page.page_number if page.page_number is not None else page_number),
                table_index=table_index,
                table=table,
            )
            for page_number, page in enumerate(self.pages, start=1)
            for table_index, table in enumerate(page.tables)
        )


class Page:
    __slots__ = (
        "page_number",
        "page_label",
        "width",
        "height",
        "rotation",
        "blocks",
        "page_class",
        "base_route",
        "confidence",
        "tables",
        "figures",
        "links",
        "annotations",
        "form_fields",
        "header",
        "footer",
        "diagnostics",
        "cropbox",
        "user_unit",
        "internal_elements",
    )

    page_number: int
    page_label: str | None
    width: float
    height: float
    rotation: int
    blocks: tuple[Block, ...]
    page_class: str
    base_route: str
    confidence: float | None
    tables: tuple[Table, ...]
    figures: tuple[Figure, ...]
    links: tuple[Link, ...]
    annotations: tuple[Annotation, ...]
    form_fields: tuple[FormField, ...]
    header: str
    footer: str
    diagnostics: tuple[Diagnostic, ...]
    cropbox: Rectangle | None
    user_unit: float
    internal_elements: tuple[PageElement, ...]

    __fields__: ClassVar[tuple[str, ...]] = (
        "page_number",
        "page_label",
        "width",
        "height",
        "rotation",
        "blocks",
        "page_class",
        "base_route",
        "confidence",
        "tables",
        "figures",
        "links",
        "annotations",
        "form_fields",
        "header",
        "footer",
        "diagnostics",
        "cropbox",
        "user_unit",
        "internal_elements",
    )
    __match_args__ = (
        "page_number",
        "page_label",
        "width",
        "height",
        "rotation",
        "blocks",
        "page_class",
        "base_route",
        "confidence",
        "tables",
        "figures",
        "links",
        "annotations",
        "form_fields",
        "header",
        "footer",
        "diagnostics",
        "cropbox",
    )

    def __init__(
        self,
        page_number: int,
        page_label: str | None = None,
        width: float = 0.0,
        height: float = 0.0,
        rotation: int = 0,
        blocks: tuple[Block, ...] = (),
        page_class: str = UNKNOWN,
        base_route: str = UNKNOWN,
        confidence: float | None = None,
        tables: tuple[Table, ...] = (),
        figures: tuple[Figure, ...] = (),
        links: tuple[Link, ...] = (),
        annotations: tuple[Annotation, ...] = (),
        form_fields: tuple[FormField, ...] = (),
        header: str = "",
        footer: str = "",
        diagnostics: tuple[Diagnostic, ...] = (),
        cropbox: Rectangle | None = None,
        *,
        user_unit: float = 1.0,
    ) -> None:
        internal_frozen_setattr(self, "page_number", page_number)
        internal_frozen_setattr(self, "page_label", page_label)
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "height", height)
        internal_frozen_setattr(self, "rotation", rotation)
        internal_frozen_setattr(self, "blocks", blocks)
        internal_frozen_setattr(self, "page_class", page_class)
        internal_frozen_setattr(self, "base_route", base_route)
        internal_frozen_setattr(self, "confidence", confidence)
        internal_frozen_setattr(self, "tables", tables)
        internal_frozen_setattr(self, "figures", figures)
        internal_frozen_setattr(self, "links", links)
        internal_frozen_setattr(self, "annotations", annotations)
        internal_frozen_setattr(self, "form_fields", form_fields)
        internal_frozen_setattr(self, "header", header)
        internal_frozen_setattr(self, "footer", footer)
        internal_frozen_setattr(self, "diagnostics", diagnostics)
        internal_frozen_setattr(self, "cropbox", cropbox)
        internal_frozen_setattr(self, "user_unit", user_unit)
        internal_frozen_setattr(self, "internal_elements", ())
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_number={self.page_number!r}, "
            f"page_label={self.page_label!r}, "
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"rotation={self.rotation!r}, "
            f"blocks={self.blocks!r}, "
            f"page_class={self.page_class!r}, "
            f"base_route={self.base_route!r}, "
            f"confidence={self.confidence!r}, "
            f"tables={self.tables!r}, "
            f"figures={self.figures!r}, "
            f"links={self.links!r}, "
            f"annotations={self.annotations!r}, "
            f"form_fields={self.form_fields!r}, "
            f"header={self.header!r}, "
            f"footer={self.footer!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"cropbox={self.cropbox!r}, "
            f"user_unit={self.user_unit!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.page_number == other.page_number
            and self.page_label == other.page_label
            and self.width == other.width
            and self.height == other.height
            and self.rotation == other.rotation
            and self.blocks == other.blocks
            and self.page_class == other.page_class
            and self.base_route == other.base_route
            and self.confidence == other.confidence
            and self.tables == other.tables
            and self.figures == other.figures
            and self.links == other.links
            and self.annotations == other.annotations
            and self.form_fields == other.form_fields
            and self.header == other.header
            and self.footer == other.footer
            and self.diagnostics == other.diagnostics
            and self.cropbox == other.cropbox
            and self.user_unit == other.user_unit
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.page_number,
                self.page_label,
                self.width,
                self.height,
                self.rotation,
                self.blocks,
                self.page_class,
                self.base_route,
                self.confidence,
                self.tables,
                self.figures,
                self.links,
                self.annotations,
                self.form_fields,
                self.header,
                self.footer,
                self.diagnostics,
                self.cropbox,
                self.user_unit,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        page_number = changes.pop("page_number", self.page_number)
        page_label = changes.pop("page_label", self.page_label)
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        rotation = changes.pop("rotation", self.rotation)
        blocks = changes.pop("blocks", self.blocks)
        page_class = changes.pop("page_class", self.page_class)
        base_route = changes.pop("base_route", self.base_route)
        confidence = changes.pop("confidence", self.confidence)
        tables = changes.pop("tables", self.tables)
        figures = changes.pop("figures", self.figures)
        links = changes.pop("links", self.links)
        annotations = changes.pop("annotations", self.annotations)
        form_fields = changes.pop("form_fields", self.form_fields)
        header = changes.pop("header", self.header)
        footer = changes.pop("footer", self.footer)
        diagnostics = changes.pop("diagnostics", self.diagnostics)
        cropbox = changes.pop("cropbox", self.cropbox)
        user_unit = changes.pop("user_unit", self.user_unit)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            page_number,
            page_label,
            width,
            height,
            rotation,
            blocks,
            page_class,
            base_route,
            confidence,
            tables,
            figures,
            links,
            annotations,
            form_fields,
            header,
            footer,
            diagnostics,
            cropbox,
            user_unit=user_unit,
        )

    def _post_init(self) -> None:
        object.__setattr__(
            self,
            "internal_elements",
            tuple(sorted((*self.blocks, *self.tables, *self.figures), key=lambda item: item.order)),
        )

    @property
    def width_points(self) -> float:
        return self.width * self.user_unit

    @property
    def height_points(self) -> float:
        return self.height * self.user_unit

    @property
    def elements(self) -> tuple[PageElement, ...]:
        return self.internal_elements

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        return tuple(self.internal_nodes())

    def internal_nodes(self, start_id: int = 0) -> Iterator[ContentNode]:
        for index, element in enumerate(self.elements, start=start_id):
            yield ContentNode(
                node_id=index,
                kind=type(element).__name__.casefold(),
                payload=element,
                page_number=self.page_number,
            )

    @property
    def text_view(self) -> TextView:
        return TextView(self.elements, page_number=self.page_number)

    @property
    def words(self) -> tuple[TextWord, ...]:
        return self.text_view.words

    @property
    def table_view(self) -> TableView:
        return TableView(self.tables, page_number=self.page_number)

    @property
    def text(self) -> str:
        return self.text_view.text

    def to_markdown(self) -> str:
        from core_pdf.impl.output.serialize import page_to_markdown

        return page_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.output.serialize import page_to_html

        return page_to_html(self)


class Diagnostic:
    __slots__ = ("code", "message", "severity", "page_number")

    code: str
    message: str
    severity: str
    page_number: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("code", "message", "severity", "page_number")
    __match_args__ = ("code", "message", "severity", "page_number")

    def __init__(
        self,
        code: str,
        message: str,
        severity: str = "warning",
        page_number: int | None = None,
    ) -> None:
        internal_frozen_setattr(self, "code", code)
        internal_frozen_setattr(self, "message", message)
        internal_frozen_setattr(self, "severity", severity)
        internal_frozen_setattr(self, "page_number", page_number)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"severity={self.severity!r}, "
            f"page_number={self.page_number!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code == other.code
            and self.message == other.message
            and self.severity == other.severity
            and self.page_number == other.page_number
        )

    def __hash__(self) -> int:
        return hash((self.code, self.message, self.severity, self.page_number))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        code = changes.pop("code", self.code)
        message = changes.pop("message", self.message)
        severity = changes.pop("severity", self.severity)
        page_number = changes.pop("page_number", self.page_number)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(code, message, severity, page_number)


class Document:
    __slots__ = ("pages", "metadata", "diagnostics", "schema_version")

    pages: tuple[Page, ...]
    metadata: Mapping[str, Any]
    diagnostics: tuple[Diagnostic, ...]
    schema_version: str

    __fields__: ClassVar[tuple[str, ...]] = ("pages", "metadata", "diagnostics", "schema_version")
    __match_args__ = ("pages", "metadata", "diagnostics", "schema_version")

    def __init__(
        self,
        pages: tuple[Page, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        diagnostics: tuple[Diagnostic, ...] = (),
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        internal_frozen_setattr(self, "pages", pages)
        internal_frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        internal_frozen_setattr(self, "diagnostics", diagnostics)
        internal_frozen_setattr(self, "schema_version", schema_version)
        self._post_init()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"pages={self.pages!r}, "
            f"metadata={self.metadata!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"schema_version={self.schema_version!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.pages == other.pages
            and self.metadata == other.metadata
            and self.diagnostics == other.diagnostics
            and self.schema_version == other.schema_version
        )

    def __hash__(self) -> int:
        return hash((self.pages, self.metadata, self.diagnostics, self.schema_version))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        pages = changes.pop("pages", self.pages)
        metadata = changes.pop("metadata", self.metadata)
        diagnostics = changes.pop("diagnostics", self.diagnostics)
        schema_version = changes.pop("schema_version", self.schema_version)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(pages, metadata, diagnostics, schema_version)

    def _post_init(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported structured schema version: {self.schema_version}")
        object.__setattr__(self, "metadata", internal_freeze(self.metadata))

    @property
    def text_view(self) -> DocumentTextView:
        return DocumentTextView(tuple(page.text_view for page in self.pages))

    @property
    def table_view(self) -> DocumentTableView:
        return DocumentTableView(tuple(page.table_view for page in self.pages))

    @property
    def nodes(self) -> tuple[ContentNode, ...]:
        nodes: list[ContentNode] = []
        for page in self.pages:
            nodes.extend(page.internal_nodes(start_id=len(nodes)))
        return tuple(nodes)

    @property
    def text(self) -> str:
        return self.text_view.text

    @property
    def words(self) -> tuple[TextWord, ...]:
        return self.text_view.words

    @property
    def lines(self) -> tuple[TextLine, ...]:
        return self.text_view.lines

    @property
    def blocks(self) -> tuple[Block, ...]:
        return self.text_view.blocks

    def to_json_dict(self) -> dict[str, JsonValue]:
        from core_pdf.impl.output.serialize import document_to_json_dict

        return document_to_json_dict(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_pdf.impl.output.serialize import document_to_json

        return document_to_json(self, indent=indent, sort_keys=sort_keys)

    def to_markdown(self) -> str:
        from core_pdf.impl.output.serialize import document_to_markdown

        return document_to_markdown(self)

    def to_html(self) -> str:
        from core_pdf.impl.output.serialize import document_to_html

        return document_to_html(self)

    def to_csv(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.output.serialize import document_to_csv

        return document_to_csv(self, pages=pages)

    def to_tei(self, *, pages: PageSelection | None = None) -> str:
        from core_pdf.impl.output.serialize import document_to_tei

        return document_to_tei(self, pages=pages)
