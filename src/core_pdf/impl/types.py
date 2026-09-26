# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import ClassVar, Protocol, Self, TypeAlias, TypeVar

from core_pdf_spec.types import (
    MISSING,
    MissingObject,
    PdfByteBuffer,
    PdfName,
    PdfReference,
    PdfString,
    Rectangle,
)
from core_records import FrozenFields as FrozenFields
from core_records import PickleFields as PickleFields
from core_records import Record as Record
from core_records import ReplaceFields as ReplaceFields
from core_records import ReprFields as ReprFields
from core_records import frozen_setattr as frozen_setattr


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


class SeekableBinaryReader(BinaryReader, Protocol):
    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def fileno(self) -> int: ...


PathSource: TypeAlias = str | PathLike[str]
PdfSource: TypeAlias = (
    PathSource | bytes | bytearray | memoryview | BinaryReader | SeekableBinaryReader
)


RecordT = TypeVar("RecordT")


class PageScoped[RecordT](Record):
    __slots__ = ("page_index", "page_number", "page_label", "record")

    page_index: int
    page_number: int
    page_label: str | None
    record: RecordT

    __fields__: ClassVar[tuple[str, ...]] = ("page_index", "page_number", "page_label", "record")
    __match_args__ = ("page_index", "page_number", "page_label", "record")

    def __init__(
        self,
        page_index: int,
        page_number: int,
        page_label: str | None,
        record: RecordT,
    ) -> None:
        frozen_setattr(self, "page_index", page_index)
        frozen_setattr(self, "page_number", page_number)
        frozen_setattr(self, "page_label", page_label)
        frozen_setattr(self, "record", record)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__ or not isinstance(other, PageScoped):
            return NotImplemented
        return (
            self.page_index == other.page_index
            and self.page_number == other.page_number
            and self.page_label == other.page_label
            and self.record == other.record
        )

    def __hash__(self) -> int:
        return hash((self.page_index, self.page_number, self.page_label, self.record))


class TextWord(Record):
    __slots__ = ("text", "bbox", "line_index", "word_index", "block_index", "page_number", "source")

    text: str
    bbox: Rectangle | None
    line_index: int
    word_index: int
    block_index: int
    page_number: int | None
    source: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "bbox",
        "line_index",
        "word_index",
        "block_index",
        "page_number",
        "source",
    )
    __match_args__ = (
        "text",
        "bbox",
        "line_index",
        "word_index",
        "block_index",
        "page_number",
        "source",
    )

    def __init__(
        self,
        text: str,
        bbox: Rectangle | None = None,
        line_index: int = 0,
        word_index: int = 0,
        block_index: int = 0,
        page_number: int | None = None,
        source: str = "unknown",
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "line_index", line_index)
        frozen_setattr(self, "word_index", word_index)
        frozen_setattr(self, "block_index", block_index)
        frozen_setattr(self, "page_number", page_number)
        frozen_setattr(self, "source", source)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.line_index == other.line_index
            and self.word_index == other.word_index
            and self.block_index == other.block_index
            and self.page_number == other.page_number
            and self.source == other.source
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.bbox,
                self.line_index,
                self.word_index,
                self.block_index,
                self.page_number,
                self.source,
            )
        )


class DrawingRecord(Record):
    __slots__ = (
        "kind",
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "path",
        "items",
        "rect",
    )

    kind: str
    seqno: int
    fill: tuple[float, ...] | None
    fill_pattern: Mapping[object, object] | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: Mapping[object, object] | None
    stroke_opacity: float | None
    line_width: float
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    fill_rule: str
    blend_mode: str | None
    soft_mask_alpha: float | None
    raw_data: bytes | memoryview | None
    dictionary: Mapping[object, object] | None
    image_source: object | None
    image_clip: Rectangle | None
    path: object | None
    items: tuple[object, ...]
    rect: Rectangle | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "kind",
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "path",
        "items",
        "rect",
    )
    __match_args__ = (
        "kind",
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "path",
        "items",
        "rect",
    )

    def __init__(
        self,
        kind: str,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_pattern: Mapping[object, object] | None,
        fill_opacity: float | None,
        stroke_color: tuple[float, ...] | None,
        stroke_pattern: Mapping[object, object] | None,
        stroke_opacity: float | None,
        line_width: float,
        line_cap: int,
        line_join: int,
        dash_pattern: tuple[list[float], float] | None,
        fill_rule: str,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        raw_data: bytes | memoryview | None,
        dictionary: Mapping[object, object] | None,
        image_source: object | None,
        image_clip: Rectangle | None,
        path: object | None,
        items: tuple[object, ...],
        rect: Rectangle | None,
    ) -> None:
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "seqno", seqno)
        frozen_setattr(self, "fill", fill)
        frozen_setattr(self, "fill_pattern", fill_pattern)
        frozen_setattr(self, "fill_opacity", fill_opacity)
        frozen_setattr(self, "stroke_color", stroke_color)
        frozen_setattr(self, "stroke_pattern", stroke_pattern)
        frozen_setattr(self, "stroke_opacity", stroke_opacity)
        frozen_setattr(self, "line_width", line_width)
        frozen_setattr(self, "line_cap", line_cap)
        frozen_setattr(self, "line_join", line_join)
        frozen_setattr(self, "dash_pattern", dash_pattern)
        frozen_setattr(self, "fill_rule", fill_rule)
        frozen_setattr(self, "blend_mode", blend_mode)
        frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        frozen_setattr(self, "raw_data", raw_data)
        frozen_setattr(self, "dictionary", dictionary)
        frozen_setattr(self, "image_source", image_source)
        frozen_setattr(self, "image_clip", image_clip)
        frozen_setattr(self, "path", path)
        frozen_setattr(self, "items", items)
        frozen_setattr(self, "rect", rect)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.kind == other.kind
            and self.seqno == other.seqno
            and self.fill == other.fill
            and self.fill_pattern == other.fill_pattern
            and self.fill_opacity == other.fill_opacity
            and self.stroke_color == other.stroke_color
            and self.stroke_pattern == other.stroke_pattern
            and self.stroke_opacity == other.stroke_opacity
            and self.line_width == other.line_width
            and self.line_cap == other.line_cap
            and self.line_join == other.line_join
            and self.dash_pattern == other.dash_pattern
            and self.fill_rule == other.fill_rule
            and self.blend_mode == other.blend_mode
            and self.soft_mask_alpha == other.soft_mask_alpha
            and self.raw_data == other.raw_data
            and self.dictionary == other.dictionary
            and self.image_source == other.image_source
            and self.image_clip == other.image_clip
            and self.path == other.path
            and self.items == other.items
            and self.rect == other.rect
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.kind,
                self.seqno,
                self.fill,
                self.fill_pattern,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_pattern,
                self.stroke_opacity,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.dash_pattern,
                self.fill_rule,
                self.blend_mode,
                self.soft_mask_alpha,
                self.raw_data,
                self.dictionary,
                self.image_source,
                self.image_clip,
                self.path,
                self.items,
                self.rect,
            )
        )

    @classmethod
    def from_captured(cls, source: object, **overrides: object) -> Self:
        values = {name: getattr(source, name) for name in DRAWING_FIELD_NAMES}
        values.update(overrides)
        return cls(**values)


DRAWING_FIELD_NAMES: tuple[str, ...] = DrawingRecord.__fields__


class ImageMetadata(Record):
    __slots__ = (
        "width",
        "height",
        "channels",
        "color_model",
        "alpha",
        "stride",
        "source_rect",
        "transform",
        "clipping",
    )

    width: int
    height: int
    channels: int
    color_model: str
    alpha: bool
    stride: int
    source_rect: Rectangle
    transform: object | None
    clipping: Rectangle | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "width",
        "height",
        "channels",
        "color_model",
        "alpha",
        "stride",
        "source_rect",
        "transform",
        "clipping",
    )
    __match_args__ = (
        "width",
        "height",
        "channels",
        "color_model",
        "alpha",
        "stride",
        "source_rect",
        "transform",
        "clipping",
    )

    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        color_model: str,
        alpha: bool,
        stride: int,
        source_rect: Rectangle,
        transform: object | None,
        clipping: Rectangle | None,
    ) -> None:
        frozen_setattr(self, "width", width)
        frozen_setattr(self, "height", height)
        frozen_setattr(self, "channels", channels)
        frozen_setattr(self, "color_model", color_model)
        frozen_setattr(self, "alpha", alpha)
        frozen_setattr(self, "stride", stride)
        frozen_setattr(self, "source_rect", source_rect)
        frozen_setattr(self, "transform", transform)
        frozen_setattr(self, "clipping", clipping)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.channels == other.channels
            and self.color_model == other.color_model
            and self.alpha == other.alpha
            and self.stride == other.stride
            and self.source_rect == other.source_rect
            and self.transform == other.transform
            and self.clipping == other.clipping
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.width,
                self.height,
                self.channels,
                self.color_model,
                self.alpha,
                self.stride,
                self.source_rect,
                self.transform,
                self.clipping,
            )
        )


class ImageRecord(DrawingRecord):
    __slots__ = ("data", "image_metadata")

    data: object | None
    image_metadata: ImageMetadata | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "kind",
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "path",
        "items",
        "rect",
        "data",
        "image_metadata",
    )
    __match_args__ = (
        "kind",
        "seqno",
        "fill",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "line_width",
        "line_cap",
        "line_join",
        "dash_pattern",
        "fill_rule",
        "blend_mode",
        "soft_mask_alpha",
        "raw_data",
        "dictionary",
        "image_source",
        "image_clip",
        "path",
        "items",
        "rect",
        "data",
        "image_metadata",
    )

    def __init__(
        self,
        kind: str,
        seqno: int,
        fill: tuple[float, ...] | None,
        fill_pattern: Mapping[object, object] | None,
        fill_opacity: float | None,
        stroke_color: tuple[float, ...] | None,
        stroke_pattern: Mapping[object, object] | None,
        stroke_opacity: float | None,
        line_width: float,
        line_cap: int,
        line_join: int,
        dash_pattern: tuple[list[float], float] | None,
        fill_rule: str,
        blend_mode: str | None,
        soft_mask_alpha: float | None,
        raw_data: bytes | memoryview | None,
        dictionary: Mapping[object, object] | None,
        image_source: object | None,
        image_clip: Rectangle | None,
        path: object | None,
        items: tuple[object, ...],
        rect: Rectangle | None,
        data: object | None = None,
        image_metadata: ImageMetadata | None = None,
    ) -> None:
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "seqno", seqno)
        frozen_setattr(self, "fill", fill)
        frozen_setattr(self, "fill_pattern", fill_pattern)
        frozen_setattr(self, "fill_opacity", fill_opacity)
        frozen_setattr(self, "stroke_color", stroke_color)
        frozen_setattr(self, "stroke_pattern", stroke_pattern)
        frozen_setattr(self, "stroke_opacity", stroke_opacity)
        frozen_setattr(self, "line_width", line_width)
        frozen_setattr(self, "line_cap", line_cap)
        frozen_setattr(self, "line_join", line_join)
        frozen_setattr(self, "dash_pattern", dash_pattern)
        frozen_setattr(self, "fill_rule", fill_rule)
        frozen_setattr(self, "blend_mode", blend_mode)
        frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        frozen_setattr(self, "raw_data", raw_data)
        frozen_setattr(self, "dictionary", dictionary)
        frozen_setattr(self, "image_source", image_source)
        frozen_setattr(self, "image_clip", image_clip)
        frozen_setattr(self, "path", path)
        frozen_setattr(self, "items", items)
        frozen_setattr(self, "rect", rect)
        frozen_setattr(self, "data", data)
        frozen_setattr(self, "image_metadata", image_metadata)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.kind == other.kind
            and self.seqno == other.seqno
            and self.fill == other.fill
            and self.fill_pattern == other.fill_pattern
            and self.fill_opacity == other.fill_opacity
            and self.stroke_color == other.stroke_color
            and self.stroke_pattern == other.stroke_pattern
            and self.stroke_opacity == other.stroke_opacity
            and self.line_width == other.line_width
            and self.line_cap == other.line_cap
            and self.line_join == other.line_join
            and self.dash_pattern == other.dash_pattern
            and self.fill_rule == other.fill_rule
            and self.blend_mode == other.blend_mode
            and self.soft_mask_alpha == other.soft_mask_alpha
            and self.raw_data == other.raw_data
            and self.dictionary == other.dictionary
            and self.image_source == other.image_source
            and self.image_clip == other.image_clip
            and self.path == other.path
            and self.items == other.items
            and self.rect == other.rect
            and self.data == other.data
            and self.image_metadata == other.image_metadata
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.kind,
                self.seqno,
                self.fill,
                self.fill_pattern,
                self.fill_opacity,
                self.stroke_color,
                self.stroke_pattern,
                self.stroke_opacity,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.dash_pattern,
                self.fill_rule,
                self.blend_mode,
                self.soft_mask_alpha,
                self.raw_data,
                self.dictionary,
                self.image_source,
                self.image_clip,
                self.path,
                self.items,
                self.rect,
                self.data,
                self.image_metadata,
            )
        )


__all__ = (
    "BinaryReader",
    "DrawingRecord",
    "ImageMetadata",
    "ImageRecord",
    "MISSING",
    "MissingObject",
    "PageScoped",
    "PathSource",
    "PdfByteBuffer",
    "PdfName",
    "PdfReference",
    "PdfSource",
    "PdfString",
    "Rectangle",
    "SeekableBinaryReader",
    "TextWord",
)
