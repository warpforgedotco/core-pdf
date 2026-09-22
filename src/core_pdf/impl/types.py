# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, ClassVar, Protocol, Self, TypeAlias, TypeVar

from core_pdf.impl.records import internal_Record
from core_pdf_spec.types import (
    MISSING,
    MissingObject,
    PdfByteBuffer,
    PdfName,
    PdfReference,
    PdfString,
    Rectangle,
)

internal_frozen_setattr = object.__setattr__


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


class PageScoped[RecordT](internal_Record):
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
        internal_frozen_setattr(self, "page_index", page_index)
        internal_frozen_setattr(self, "page_number", page_number)
        internal_frozen_setattr(self, "page_label", page_label)
        internal_frozen_setattr(self, "record", record)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"page_index={self.page_index!r}, "
            f"page_number={self.page_number!r}, "
            f"page_label={self.page_label!r}, "
            f"record={self.record!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        page_index = changes.pop("page_index", self.page_index)
        page_number = changes.pop("page_number", self.page_number)
        page_label = changes.pop("page_label", self.page_label)
        record = changes.pop("record", self.record)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(page_index, page_number, page_label, record)


class TextWord(internal_Record):
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
        internal_frozen_setattr(self, "text", text)
        internal_frozen_setattr(self, "bbox", bbox)
        internal_frozen_setattr(self, "line_index", line_index)
        internal_frozen_setattr(self, "word_index", word_index)
        internal_frozen_setattr(self, "block_index", block_index)
        internal_frozen_setattr(self, "page_number", page_number)
        internal_frozen_setattr(self, "source", source)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"line_index={self.line_index!r}, "
            f"word_index={self.word_index!r}, "
            f"block_index={self.block_index!r}, "
            f"page_number={self.page_number!r}, "
            f"source={self.source!r}"
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

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        line_index = changes.pop("line_index", self.line_index)
        word_index = changes.pop("word_index", self.word_index)
        block_index = changes.pop("block_index", self.block_index)
        page_number = changes.pop("page_number", self.page_number)
        source = changes.pop("source", self.source)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(text, bbox, line_index, word_index, block_index, page_number, source)


class DrawingRecord(internal_Record):
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
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "seqno", seqno)
        internal_frozen_setattr(self, "fill", fill)
        internal_frozen_setattr(self, "fill_pattern", fill_pattern)
        internal_frozen_setattr(self, "fill_opacity", fill_opacity)
        internal_frozen_setattr(self, "stroke_color", stroke_color)
        internal_frozen_setattr(self, "stroke_pattern", stroke_pattern)
        internal_frozen_setattr(self, "stroke_opacity", stroke_opacity)
        internal_frozen_setattr(self, "line_width", line_width)
        internal_frozen_setattr(self, "line_cap", line_cap)
        internal_frozen_setattr(self, "line_join", line_join)
        internal_frozen_setattr(self, "dash_pattern", dash_pattern)
        internal_frozen_setattr(self, "fill_rule", fill_rule)
        internal_frozen_setattr(self, "blend_mode", blend_mode)
        internal_frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        internal_frozen_setattr(self, "raw_data", raw_data)
        internal_frozen_setattr(self, "dictionary", dictionary)
        internal_frozen_setattr(self, "image_source", image_source)
        internal_frozen_setattr(self, "image_clip", image_clip)
        internal_frozen_setattr(self, "path", path)
        internal_frozen_setattr(self, "items", items)
        internal_frozen_setattr(self, "rect", rect)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"kind={self.kind!r}, "
            f"seqno={self.seqno!r}, "
            f"fill={self.fill!r}, "
            f"fill_pattern={self.fill_pattern!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_pattern={self.stroke_pattern!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"line_width={self.line_width!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"fill_rule={self.fill_rule!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"raw_data={self.raw_data!r}, "
            f"dictionary={self.dictionary!r}, "
            f"image_source={self.image_source!r}, "
            f"image_clip={self.image_clip!r}, "
            f"path={self.path!r}, "
            f"items={self.items!r}, "
            f"rect={self.rect!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        kind = changes.pop("kind", self.kind)
        seqno = changes.pop("seqno", self.seqno)
        fill = changes.pop("fill", self.fill)
        fill_pattern = changes.pop("fill_pattern", self.fill_pattern)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_pattern = changes.pop("stroke_pattern", self.stroke_pattern)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        line_width = changes.pop("line_width", self.line_width)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        fill_rule = changes.pop("fill_rule", self.fill_rule)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        raw_data = changes.pop("raw_data", self.raw_data)
        dictionary = changes.pop("dictionary", self.dictionary)
        image_source = changes.pop("image_source", self.image_source)
        image_clip = changes.pop("image_clip", self.image_clip)
        path = changes.pop("path", self.path)
        items = changes.pop("items", self.items)
        rect = changes.pop("rect", self.rect)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            kind,
            seqno,
            fill,
            fill_pattern,
            fill_opacity,
            stroke_color,
            stroke_pattern,
            stroke_opacity,
            line_width,
            line_cap,
            line_join,
            dash_pattern,
            fill_rule,
            blend_mode,
            soft_mask_alpha,
            raw_data,
            dictionary,
            image_source,
            image_clip,
            path,
            items,
            rect,
        )

    @classmethod
    def from_captured(cls, source: object, **overrides: object) -> Self:
        values = {name: getattr(source, name) for name in internal_DRAWING_FIELD_NAMES}
        values.update(overrides)
        return cls(**values)


internal_DRAWING_FIELD_NAMES: tuple[str, ...] = DrawingRecord.__fields__


class ImageMetadata(internal_Record):
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
        internal_frozen_setattr(self, "width", width)
        internal_frozen_setattr(self, "height", height)
        internal_frozen_setattr(self, "channels", channels)
        internal_frozen_setattr(self, "color_model", color_model)
        internal_frozen_setattr(self, "alpha", alpha)
        internal_frozen_setattr(self, "stride", stride)
        internal_frozen_setattr(self, "source_rect", source_rect)
        internal_frozen_setattr(self, "transform", transform)
        internal_frozen_setattr(self, "clipping", clipping)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"width={self.width!r}, "
            f"height={self.height!r}, "
            f"channels={self.channels!r}, "
            f"color_model={self.color_model!r}, "
            f"alpha={self.alpha!r}, "
            f"stride={self.stride!r}, "
            f"source_rect={self.source_rect!r}, "
            f"transform={self.transform!r}, "
            f"clipping={self.clipping!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        width = changes.pop("width", self.width)
        height = changes.pop("height", self.height)
        channels = changes.pop("channels", self.channels)
        color_model = changes.pop("color_model", self.color_model)
        alpha = changes.pop("alpha", self.alpha)
        stride = changes.pop("stride", self.stride)
        source_rect = changes.pop("source_rect", self.source_rect)
        transform = changes.pop("transform", self.transform)
        clipping = changes.pop("clipping", self.clipping)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            width,
            height,
            channels,
            color_model,
            alpha,
            stride,
            source_rect,
            transform,
            clipping,
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
        internal_frozen_setattr(self, "kind", kind)
        internal_frozen_setattr(self, "seqno", seqno)
        internal_frozen_setattr(self, "fill", fill)
        internal_frozen_setattr(self, "fill_pattern", fill_pattern)
        internal_frozen_setattr(self, "fill_opacity", fill_opacity)
        internal_frozen_setattr(self, "stroke_color", stroke_color)
        internal_frozen_setattr(self, "stroke_pattern", stroke_pattern)
        internal_frozen_setattr(self, "stroke_opacity", stroke_opacity)
        internal_frozen_setattr(self, "line_width", line_width)
        internal_frozen_setattr(self, "line_cap", line_cap)
        internal_frozen_setattr(self, "line_join", line_join)
        internal_frozen_setattr(self, "dash_pattern", dash_pattern)
        internal_frozen_setattr(self, "fill_rule", fill_rule)
        internal_frozen_setattr(self, "blend_mode", blend_mode)
        internal_frozen_setattr(self, "soft_mask_alpha", soft_mask_alpha)
        internal_frozen_setattr(self, "raw_data", raw_data)
        internal_frozen_setattr(self, "dictionary", dictionary)
        internal_frozen_setattr(self, "image_source", image_source)
        internal_frozen_setattr(self, "image_clip", image_clip)
        internal_frozen_setattr(self, "path", path)
        internal_frozen_setattr(self, "items", items)
        internal_frozen_setattr(self, "rect", rect)
        internal_frozen_setattr(self, "data", data)
        internal_frozen_setattr(self, "image_metadata", image_metadata)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"kind={self.kind!r}, "
            f"seqno={self.seqno!r}, "
            f"fill={self.fill!r}, "
            f"fill_pattern={self.fill_pattern!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_pattern={self.stroke_pattern!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"line_width={self.line_width!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"fill_rule={self.fill_rule!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"soft_mask_alpha={self.soft_mask_alpha!r}, "
            f"raw_data={self.raw_data!r}, "
            f"dictionary={self.dictionary!r}, "
            f"image_source={self.image_source!r}, "
            f"image_clip={self.image_clip!r}, "
            f"path={self.path!r}, "
            f"items={self.items!r}, "
            f"rect={self.rect!r}, "
            f"data={self.data!r}, "
            f"image_metadata={self.image_metadata!r}"
            ")"
        )

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

    def __replace__(self, /, **changes: Any) -> Self:
        kind = changes.pop("kind", self.kind)
        seqno = changes.pop("seqno", self.seqno)
        fill = changes.pop("fill", self.fill)
        fill_pattern = changes.pop("fill_pattern", self.fill_pattern)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_pattern = changes.pop("stroke_pattern", self.stroke_pattern)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        line_width = changes.pop("line_width", self.line_width)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        fill_rule = changes.pop("fill_rule", self.fill_rule)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        soft_mask_alpha = changes.pop("soft_mask_alpha", self.soft_mask_alpha)
        raw_data = changes.pop("raw_data", self.raw_data)
        dictionary = changes.pop("dictionary", self.dictionary)
        image_source = changes.pop("image_source", self.image_source)
        image_clip = changes.pop("image_clip", self.image_clip)
        path = changes.pop("path", self.path)
        items = changes.pop("items", self.items)
        rect = changes.pop("rect", self.rect)
        data = changes.pop("data", self.data)
        image_metadata = changes.pop("image_metadata", self.image_metadata)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            kind,
            seqno,
            fill,
            fill_pattern,
            fill_opacity,
            stroke_color,
            stroke_pattern,
            stroke_opacity,
            line_width,
            line_cap,
            line_join,
            dash_pattern,
            fill_rule,
            blend_mode,
            soft_mask_alpha,
            raw_data,
            dictionary,
            image_source,
            image_clip,
            path,
            items,
            rect,
            data,
            image_metadata,
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
