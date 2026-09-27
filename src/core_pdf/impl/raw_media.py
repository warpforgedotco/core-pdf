# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Self

from core_pdf.impl.types import Record, Rectangle, frozen_setattr


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

    __fields__: ClassVar[tuple[str, ...]] = (*DrawingRecord.__fields__, "data", "image_metadata")
    __match_args__ = (*DrawingRecord.__match_args__, "data", "image_metadata")

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
        super().__init__(
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
        frozen_setattr(self, "data", data)
        frozen_setattr(self, "image_metadata", image_metadata)

    def __eq__(self, other: object) -> bool:
        equal = super().__eq__(other)
        if equal is not True or self is other:
            return equal
        assert isinstance(other, ImageRecord)
        return self.data == other.data and self.image_metadata == other.image_metadata

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


__all__ = ("DRAWING_FIELD_NAMES", "DrawingRecord", "ImageMetadata", "ImageRecord")
