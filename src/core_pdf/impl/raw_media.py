# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Self

from core_pdf.impl.types import GeneratedRecord, Rectangle, frozen_setattr


class DrawingRecord(GeneratedRecord):
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

    @classmethod
    def from_captured(cls, source: object, **overrides: object) -> Self:
        values = {name: getattr(source, name) for name in DRAWING_FIELD_NAMES}
        values.update(overrides)
        return cls(**values)


DRAWING_FIELD_NAMES: tuple[str, ...] = DrawingRecord.__fields__


class ImageMetadata(GeneratedRecord):
    width: int
    height: int
    channels: int
    color_model: str
    alpha: bool
    stride: int
    source_rect: Rectangle
    transform: object | None
    clipping: Rectangle | None


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
