# SPDX-License-Identifier: AGPL-3.0-only
"""PDF primitives, extraction records, and shared boundary types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from os import PathLike
from typing import Generic, Protocol, Self, TypeAlias, TypeVar

from core_pdf_spec.types import (
    MISSING,
    MissingObject,
    PdfByteBuffer,
    PdfName,
    PdfReference,
    PdfString,
    Rectangle,
)


class BinaryReader(Protocol):
    """Readable binary source that can be materialized before parsing."""

    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


class SeekableBinaryReader(BinaryReader, Protocol):
    """Random-access binary source aligned with PDF byte-offset structure."""

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def fileno(self) -> int: ...


PathSource: TypeAlias = str | PathLike[str]
PdfSource: TypeAlias = (
    PathSource | bytes | bytearray | memoryview | BinaryReader | SeekableBinaryReader
)


RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class PageScoped(Generic[RecordT]):
    """A page-level extraction record with its document-level context."""

    page_index: int
    page_number: int
    page_label: str | None
    record: RecordT


@dataclass(frozen=True, slots=True)
class TextWord:
    """One canonical word record shared by layout and structured extraction."""

    text: str
    bbox: Rectangle | None = None
    line_index: int = 0
    word_index: int = 0
    block_index: int = 0
    page_number: int | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class DrawingRecord:
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
        """Build a record from an object exposing the same fields."""
        values = {name: getattr(source, name) for name in internal_DRAWING_FIELD_NAMES}
        values.update(overrides)
        return cls(**values)


internal_DRAWING_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(DrawingRecord))


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    channels: int
    color_model: str
    alpha: bool
    stride: int
    source_rect: Rectangle
    transform: object | None
    clipping: Rectangle | None


@dataclass(frozen=True, slots=True)
class ImageRecord(DrawingRecord):
    data: object | None = None
    image_metadata: ImageMetadata | None = None


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
