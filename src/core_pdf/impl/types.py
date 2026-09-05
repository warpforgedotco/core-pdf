# SPDX-License-Identifier: AGPL-3.0-only
"""PDF primitives, extraction records, and shared boundary types."""

from __future__ import annotations

import mmap
from collections.abc import Mapping
from dataclasses import dataclass, fields
from os import PathLike
from typing import Final, Generic, Protocol, Self, TypeAlias, TypeVar


class BinaryReader(Protocol):
    """Readable binary source that can be materialized before parsing."""

    def read(self, size: int = -1, /) -> bytes | bytearray | memoryview: ...


class SeekableBinaryReader(BinaryReader, Protocol):
    """Random-access binary source aligned with PDF byte-offset structure."""

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def tell(self) -> int: ...

    def fileno(self) -> int: ...


PdfByteBuffer: TypeAlias = bytes | mmap.mmap
PathSource: TypeAlias = str | PathLike[str]
PdfSource: TypeAlias = (
    PathSource | bytes | bytearray | memoryview | BinaryReader | SeekableBinaryReader
)
Rectangle: TypeAlias = tuple[float, float, float, float]


class MissingObject:
    __slots__ = ()


MISSING: Final = MissingObject()


class PdfName:
    """PDF name object.

    Names are atomic identifiers in PDF syntax. This implementation stores the
    decoded Latin-1 value for fast dictionary lookup while preserving equality
    with raw bytes and strings used by recovery paths.
    """

    __slots__ = ("value_bytes", "str_value")

    value_bytes: bytes
    str_value: str

    def __init__(self, value_bytes: bytes) -> None:
        object.__setattr__(self, "value_bytes", value_bytes)
        object.__setattr__(self, "str_value", value_bytes.decode("latin-1"))

    @property
    def value(self) -> str:
        return self.str_value

    @classmethod
    def of(cls, value: str | bytes | memoryview | "PdfName") -> "PdfName":
        if type(value) is PdfName:
            return value
        if type(value) is str:
            key_bytes = value.encode("latin-1")
        elif type(value) is memoryview:
            key_bytes = value.tobytes()
        elif type(value) is bytes:
            key_bytes = value
        else:
            raise TypeError("PDF names must be str, bytes, memoryview, or PdfName")
        return cls(key_bytes)

    def __str__(self) -> str:
        return self.str_value

    def __repr__(self) -> str:
        return f"PdfName({self.value!r})"

    def __hash__(self) -> int:
        return hash(self.str_value)

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfName:
            return self.str_value == other.str_value
        if type(other) is bytes:
            return self.value_bytes == other
        if type(other) is str:
            return self.str_value == other
        return False

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")


class PdfReference:
    """Indirect object reference: object number plus generation number."""

    __slots__ = ("object_number", "generation_number")

    object_number: int
    generation_number: int

    def __init__(self, object_number: int, generation_number: int = 0) -> None:
        if object_number < 0 or generation_number < 0:
            raise ValueError("invalid PDF reference")
        self.object_number = object_number
        self.generation_number = generation_number

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfReference:
            return (
                self.object_number == other.object_number
                and self.generation_number == other.generation_number
            )
        return False

    def __hash__(self) -> int:
        return hash((self.object_number, self.generation_number))

    def __str__(self) -> str:
        return f"{self.object_number} {self.generation_number} R"

    def __repr__(self) -> str:
        return f"PdfReference({self.object_number}, {self.generation_number})"


class PdfString:
    """PDF string object containing the raw byte representation."""

    __slots__ = ("data", "is_literal")

    data: bytes
    is_literal: bool | None

    def __init__(
        self,
        data: bytes,
        *,
        is_literal: bool | None = None,
    ) -> None:
        if not isinstance(data, bytes):
            raise ValueError("invalid PDF string")
        self.data = data
        self.is_literal = is_literal

    def __eq__(self, other: object) -> bool:
        if type(other) is PdfString:
            return self.data == other.data
        return False

    def __hash__(self) -> int:
        return hash(self.data)

    def __repr__(self) -> str:
        return f"PdfString({self.data!r})"


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
