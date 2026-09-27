# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from os import PathLike
from typing import Protocol, TypeAlias, TypeVar

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
from core_records import GeneratedRecord as GeneratedRecord
from core_records import PickleFields as PickleFields
from core_records import Record as Record
from core_records import RecordType as RecordType
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


class PageScoped[RecordT](GeneratedRecord):
    page_index: int
    page_number: int
    page_label: str | None
    record: RecordT

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


__all__ = (
    "BinaryReader",
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
)
