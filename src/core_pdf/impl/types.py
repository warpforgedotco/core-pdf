# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import mmap
from os import PathLike
from typing import Protocol, TypeAlias


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

__all__ = (
    "BinaryReader",
    "PathSource",
    "PdfByteBuffer",
    "PdfSource",
    "Rectangle",
    "SeekableBinaryReader",
)
