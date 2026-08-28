# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import mmap
from os import PathLike
from typing import TypeAlias

from core_pdf.impl.protocols import BinaryReader, SeekableBinaryReader

PdfByteBuffer: TypeAlias = bytes | mmap.mmap
PathSource: TypeAlias = str | PathLike[str]
PdfSource: TypeAlias = (
    PathSource | bytes | bytearray | memoryview | BinaryReader | SeekableBinaryReader
)
Point: TypeAlias = tuple[float, float]
Rectangle: TypeAlias = tuple[float, float, float, float]

__all__ = (
    "BinaryReader",
    "PathSource",
    "PdfByteBuffer",
    "PdfSource",
    "Point",
    "Rectangle",
    "SeekableBinaryReader",
)
