# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import mmap
from collections.abc import Callable, Sequence
from os import PathLike
from typing import TypeAlias

from core_pdf.impl.objects import PdfName, PdfReference, PdfStream, PdfString
from core_pdf.impl.protocols import BinaryReader, SeekableBinaryReader

PdfNull: TypeAlias = None
PdfBoolean: TypeAlias = bool
PdfInteger: TypeAlias = int
PdfReal: TypeAlias = float
PdfNumber: TypeAlias = PdfInteger | PdfReal
PdfTextString: TypeAlias = PdfString
PdfByteString: TypeAlias = bytes | bytearray | memoryview
PdfStringObject: TypeAlias = PdfTextString | PdfByteString
PdfKey: TypeAlias = str | bytes | PdfName
PdfArray: TypeAlias = list["PdfObject"]
PdfDict: TypeAlias = dict[PdfKey, "PdfObject"]
PdfPrimitive: TypeAlias = (
    PdfNull | PdfBoolean | PdfInteger | PdfReal | PdfStringObject | PdfName
)
PdfDirectObject: TypeAlias = PdfPrimitive | PdfArray | PdfDict | PdfStream
PdfObject: TypeAlias = PdfDirectObject | PdfReference
PdfByteBuffer: TypeAlias = bytes | mmap.mmap
Decipher: TypeAlias = Callable[[int, int, bytes, PdfDict | None], bytes | memoryview]
PathSource: TypeAlias = str | PathLike[str]
PdfSource: TypeAlias = (
    PathSource | bytes | bytearray | memoryview | BinaryReader | SeekableBinaryReader
)
Point: TypeAlias = tuple[float, float]
Rectangle: TypeAlias = tuple[float, float, float, float]
BBox: TypeAlias = Rectangle
PageSelection: TypeAlias = int | str | range | Sequence[int]

__all__ = (
    "BinaryReader",
    "BBox",
    "Decipher",
    "PageSelection",
    "PathSource",
    "PdfArray",
    "PdfBoolean",
    "PdfByteBuffer",
    "PdfByteString",
    "PdfDirectObject",
    "PdfDict",
    "PdfInteger",
    "PdfKey",
    "PdfNull",
    "PdfNumber",
    "PdfObject",
    "PdfPrimitive",
    "PdfReal",
    "PdfSource",
    "PdfStringObject",
    "PdfTextString",
    "Point",
    "Rectangle",
    "SeekableBinaryReader",
)
