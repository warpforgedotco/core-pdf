# SPDX-License-Identifier: AGPL-3.0-only
"""Recursive type vocabulary for PDF object values."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from core_pdf.impl.engine.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString

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
PdfPrimitive: TypeAlias = PdfNull | PdfBoolean | PdfInteger | PdfReal | PdfStringObject | PdfName
PdfDirectObject: TypeAlias = PdfPrimitive | PdfArray | PdfDict | PdfStream
PdfObject: TypeAlias = PdfDirectObject | PdfReference
Decipher: TypeAlias = Callable[[int, int, bytes, PdfDict | None], bytes | memoryview]

__all__ = (
    "Decipher",
    "PdfArray",
    "PdfBoolean",
    "PdfByteString",
    "PdfDict",
    "PdfDirectObject",
    "PdfInteger",
    "PdfKey",
    "PdfNull",
    "PdfNumber",
    "PdfObject",
    "PdfPrimitive",
    "PdfReal",
    "PdfStringObject",
    "PdfTextString",
)
