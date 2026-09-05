# SPDX-License-Identifier: AGPL-3.0-only
"""Recursive type vocabulary for PDF object values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeAlias

from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.types import PdfName, PdfReference, PdfString

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

CachedPdfObject: TypeAlias = (
    PdfPrimitive
    | str
    | PdfReference
    | PdfStream
    | list["CachedPdfObject"]
    | tuple["CachedPdfObject", ...]
    | dict[PdfKey, "CachedPdfObject"]
)

ObjectCache: TypeAlias = dict[int, CachedPdfObject]
ResolvedObjectCache: TypeAlias = dict[tuple[int, int], CachedPdfObject]
InheritedValueMap: TypeAlias = dict[str, CachedPdfObject]


class PdfValueResolver(Protocol):
    """The resolution operations needed by higher-level PDF features."""

    def resolve(self, ref: object) -> object: ...

    def deep_resolve(self, value: object, seen: set[int] | None = None) -> object: ...

    def resolve_dict(self, value: object) -> PdfDict | None: ...

    def resolve_box(self, value: object) -> tuple[float, float, float, float] | None: ...

    def resolve_font_dict(self, font: PdfDict) -> PdfDict: ...

    def resolve_float(self, value: object, default: float | None = 0.0) -> float | None: ...

    def resolve_name(self, value: object) -> str | None: ...

    def resolve_name_like_value(self, resolved: object) -> str | None: ...

    def resolve_name_or_text(self, value: object, *, name_like: bool = False) -> str | None: ...

    def resolve_int(self, value: object, default: int | None = None) -> int | None: ...

    def resolve_str(self, value: object) -> str | None: ...


__all__ = (
    "CachedPdfObject",
    "Decipher",
    "InheritedValueMap",
    "ObjectCache",
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
    "PdfValueResolver",
    "ResolvedObjectCache",
)
