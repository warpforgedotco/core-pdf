# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias

from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import (
    MissingObject,
    PdfName,
    PdfReference,
    PdfString,
)

PdfObjectKey: TypeAlias = str | bytes | PdfName
CachedPdfScalar: TypeAlias = (
    str
    | bytes
    | bytearray
    | memoryview
    | int
    | float
    | bool
    | None
    | PdfName
    | PdfReference
    | PdfStream
    | PdfString
)
CachedPdfArray: TypeAlias = list["CachedPdfObject"]
CachedPdfTuple: TypeAlias = tuple["CachedPdfObject", ...]
CachedPdfDict: TypeAlias = dict[PdfObjectKey, "CachedPdfObject"]
CachedPdfObject: TypeAlias = CachedPdfScalar | CachedPdfArray | CachedPdfTuple | CachedPdfDict

ObjectGenerationKey: TypeAlias = tuple[int, int]
ObjectCache: TypeAlias = dict[int, CachedPdfObject]
DeepObjectCache: TypeAlias = dict[int, CachedPdfObject]
ResolvedObjectCache: TypeAlias = dict[ObjectGenerationKey, CachedPdfObject]
GenerationZeroObjectCache: TypeAlias = list[CachedPdfObject | MissingObject]
InheritedValueMap: TypeAlias = dict[str, CachedPdfObject]
InheritedValuesCache: TypeAlias = dict[int, InheritedValueMap]


__all__ = (
    "CachedPdfArray",
    "CachedPdfDict",
    "CachedPdfObject",
    "CachedPdfScalar",
    "CachedPdfTuple",
    "DeepObjectCache",
    "GenerationZeroObjectCache",
    "InheritedValueMap",
    "InheritedValuesCache",
    "ObjectCache",
    "ObjectGenerationKey",
    "PdfObjectKey",
    "ResolvedObjectCache",
)
