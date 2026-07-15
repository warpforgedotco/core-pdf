# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias, TypedDict

from core_pdf.impl.objects import PdfName, PdfReference, PdfStream, PdfString

MetadataScalar: TypeAlias = (
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
MetadataList: TypeAlias = list["MetadataValue"]
MetadataMap: TypeAlias = dict[str, "MetadataValue"]
MetadataValue: TypeAlias = MetadataScalar | MetadataList | MetadataMap
InfoMetadataRecord: TypeAlias = dict[str, MetadataValue]


class XmpNodeRecord(TypedDict, total=False):
    tag: str
    attributes: dict[str, str]
    text: str
    children: list["XmpNodeRecord"]
    parse_error: str


class MetadataRecord(TypedDict):
    info: InfoMetadataRecord
    xmp: XmpNodeRecord | None


__all__ = (
    "InfoMetadataRecord",
    "MetadataList",
    "MetadataMap",
    "MetadataRecord",
    "MetadataScalar",
    "MetadataValue",
    "XmpNodeRecord",
)
