# Ported from pypdf.types, Copyright (c) 2006, Mathieu Fenniak, under its BSD-style license.
"""Helpers for working with PDF types."""

from typing import Literal, TypeAlias

from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    Destination,
    NameObject,
    NullObject,
    NumberObject,
    OutlineItem,
)

BorderArrayType: TypeAlias = list[NameObject | NumberObject | ArrayObject]

OutlineItemType: TypeAlias = OutlineItem | Destination

FitType: TypeAlias = Literal["/XYZ", "/Fit", "/FitH", "/FitV", "/FitR", "/FitB", "/FitBH", "/FitBV"]
# These go with the FitType, they specify values for the fit
ZoomArgType: TypeAlias = NumberObject | NullObject | float
ZoomArgsType: TypeAlias = list[ZoomArgType]

OutlineType: TypeAlias = list[Destination | list[Destination | list[Destination]]]

LayoutType: TypeAlias = Literal[
    "/NoLayout",
    "/SinglePage",
    "/OneColumn",
    "/TwoColumnLeft",
    "/TwoColumnRight",
    "/TwoPageLeft",
    "/TwoPageRight",
]

PagemodeType: TypeAlias = Literal[
    "/UseNone",
    "/UseOutlines",
    "/UseThumbs",
    "/FullScreen",
    "/UseOC",
    "/UseAttachments",
]

AnnotationSubtype: TypeAlias = Literal[
    "/Text",
    "/Link",
    "/FreeText",
    "/Line",
    "/Square",
    "/Circle",
    "/Polygon",
    "/PolyLine",
    "/Highlight",
    "/Underline",
    "/Squiggly",
    "/StrikeOut",
    "/Caret",
    "/Stamp",
    "/Ink",
    "/Popup",
    "/FileAttachment",
    "/Sound",
    "/Movie",
    "/Screen",
    "/Widget",
    "/PrinterMark",
    "/TrapNet",
    "/Watermark",
    "/3D",
    "/Redact",
    "/Projection",
    "/RichMedia",
]
