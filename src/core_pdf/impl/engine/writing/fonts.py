# SPDX-License-Identifier: AGPL-3.0-only
"""Font-resource contracts for semantic PDF writing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.objects import PdfName, PdfReference

STANDARD_TYPE1_FONTS = frozenset(
    {"Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique", "Helvetica"}
    | {"Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"}
    | {"Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic", "Symbol", "ZapfDingbats"}
)


@dataclass(frozen=True, slots=True)
class PdfFontResource:
    resource_name: str
    reference: PdfReference
    encode_text: Callable[[str], bytes]


class PdfFontProvider(Protocol):
    def add_to_graph(self, graph: PdfObjectGraph) -> PdfFontResource: ...


class StandardType1FontProvider:
    """Provide one of PDF's built-in Type1 fonts using WinAnsi encoding."""

    def __init__(self, font_name: str = "Helvetica") -> None:
        if font_name not in STANDARD_TYPE1_FONTS:
            raise ValueError(f"unsupported standard PDF font: {font_name!r}")
        self.font_name = font_name

    def add_to_graph(self, graph: PdfObjectGraph) -> PdfFontResource:
        reference = graph.add(
            {
                PdfName.of("Type"): PdfName.of("Font"),
                PdfName.of("Subtype"): PdfName.of("Type1"),
                PdfName.of("BaseFont"): PdfName.of(self.font_name),
                PdfName.of("Encoding"): PdfName.of("WinAnsiEncoding"),
            }
        )
        return PdfFontResource("F1", reference, self.encode_text)

    @staticmethod
    def encode_text(text: str) -> bytes:
        return text.encode("cp1252")


__all__ = (
    "PdfFontProvider",
    "PdfFontResource",
    "STANDARD_TYPE1_FONTS",
    "StandardType1FontProvider",
)
