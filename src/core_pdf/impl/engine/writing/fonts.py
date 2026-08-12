# SPDX-License-Identifier: AGPL-3.0-only
"""Font-resource contracts for semantic PDF writing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream

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
    def add_to_graph(self, graph: PdfObjectGraph, texts: Iterable[str]) -> PdfFontResource: ...


class StandardType1FontProvider:
    """Provide one of PDF's built-in Type1 fonts using WinAnsi encoding."""

    def __init__(self, font_name: str = "Helvetica") -> None:
        if font_name not in STANDARD_TYPE1_FONTS:
            raise ValueError(f"unsupported standard PDF font: {font_name!r}")
        self.font_name = font_name

    def add_to_graph(
        self,
        graph: PdfObjectGraph,
        texts: Iterable[str] = (),
    ) -> PdfFontResource:
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


class TrueTypeFontProvider:
    """Embed a TrueType font as a Unicode-capable Type0 font."""

    def __init__(
        self,
        font_data: bytes,
        *,
        resource_name: str = "F1",
        font_number: int = 0,
    ) -> None:
        if not font_data:
            raise ValueError("TrueType font data must not be empty")
        if font_number < 0:
            raise ValueError("font number must not be negative")
        self.font_data = font_data
        self.resource_name = resource_name
        self.font_number = font_number

    def add_to_graph(self, graph: PdfObjectGraph, texts: Iterable[str]) -> PdfFontResource:
        from io import BytesIO

        from core_pdf._vendor.fontTools.ttLib import TTFont

        font = TTFont(
            BytesIO(self.font_data),
            fontNumber=self.font_number,
            recalcBBoxes=False,
            recalcTimestamp=False,
        )
        cmap = font.getBestCmap() or {}
        codepoints = sorted({ord(char) for text in texts for char in text})
        missing = [codepoint for codepoint in codepoints if codepoint not in cmap]
        if missing:
            raise ValueError(f"font does not contain Unicode code points: {missing!r}")
        cid_by_codepoint = {codepoint: index + 1 for index, codepoint in enumerate(codepoints)}
        glyph_ids = {
            cid: font.getGlyphID(cmap[codepoint]) for codepoint, cid in cid_by_codepoint.items()
        }
        glyph_order = font.getGlyphOrder()
        units_per_em = max(1, int(cast(Any, font["head"]).unitsPerEm))
        hmetrics = font["hmtx"].metrics
        widths = [
            (cid, round(hmetrics[glyph_order[glyph_ids[cid]]][0] * 1000 / units_per_em))
            for cid in sorted(glyph_ids)
        ]
        font_name = internal_font_name(font)
        font_file = graph.add(
            PdfStream({PdfName.of("Length1"): len(self.font_data)}, self.font_data)
        )
        descriptor = graph.add(
            {
                PdfName.of("Type"): PdfName.of("FontDescriptor"),
                PdfName.of("FontName"): PdfName.of(font_name),
                PdfName.of("Flags"): 4,
                PdfName.of("FontBBox"): internal_font_bbox(font, units_per_em),
                PdfName.of("ItalicAngle"): 0,
                PdfName.of("Ascent"): internal_font_metric(font, "ascent", units_per_em),
                PdfName.of("Descent"): internal_font_metric(font, "descent", units_per_em),
                PdfName.of("CapHeight"): internal_font_cap_height(font, cmap, units_per_em),
                PdfName.of("StemV"): 80,
                PdfName.of("FontFile2"): font_file,
            }
        )
        cid_to_gid = bytearray((max(glyph_ids, default=0) + 1) * 2)
        for cid, glyph_id in glyph_ids.items():
            cid_to_gid[cid * 2 : cid * 2 + 2] = glyph_id.to_bytes(2, "big")
        cid_map = graph.add(PdfStream({}, bytes(cid_to_gid)))
        cid_font = graph.add(
            {
                PdfName.of("Type"): PdfName.of("Font"),
                PdfName.of("Subtype"): PdfName.of("CIDFontType2"),
                PdfName.of("BaseFont"): PdfName.of(font_name),
                PdfName.of("CIDSystemInfo"): {
                    PdfName.of("Registry"): "Adobe",
                    PdfName.of("Ordering"): "Identity",
                    PdfName.of("Supplement"): 0,
                },
                PdfName.of("FontDescriptor"): descriptor,
                PdfName.of("DW"): 1000,
                PdfName.of("W"): internal_widths_array(widths),
                PdfName.of("CIDToGIDMap"): cid_map,
            }
        )
        to_unicode = graph.add(PdfStream({}, internal_to_unicode_cmap(cid_by_codepoint)))
        type0 = graph.add(
            {
                PdfName.of("Type"): PdfName.of("Font"),
                PdfName.of("Subtype"): PdfName.of("Type0"),
                PdfName.of("BaseFont"): PdfName.of(font_name),
                PdfName.of("Encoding"): PdfName.of("Identity-H"),
                PdfName.of("DescendantFonts"): [cid_font],
                PdfName.of("ToUnicode"): to_unicode,
            }
        )
        return PdfFontResource(
            self.resource_name,
            type0,
            lambda text: b"".join(cid_by_codepoint[ord(char)].to_bytes(2, "big") for char in text),
        )


def internal_font_name(font: Any) -> str:
    return "CoreTTFont"


def internal_font_bbox(font: Any, units_per_em: int) -> list[int]:
    head = font["head"]
    return [
        round(value * 1000 / units_per_em) for value in (head.xMin, head.yMin, head.xMax, head.yMax)
    ]


def internal_font_metric(font: Any, name: str, units_per_em: int) -> int:
    value = getattr(font["hhea"], name, 0)
    return round(value * 1000 / units_per_em)


def internal_font_cap_height(font: Any, cmap: dict[int, str], units_per_em: int) -> int:
    """Derive cap height from the "H" glyph's top edge (no OS/2 table is vendored)."""
    glyph_name = cmap.get(ord("H"))
    if glyph_name is None or "glyf" not in font:
        return internal_font_metric(font, "ascent", units_per_em)
    y_max = getattr(font["glyf"][glyph_name], "yMax", None)
    if y_max is None:
        return internal_font_metric(font, "ascent", units_per_em)
    return round(y_max * 1000 / units_per_em)


def internal_widths_array(widths: list[tuple[int, int]]) -> list[object]:
    return [item for cid, width in widths for item in (cid, [width])]


def internal_to_unicode_cmap(cid_by_codepoint: dict[int, int]) -> bytes:
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /CoreUnicode def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000><FFFF>",
        "endcodespacerange",
        f"{len(cid_by_codepoint)} beginbfchar",
    ]
    lines.extend(
        f"<{cid:04X}><{chr(codepoint).encode('utf-16-be').hex().upper()}>"
        for codepoint, cid in sorted(cid_by_codepoint.items(), key=lambda item: item[1])
    )
    lines.extend(
        [
            "endbfchar",
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return "\n".join(lines).encode("ascii")


__all__ = (
    "PdfFontProvider",
    "PdfFontResource",
    "STANDARD_TYPE1_FONTS",
    "StandardType1FontProvider",
    "TrueTypeFontProvider",
)
