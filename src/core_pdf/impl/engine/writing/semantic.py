# SPDX-License-Identifier: AGPL-3.0-only
"""Write the core-document IR as a basic, standards-compliant PDF."""

from __future__ import annotations

from collections.abc import Iterable

from core_document import Document, Page, TextLine

from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.engine.writing.objects import serialize_pdf_string
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream

STANDARD_TYPE1_FONTS = frozenset(
    {"Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique", "Helvetica"}
    | {"Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"}
    | {"Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic", "Symbol", "ZapfDingbats"}
)


def serialize_document_to_pdf(
    document: Document,
    *,
    font_name: str = "Helvetica",
    version: str = "1.7",
) -> bytes:
    """Serialize pages and their extracted text into a new PDF file."""
    if font_name not in STANDARD_TYPE1_FONTS:
        raise ValueError(f"unsupported standard PDF font: {font_name!r}")

    graph = PdfObjectGraph()
    pages_reference = graph.add(None)
    font_reference = graph.add(
        {
            PdfName.of("Type"): PdfName.of("Font"),
            PdfName.of("Subtype"): PdfName.of("Type1"),
            PdfName.of("BaseFont"): PdfName.of(font_name),
            PdfName.of("Encoding"): PdfName.of("WinAnsiEncoding"),
        }
    )
    page_references: list[PdfReference] = []
    for page in document.pages:
        content = content_stream_for_page(page)
        content_reference = graph.add(PdfStream({}, content))
        page_references.append(
            graph.add(
                {
                    PdfName.of("Type"): PdfName.of("Page"),
                    PdfName.of("Parent"): pages_reference,
                    PdfName.of("MediaBox"): [0, 0, page.width or 612.0, page.height or 792.0],
                    PdfName.of("Resources"): {
                        PdfName.of("Font"): {PdfName.of("F1"): font_reference},
                    },
                    PdfName.of("Contents"): content_reference,
                }
            )
        )
    graph.replace(
        pages_reference,
        {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): page_references,
            PdfName.of("Count"): len(page_references),
        },
    )
    catalog_reference = graph.add(
        {
            PdfName.of("Type"): PdfName.of("Catalog"),
            PdfName.of("Pages"): pages_reference,
        }
    )
    return graph.to_pdf(
        trailer={PdfName.of("Root"): catalog_reference},
        version=version,
    )


def content_stream_for_page(page: Page) -> bytes:
    commands: list[bytes] = []
    for line in _page_lines(page):
        text = line.text.replace("\n", " ")
        encoded = text.encode("cp1252")
        x, y = _line_position(page, line)
        font_size = _line_font_size(line)
        commands.extend(
            (
                b"BT\n",
                f"/F1 {_number(font_size)} Tf\n".encode("ascii"),
                f"1 0 0 1 {_number(x)} {_number(y)} Tm\n".encode("ascii"),
                serialize_pdf_string(encoded) + b" Tj\nET\n",
            )
        )
    return b"".join(commands)


def _page_lines(page: Page) -> Iterable[TextLine]:
    for block in page.blocks:
        yield from block.lines
    for table in page.tables:
        for row in table.rows:
            yield TextLine(" | ".join(cell.text for cell in row))


def _line_position(page: Page, line: TextLine) -> tuple[float, float]:
    if line.bbox is not None:
        return line.bbox[0], line.bbox[1]
    return 36.0, max(36.0, (page.height or 792.0) - 36.0)


def _line_font_size(line: TextLine) -> float:
    if line.bbox is None:
        return 12.0
    return max(1.0, line.bbox[3] - line.bbox[1])


def _number(value: float) -> str:
    return format(value, ".4g")


__all__ = ("content_stream_for_page", "serialize_document_to_pdf")
