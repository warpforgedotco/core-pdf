# SPDX-License-Identifier: AGPL-3.0-only
"""Write the core-document IR as a basic, standards-compliant PDF."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from core_document import Document, Page, TextLine

from core_pdf.impl.engine.writing.document import serialize_encrypted_pdf_file
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.fonts import (
    PdfFontProvider,
    PdfFontResource,
    StandardType1FontProvider,
)
from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.engine.writing.objects import serialize_pdf_string
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream


def serialize_document_to_pdf(
    document: Document,
    *,
    font_name: str = "Helvetica",
    font_provider: PdfFontProvider | None = None,
    encryption: StandardPdfEncryption | None = None,
    version: str = "1.7",
) -> bytes:
    """Serialize pages and their extracted text into a new PDF file."""
    graph = PdfObjectGraph()
    pages_reference = graph.add(None)
    font = font_provider or StandardType1FontProvider(font_name)
    page_lines = tuple(tuple(_page_lines(page)) for page in document.pages)
    font_resource = font.add_to_graph(
        graph,
        (line.text for lines in page_lines for line in lines),
    )
    page_references: list[PdfReference] = []
    for page, lines in zip(document.pages, page_lines, strict=True):
        content = content_stream_for_page(page, font_resource, lines)
        content_reference = graph.add(PdfStream({}, content))
        page_references.append(
            graph.add(
                {
                    PdfName.of("Type"): PdfName.of("Page"),
                    PdfName.of("Parent"): pages_reference,
                    PdfName.of("MediaBox"): [0, 0, page.width or 612.0, page.height or 792.0],
                    PdfName.of("Resources"): {
                        PdfName.of("Font"): {
                            PdfName.of(font_resource.resource_name): font_resource.reference
                        },
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
    trailer: dict[object, object] = {PdfName.of("Root"): catalog_reference}
    if encryption is None:
        return graph.to_pdf(trailer=trailer, version=version)
    file_id = sha256(document.to_json(sort_keys=True).encode("utf-8")).digest()[:16]
    return serialize_encrypted_pdf_file(
        graph.objects,
        trailer=trailer,
        encryption=encryption,
        file_id=file_id,
        version=version,
    )


def content_stream_for_page(
    page: Page,
    font: PdfFontResource | None = None,
    lines: Iterable[TextLine] | None = None,
) -> bytes:
    font = font or StandardType1FontProvider().add_to_graph(PdfObjectGraph(), ())
    commands: list[bytes] = []
    for line in lines or _page_lines(page):
        text = line.text.replace("\n", " ")
        encoded = font.encode_text(text)
        x, y = _line_position(page, line)
        font_size = _line_font_size(line)
        commands.extend(
            (
                b"BT\n",
                f"/{font.resource_name} {_number(font_size)} Tf\n".encode("ascii"),
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
