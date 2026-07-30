# SPDX-License-Identifier: AGPL-3.0-only
"""Write the core-document IR as a basic, standards-compliant PDF."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from core_pdf.impl.engine.structured import Document, Page, TextLine
from core_pdf.impl.engine.writing.document import serialize_encrypted_pdf_file
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.fonts import (
    PdfFontProvider,
    PdfFontResource,
    StandardType1FontProvider,
)
from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.engine.writing.objects import serialize_pdf_string
from core_pdf.impl.engine.writing.signatures import (
    PdfSignaturePlan,
    apply_signature_plan,
)
from core_pdf.impl.objects import PdfName, PdfReference, PdfStream


def serialize_document_to_pdf(
    document: Document,
    *,
    font_name: str = "Helvetica",
    font_provider: PdfFontProvider | None = None,
    encryption: StandardPdfEncryption | None = None,
    signature: PdfSignaturePlan | None = None,
    version: str = "1.7",
) -> bytes:
    """Serialize pages and their extracted text into a new PDF file."""
    if signature is not None and encryption is not None:
        raise ValueError("PDF encryption and signature containers cannot be combined")
    if signature is not None and not document.pages:
        raise ValueError("a signed PDF requires at least one page")
    graph = PdfObjectGraph()
    pages_reference = graph.add(None)
    font = font_provider or StandardType1FontProvider(font_name)
    page_lines = tuple(tuple(internal_page_lines(page)) for page in document.pages)
    font_resource = font.add_to_graph(
        graph,
        (line.text for lines in page_lines for line in lines),
    )
    page_references: list[PdfReference] = []
    page_objects: list[tuple[PdfReference, dict[PdfName, object]]] = []
    for page, lines in zip(document.pages, page_lines, strict=True):
        content = content_stream_for_page(page, font_resource, lines)
        content_reference = graph.add(PdfStream({}, content))
        page_object: dict[PdfName, object] = {
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
        page_reference = graph.add(page_object)
        page_references.append(page_reference)
        page_objects.append((page_reference, page_object))
    signature_field: PdfReference | None = None
    if signature is not None:
        signature_dictionary = graph.add(
            {
                PdfName.of("Type"): PdfName.of("Sig"),
                PdfName.of("Filter"): PdfName.of("Adobe.PPKLite"),
                PdfName.of("SubFilter"): PdfName.of("adbe.pkcs7.detached"),
                PdfName.of("ByteRange"): signature.byte_range_placeholder,
                PdfName.of("Contents"): signature.contents_placeholder,
            }
        )
        signature_field = graph.add(
            {
                PdfName.of("Type"): PdfName.of("Annot"),
                PdfName.of("Subtype"): PdfName.of("Widget"),
                PdfName.of("FT"): PdfName.of("Sig"),
                PdfName.of("Rect"): [0, 0, 0, 0],
                PdfName.of("T"): "Signature1",
                PdfName.of("V"): signature_dictionary,
                PdfName.of("F"): 4,
            }
        )
        first_page_reference, first_page = page_objects[0]
        graph.replace(
            first_page_reference,
            {**first_page, PdfName.of("Annots"): [signature_field]},
        )
    graph.replace(
        pages_reference,
        {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): page_references,
            PdfName.of("Count"): len(page_references),
        },
    )
    catalog = {
        PdfName.of("Type"): PdfName.of("Catalog"),
        PdfName.of("Pages"): pages_reference,
    }
    if signature_field is not None:
        catalog[PdfName.of("AcroForm")] = {
            PdfName.of("SigFlags"): 3,
            PdfName.of("Fields"): [signature_field],
        }
    catalog_reference = graph.add(catalog)
    trailer: dict[object, object] = {PdfName.of("Root"): catalog_reference}
    if encryption is None:
        output = graph.to_pdf(trailer=trailer, version=version)
        return apply_signature_plan(output, signature) if signature is not None else output
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
    for line in lines or internal_page_lines(page):
        text = line.text.replace("\n", " ")
        encoded = font.encode_text(text)
        x, y = internal_line_position(page, line)
        font_size = internal_line_font_size(line)
        commands.extend(
            (
                b"BT\n",
                f"/{font.resource_name} {internal_number(font_size)} Tf\n".encode("ascii"),
                f"1 0 0 1 {internal_number(x)} {internal_number(y)} Tm\n".encode("ascii"),
                serialize_pdf_string(encoded) + b" Tj\nET\n",
            )
        )
    return b"".join(commands)


def internal_page_lines(page: Page) -> Iterable[TextLine]:
    for block in page.blocks:
        yield from block.lines
    for table in page.tables:
        for row in table.rows:
            yield TextLine(" | ".join(cell.text for cell in row))


def internal_line_position(page: Page, line: TextLine) -> tuple[float, float]:
    if line.bbox is not None:
        return line.bbox[0], line.bbox[1]
    return 36.0, max(36.0, (page.height or 792.0) - 36.0)


def internal_line_font_size(line: TextLine) -> float:
    if line.bbox is None:
        return 12.0
    return max(1.0, line.bbox[3] - line.bbox[1])


def internal_number(value: float) -> str:
    return format(value, ".4g")


__all__ = ("content_stream_for_page", "serialize_document_to_pdf")
