from io import BytesIO

from core_pdf import (
    PdfDocument,
    StandardPdfEncryption,
    serialize_document_to_pdf,
)
from core_pdf.impl.engine.structured import Block, BlockKind, Document, Page, TextLine
from core_pdf.impl.engine.writing import append_incremental_update, serialize_pdf_file
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString


def test_append_incremental_update_preserves_and_reopens_original_file() -> None:
    original = serialize_pdf_file(
        {
            1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
            2: {
                PdfName.of("Type"): PdfName.of("Pages"),
                PdfName.of("Kids"): [PdfReference(3)],
                PdfName.of("Count"): 1,
            },
            3: {
                PdfName.of("Type"): PdfName.of("Page"),
                PdfName.of("Parent"): PdfReference(2),
                PdfName.of("MediaBox"): [0, 0, 10, 10],
                PdfName.of("Contents"): PdfReference(4),
            },
            4: PdfStream({}, b"q\nQ\n"),
        },
        trailer={PdfName.of("Root"): PdfReference(1)},
    )
    previous_xref_offset = original.rfind(b"xref\n")

    updated = append_incremental_update(
        original,
        {4: PdfStream({}, b"q\n0 0 1 rg\nQ\n")},
        trailer={PdfName.of("Root"): PdfReference(1)},
        previous_xref_offset=previous_xref_offset,
        previous_size=5,
    )

    assert updated.startswith(original)
    assert updated.count(b"%%EOF") == 2
    assert b"/Prev " + str(previous_xref_offset).encode("ascii") in updated
    with PdfDocument.open(updated) as document:
        assert len(document.pages) == 1


def test_pdf_document_save_incremental_writes_to_binary_target() -> None:
    original = serialize_pdf_file(
        {
            1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
            2: {
                PdfName.of("Type"): PdfName.of("Pages"),
                PdfName.of("Kids"): [PdfReference(3)],
                PdfName.of("Count"): 1,
            },
            3: {
                PdfName.of("Type"): PdfName.of("Page"),
                PdfName.of("Parent"): PdfReference(2),
                PdfName.of("MediaBox"): [0, 0, 10, 10],
                PdfName.of("Contents"): PdfReference(4),
            },
            4: PdfStream({}, b"q\nQ\n"),
        },
        trailer={PdfName.of("Root"): PdfReference(1)},
    )
    target = BytesIO()

    with PdfDocument.open(original) as document:
        updated = document.save_incremental(
            target,
            {4: PdfStream({}, b"q\nQ\n")},
        )

    assert target.getvalue() == updated
    assert updated.startswith(original)


def test_pdf_document_encrypts_incremental_objects_for_encrypted_input() -> None:
    encrypted = serialize_document_to_pdf(
        Document(
            pages=(
                Page(
                    page_number=1,
                    blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("secret"),)),),
                ),
            )
        ),
        encryption=StandardPdfEncryption(user_password="open"),
    )

    with PdfDocument.open(encrypted, password="open") as document:
        updated = document.save_incremental(
            BytesIO(),
            {99: PdfString(b"incremental secret")},
        )

    assert b"incremental secret" not in updated
    with PdfDocument.open(updated, password="open") as document:
        assert len(document.pages) == 1
