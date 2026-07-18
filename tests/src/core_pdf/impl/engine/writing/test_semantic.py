import pytest
from core_document import Block, BlockKind, Document, Page, Table, TableCell, TextLine

from core_pdf import PdfDocument
from core_pdf import serialize_document_to_pdf as public_writer
from core_pdf.impl.engine.writing import (
    StandardPdfEncryption,
    StandardType1FontProvider,
    TrueTypeFontProvider,
    serialize_document_to_pdf,
)


def test_semantic_writer_round_trips_text_geometry_and_tables() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                width=300.0,
                height=400.0,
                blocks=(
                    Block(
                        1,
                        BlockKind.HEADING,
                        lines=(TextLine("Hello", bbox=(24.0, 320.0, 80.0, 336.0)),),
                        level=1,
                    ),
                ),
                tables=(
                    Table(
                        1,
                        rows=((TableCell(0, 0, "Name"), TableCell(0, 1, "Value")),),
                    ),
                ),
            ),
        )
    )

    pdf = serialize_document_to_pdf(document)

    with PdfDocument.open(pdf) as parsed:
        assert len(parsed.pages) == 1
        extracted = parsed.extract()
    assert "Hello" in extracted.text
    assert "Name" in extracted.text
    assert extracted.pages[0].width == 300.0


def test_semantic_writer_rejects_text_outside_standard_encoding() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("你好"),)),),
            ),
        )
    )

    with pytest.raises(UnicodeEncodeError):
        serialize_document_to_pdf(document)


def test_semantic_writer_accepts_a_font_provider() -> None:
    document = Document(pages=(Page(page_number=1),))

    output = serialize_document_to_pdf(
        document,
        font_provider=StandardType1FontProvider("Times-Roman"),
    )

    assert b"/BaseFont /Times-Roman" in output


def test_true_type_font_provider_embeds_a_unicode_font() -> None:
    from pathlib import Path

    font_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if not font_path.exists():
        return
    document = Document(
        pages=(Page(page_number=1, blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("你好"),)),)),)
    )

    output = serialize_document_to_pdf(
        document,
        font_provider=TrueTypeFontProvider(font_path.read_bytes(), font_number=0),
    )

    with PdfDocument.open(output) as parsed:
        assert "你好" in parsed.extract().text


def test_semantic_writer_is_available_from_public_core_pdf_api() -> None:
    output = public_writer(Document(pages=(Page(page_number=1),)))

    assert output.startswith(b"%PDF-1.7")


def test_semantic_writer_supports_standard_pdf_encryption() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("secret"),)),),
            ),
        )
    )
    encrypted = serialize_document_to_pdf(
        document,
        encryption=StandardPdfEncryption(user_password="open"),
    )

    with PdfDocument.open(encrypted, password="open") as parsed:
        assert "secret" in parsed.extract().text

    with pytest.raises(Exception):
        PdfDocument.open(encrypted, password="wrong")
