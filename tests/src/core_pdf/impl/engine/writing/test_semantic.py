import pytest
from core_document import Block, BlockKind, Document, Page, Table, TableCell, TextLine

from core_pdf import PdfDocument
from core_pdf.impl.engine.writing import serialize_document_to_pdf


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
