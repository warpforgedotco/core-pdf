import pytest

from core_pdf import PdfDocument, PdfSignaturePlan, PdfUnsupportedError
from core_pdf import StandardPdfEncryption as PublicEncryption
from core_pdf import serialize_document_to_pdf as public_writer
from core_pdf.impl.engine.structured import (
    Annotation,
    Block,
    BlockKind,
    Document,
    Page,
    Table,
    TableCell,
    TextLine,
)
from core_pdf.impl.engine.writing import (
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


def test_semantic_writer_emits_minimal_tagged_page_structure() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("One"), TextLine("Two"))),),
            ),
        )
    )

    pdf = serialize_document_to_pdf(document)

    with PdfDocument.open(pdf) as parsed:
        structure = parsed.structure
        assert structure is not None
        elements = tuple(structure.find_all())
        assert [element.role for element in elements] == ["Div", "P", "P"]
        assert [element.page_index for element in elements] == [0, 0, 0]


def test_semantic_writer_preserves_heading_tag_roles() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.HEADING, (TextLine("Title"),), level=2),),
            ),
        )
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        structure = parsed.structure
        assert structure is not None
        assert [element.role for element in structure.find_all()] == ["Div", "Sect", "H2"]


def test_semantic_writer_nests_lower_level_headings() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(
                    Block(1, BlockKind.HEADING, (TextLine("One"),), level=1),
                    Block(2, BlockKind.HEADING, (TextLine("One.A"),), level=2),
                ),
            ),
        )
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        structure = parsed.structure
        assert structure is not None
        sections = tuple(structure.find_all("Sect"))
        assert len(sections) == 2
        assert sections[1].parent is not None
        assert sections[1].parent.role == "Sect"


def test_semantic_writer_emits_list_and_table_cell_roles() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.LIST, (TextLine("Item"),)),),
                tables=(
                    Table(
                        1,
                        rows=(
                            (TableCell(0, 0, "Header"),),
                            (TableCell(1, 0, "Value"),),
                        ),
                    ),
                ),
            ),
        )
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        structure = parsed.structure
        assert structure is not None
        assert [element.role for element in structure.find_all()] == ["Div", "LI", "TH", "TD"]


def test_semantic_writer_emits_figure_alt_and_artifact_roles() -> None:
    from core_pdf.impl.engine.structured import Figure

    document = Document(
        pages=(
            Page(
                page_number=1,
                figures=(
                    Figure(1, kind="image", metadata={"alt": "A chart"}),
                    Figure(2, kind="rule", metadata={"decorative": True}),
                ),
            ),
        )
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        structure = parsed.structure
        assert structure is not None
        figures = tuple(structure.find_all("Figure"))
        assert [element.alternate_description for element in figures] == ["A chart"]
        assert tuple(structure.find_all("Artifact"))
        assert structure.role_map["CoreFigure"] == "Figure"


def test_semantic_writer_propagates_document_language() -> None:
    document = Document(
        pages=(Page(page_number=1),),
        metadata={"Lang": "en-US", "Title": "Accessible document"},
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        language = parsed.catalog()[next(key for key in parsed.catalog() if str(key) == "Lang")]
        assert parsed.resolver.resolve_str(language) == "en-US"
        assert parsed.structure is not None
        structure_language = parsed.structure.props[
            next(key for key in parsed.structure.props if str(key) == "Lang")
        ]
        assert parsed.resolver.resolve_str(structure_language) == "en-US"


def test_semantic_writer_generates_outlines_from_headings() -> None:
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.HEADING, (TextLine("Introduction"),), level=1),),
            ),
        )
    )

    with PdfDocument.open(serialize_document_to_pdf(document)) as parsed:
        outlines = tuple(parsed.outlines)
        assert len(outlines) == 1
        assert outlines[0].level == 0
        assert outlines[0].title == "Introduction"
        assert outlines[0].page_index == 0


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
        encryption=PublicEncryption(user_password="open"),
    )

    with PdfDocument.open(encrypted, password="open") as parsed:
        assert "secret" in parsed.extract().text

    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument.open(encrypted, password="wrong")


def test_semantic_writer_delegates_detached_signature_to_external_provider() -> None:
    class RecordingSigner:
        data: bytes | None = None

        def sign(self, data: bytes) -> bytes:
            self.data = data
            return b"cms-signature"

    signer = RecordingSigner()
    document = Document(
        pages=(
            Page(
                page_number=1,
                blocks=(Block(1, BlockKind.PARAGRAPH, (TextLine("signed"),)),),
            ),
        )
    )

    output = serialize_document_to_pdf(
        document,
        signature=PdfSignaturePlan(signer, contents_length=128),
    )

    import re

    match = re.search(rb"/ByteRange \[0 (\d+) (\d+) (\d+)\]", output)
    assert match is not None
    contents_start, contents_end, trailing_length = (int(value) for value in match.groups())
    assert trailing_length == len(output) - contents_end
    assert signer.data == output[:contents_start] + output[contents_end:]
    assert b"/SubFilter /adbe.pkcs7.detached" in output
    assert b"636D732D7369676E6174757265" in output


def test_semantic_writer_preserves_page_annotations_when_signing() -> None:
    class Signer:
        def sign(self, data: bytes) -> bytes:
            return data[:16]

    document = Document(
        pages=(
            Page(
                page_number=1,
                annotations=(
                    Annotation(subtype="Text", bbox=(10.0, 10.0, 20.0, 20.0), contents="note"),
                ),
            ),
        )
    )

    output = serialize_document_to_pdf(
        document,
        signature=PdfSignaturePlan(Signer(), contents_length=32),
    )

    with PdfDocument.open(output) as parsed:
        annotations = parsed.pages[0].get_annotations()
        assert [annotation.subtype for annotation in annotations] == ["Text", "Widget"]
        assert annotations[0].contents == "note"


def test_semantic_writer_rejects_combined_encryption_and_signature() -> None:
    class Signer:
        def sign(self, data: bytes) -> bytes:
            return data[:1]

    with pytest.raises(ValueError, match="cannot be combined"):
        serialize_document_to_pdf(
            Document(pages=(Page(page_number=1),)),
            encryption=PublicEncryption(user_password="secret"),
            signature=PdfSignaturePlan(Signer()),
        )
