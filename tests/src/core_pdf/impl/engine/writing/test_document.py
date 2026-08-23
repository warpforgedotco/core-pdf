import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.writing import serialize_pdf_file
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfName, PdfReference


def test_serialize_pdf_file_builds_a_parseable_classic_xref_pdf() -> None:
    objects = {
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
    }

    result = serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})

    assert result.startswith(b"%PDF-1.7\n")
    assert b"xref\n0 5\n" in result
    assert b"trailer\n" in result
    assert result.endswith(b"%%EOF\n")

    with PdfDocument.open(result) as document:
        assert len(document.pages) == 1


def test_serialize_pdf_file_rejects_empty_object_graph() -> None:
    with pytest.raises(ValueError, match="at least one"):
        serialize_pdf_file({}, trailer={})
