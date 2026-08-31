import pytest

from core_pdf import PdfDocument
from core_pdf.impl.primitives import PdfName, PdfReference
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.writing import PdfObjectGraph, serialize_pdf_file


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


def test_pdf_object_graph_allocates_references_and_freezes() -> None:
    graph = PdfObjectGraph()
    pages = graph.add(
        {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [],
            PdfName.of("Count"): 0,
        }
    )
    catalog = graph.add({PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): pages})

    frozen = graph.freeze()

    assert pages == PdfReference(1)
    assert catalog == PdfReference(2)
    assert tuple(frozen) == (1, 2)

    with pytest.raises(RuntimeError, match="frozen"):
        graph.add(None)


def test_pdf_object_graph_can_serialize_a_new_file() -> None:
    graph = PdfObjectGraph()
    pages = graph.add(
        {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [],
            PdfName.of("Count"): 0,
        }
    )
    catalog = graph.add({PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): pages})

    output = graph.to_pdf(trailer={PdfName.of("Root"): catalog})

    with PdfDocument.open(output) as document:
        assert document.page_count() == 0
