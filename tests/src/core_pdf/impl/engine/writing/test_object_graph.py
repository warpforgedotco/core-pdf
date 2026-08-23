import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine.writing import PdfObjectGraph
from core_pdf.impl.primitives import PdfName, PdfReference


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
