import pytest

from core_pdf import PdfDocument
from tests.src.core_pdf.pdf_bytes import serialize_pdf, stream_object


def page_tree_pdf(declared_count: int, kids: int = 1) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids ["
        + b" ".join(b"%d 0 R" % (3 + 2 * index) for index in range(kids))
        + b"] /Count %d >>" % declared_count,
    }
    for index in range(kids):
        number = 3 + 2 * index
        objects[number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents %d 0 R >>"
            % (number + 1)
        )
        objects[number + 1] = stream_object(b"0 0 10 10 re f")
    return serialize_pdf(objects)


@pytest.mark.parametrize(("declared_count", "kids"), [(2, 1), (5, 2), (1, 3), (0, 2)])
def test_page_count_is_the_number_of_pages_that_exist(declared_count: int, kids: int) -> None:
    with PdfDocument(page_tree_pdf(declared_count, kids)) as document:
        assert document.page_count() == len(document.pages) == kids


def test_every_counted_page_can_be_extracted() -> None:
    with PdfDocument(page_tree_pdf(2)) as document:
        extracted = document.extract(pages=range(1, document.page_count() + 1))
    assert len(extracted.pages) == 1


@pytest.mark.parametrize(("declared_count", "expected"), [(2, 2), (0, 0)])
def test_the_declared_count_still_reports_the_page_tree_count(
    declared_count: int, expected: int
) -> None:
    with PdfDocument(page_tree_pdf(declared_count)) as document:
        assert document.declared_page_count() == expected
