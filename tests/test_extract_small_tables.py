# SPDX-License-Identifier: AGPL-3.0-only
"""Small ruled tables retain their structure independently of their vocabulary."""

import pytest

from core_pdf import PdfDocument
from tests.helpers.pdf_bytes import one_page_pdf


def internal_small_table_pdf(left: str, right: str, *, heading: bool = False) -> bytes:
    content = (
        b"0.5 w 20 120 m 260 120 l 260 170 l 20 170 l 20 120 l S "
        b"140 120 m 140 170 l S\n"
        + f"BT /F1 12 Tf 30 140 Td ({left}) Tj ET\n".encode()
        + f"BT /F1 12 Tf 170 140 Td ({right}) Tj ET".encode()
    )
    if heading:
        content = f"BT /F1 12 Tf 20 320 Td ({left} {right}) Tj ET\n".encode() + content
    return one_page_pdf(content, media_box=(0, 0, 300, 400))


@pytest.mark.parametrize("right", ["I", "J"])
def test_small_table_structure_does_not_depend_on_its_letters(right: str) -> None:
    # qpdf 12.3.2 validates both PDFs; Poppler 26.07.0 pdftotext and pdftoppm
    # confirm the same two-cell ruled row for B/I and B/J. B/I was deleted.
    with PdfDocument(internal_small_table_pdf("B", right)) as document:
        page = document.extract().pages[0]

    assert len(page.tables) == 1
    assert [[cell.text for cell in row] for row in page.tables[0].rows] == [["B", right]]
    assert page.text == f"B\t{right}"


def test_heading_and_disjoint_small_table_both_survive_with_matching_words() -> None:
    # Poppler 26.07.0 preserves both occurrences in small-disjoint-grid.pdf.
    # Layout combines them into a tall block, so block-level overlap alone
    # cannot tell the table's duplicate line from its separate heading.
    with PdfDocument(internal_small_table_pdf("Alpha beta", "Gamma delta", heading=True)) as doc:
        page = doc.extract().pages[0]

    assert len(page.tables) == 1
    assert [[cell.text for cell in row] for row in page.tables[0].rows] == [
        ["Alpha beta", "Gamma delta"]
    ]
    assert [block.text for block in page.blocks] == ["Alpha beta Gamma delta"]
    assert page.text.split() == [
        "Alpha",
        "beta",
        "Gamma",
        "delta",
        "Alpha",
        "beta",
        "Gamma",
        "delta",
    ]
    assert page.blocks[0].bbox is not None
    assert page.blocks[0].bbox[1] > 300


def test_table_text_is_not_duplicated_when_a_line_grazes_the_next_column() -> None:
    # Poppler 26.07.0 emits each phrase once in this ruled-table fixture.
    # The inferred stream-table boundary is x=100; Gamma delta ends at x=102,
    # so considering every touched cell incorrectly added the next column.
    content = (
        b"0.5 w 20 120 m 260 120 l 260 180 l 20 180 l 20 120 l S "
        b"140 120 m 140 180 l S\n"
        b"BT /F1 12 Tf 30 158 Td (Alpha beta) Tj 0 -20 Td (Gamma delta) Tj ET\n"
        b"BT /F1 12 Tf 170 158 Td (Epsilon zeta) Tj 0 -20 Td (Eta theta) Tj ET"
    )
    with PdfDocument(one_page_pdf(content)) as document:
        page = document.extract().pages[0]

    assert page.text.splitlines() == ["Alpha beta\tEpsilon zeta", "Gamma delta\tEta theta"]
    assert len(page.tables) == 1
    assert page.blocks == ()
