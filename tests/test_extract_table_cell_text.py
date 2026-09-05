# SPDX-License-Identifier: AGPL-3.0-only
"""Table cells preserve authored spaces, operators, and punctuation.

These generated ruled PDFs were checked with Poppler 26.07.0 before setting
expectations. ``pdftotext -raw`` retains every authored value, including the
separated labels and numeric groups; it joins adjacent fragments across fonts.
"""

import pytest

from core_pdf.impl._impl.extract.table_cleanup import internal_cell_text
from tests.helpers.extract_fakes import observations
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


def table_cell_pdf(first: bytes, second: bytes) -> bytes:
    """Two data rows whose first cells contain the supplied text-showing operators."""
    commands = [b"0.5 w"]
    commands.extend(f"{x} 170 m {x} 260 l S".encode() for x in (20, 300, 380))
    commands.extend(f"20 {y} m 380 {y} l S".encode() for y in (170, 200, 230, 260))
    commands.extend(
        (
            b"BT /F1 12 Tf 30 240 Td (Values) Tj ET",
            b"BT /F1 12 Tf 310 240 Td (Count) Tj ET",
        )
    )
    for y, operators, count in ((210, first, 1), (180, second, 2)):
        commands.extend(
            (
                f"BT /F1 12 Tf 30 {y} Td ".encode() + operators + b" ET",
                f"BT /F1 12 Tf 310 {y} Td ({count}) Tj ET".encode(),
            )
        )
    return one_page_pdf(
        b"\n".join(commands),
        media_box=(0, 0, 400, 300),
        resources=b"<< /Font << /F1 5 0 R /F2 6 0 R >> >>",
        extra_objects=(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",),
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("A   B   C   D   E   F   G   H", "I   J   K   L   M   N   O   P"),
        ("0   1   2   3   4   5   6   7", "8   7   6   5   4   3   2   1"),
        ("1   2 / 3   4", "5   6 / 7   8"),
        ("---", "..."),
        ("value ...", ". . value"),
        ("10 /1 9/21 ...", "40 / 20"),
        ("now here", "a part"),
    ],
)
def test_ruled_table_preserves_decoded_cell_content(first: str, second: str) -> None:
    with open_pdf(table_cell_pdf(f"({first}) Tj".encode(), f"({second}) Tj".encode())) as document:
        page = document.pages[0].extract()

    assert len(page.tables) == 1
    rows = tuple(tuple(cell.text for cell in row) for row in page.tables[0].rows)
    assert rows == (
        ("Values", "Count"),
        (" ".join(first.split()), "1"),
        (" ".join(second.split()), "2"),
    )
    for row in page.tables[0].rows:
        assert all(cell.bbox is not None for cell in row)
    # The same source glyphs may also form text blocks. Keeping both projections'
    # text intact allows the spatial deduplicator to remove those copies.
    assert page.blocks == ()
    assert page.text.split() == ["Values", "Count", *first.split(), "1", *second.split(), "2"]


def test_source_geometry_joins_adjacent_fragments_across_font_changes() -> None:
    pdf = table_cell_pdf(b"(12) Tj /F2 12 Tf (34) Tj", b"(10/1) Tj /F2 12 Tf (9/21) Tj")
    with open_pdf(pdf) as document:
        page = document.pages[0].extract()

    assert len(page.tables) == 1
    assert tuple(tuple(cell.text for cell in row) for row in page.tables[0].rows) == (
        ("Values", "Count"),
        ("1234", "1"),
        ("10/19/21", "2"),
    )
    assert page.blocks == ()


@pytest.mark.parametrize("text", ["---", "...", "● ●", "◦ ◦", "1 2 / 3 4", "A B C D E F G H"])
def test_prepared_observations_preserve_punctuation_and_word_boundaries(text: str) -> None:
    batch = observations(((text, (0.0, 0.0, 100.0, 10.0)),))

    assert internal_cell_text(batch, [0]) == text
