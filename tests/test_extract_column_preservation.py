# SPDX-License-Identifier: AGPL-3.0-only
"""Real PDF regressions for columns lost by late text/table heuristics.

The generated PDFs were checked with qpdf 12.3.2 and Poppler 26.07.0.
Poppler's layout text and raster both retain the two separated columns. Its
text extractor collapses authored letter spacing; core retains explicit spaces.
"""

import pytest

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


def column_pdf(left: str, right: str) -> bytes:
    content = [b"BT /F1 10 Tf 20 380 Td (Data report) Tj ET"]
    for row in range(9):
        y = 340 - row * 30
        content.extend(
            f"BT /F1 10 Tf {x} {y} Td ({text}) Tj ET".encode()
            for x, text in ((20, left), (230, right))
        )
    return one_page_pdf(b"\n".join(content), media_box=(0, 0, 600, 420))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Alpha beta gamma delta", "Epsilon zeta eta theta"),
        ("I 0 ll 13 8 7 o o", "531o6 10llfo2 relDc c t2 l"),
    ],
)
def test_two_column_tables_retain_all_cells(left: str, right: str) -> None:
    with open_pdf(column_pdf(left, right)) as document:
        page = document.pages[0].extract()
    assert len(page.tables) == 1
    rows = [[cell.text for cell in row] for row in page.tables[0].rows]
    assert [row for row in rows if row[0] == left] == [[left, right]] * 9
    assert page.text.count(left) == page.text.count(right) == 9


def test_authored_character_spaces_do_not_erase_a_column_boundary() -> None:
    left = "T o t a l o f 2 0 2 3"
    right = "S e n i o r N o t e s 2 6 2 7 5"
    with open_pdf(column_pdf(left, right)) as document:
        page = document.pages[0].extract()
    assert page.text.count(f"{left} {right}") == 9
    for block in page.blocks:
        for line in block.lines:
            if line.text.startswith(left):
                words = line.words
                assert words[0].text == "T"
                assert words[-1].text == "5"
                assert all(word.bbox is not None for word in words)


@pytest.mark.parametrize("right_offset", [0, 6, 10])
def test_detection_keeps_wrapped_prose_in_columns_as_blocks(right_offset: int) -> None:
    sentences = (
        "This paragraph continues across its lines.",
        "The line lengths are deliberately different.",
        "Short continuation.",
        "Another sentence carries the same thought.",
        "Final words of this paragraph.",
    )
    content = b"\n".join(
        f"BT /F1 10 Tf {x} {340 - row * 14 - offset} Td ({text}) Tj ET".encode()
        for x, offset in ((20, 0), (310, right_offset))
        for row, text in enumerate(sentences)
    )
    with open_pdf(one_page_pdf(content, media_box=(0, 0, 600, 420))) as document:
        page = document.pages[0].extract()
    assert page.tables == ()
    assert len(page.blocks) == 2
    assert [block.text for block in page.blocks] == ["\n".join(sentences)] * 2
