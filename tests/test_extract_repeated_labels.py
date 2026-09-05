# SPDX-License-Identifier: AGPL-3.0-only
"""Repeated labels at distinct positions remain part of native output."""

import pytest

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize("label", ["B", "I", "x"])
@pytest.mark.parametrize("count", [3, 4])
def test_repeated_single_letter_labels_survive_extraction(label: str, count: int) -> None:
    # Before changing expectations, qpdf 12.3.2 validated these exact PDFs and
    # Poppler 26.07.0 pdftotext -layout preserved every printed label. Adding a
    # fourth label previously caused native layout to discard all four.
    commands = [b"BT /F1 16 Tf 50 740 Td (Printed labels) Tj ET"]
    commands.extend(
        f"BT /F1 16 Tf 50 {690 - 60 * index} Td ({label}) Tj ET".encode() for index in range(count)
    )

    with open_pdf(one_page_pdf(b"\n".join(commands))) as document:
        program = document.pages[0].get_page_program()
        result = document.extract()

    assert sum(glyph.text == label for glyph in program.glyphs) == count
    assert result.text.split() == ["Printed", "labels", *([label] * count)]


def test_repeated_boxed_diagram_labels_survive_extraction() -> None:
    # qpdf 12.3.2 validates this exact PDF. Poppler 26.07.0 pdftotext retains
    # four x labels, and its pdftoppm raster confirms four separate boxed nodes.
    commands = [b"BT /F1 16 Tf 50 740 Td (Diagram nodes) Tj ET"]
    for x, y in ((80, 650), (220, 500), (390, 650), (500, 450)):
        commands.append(f"{x - 10} {y - 10} 32 36 re S".encode())
        commands.append(f"BT /F1 16 Tf {x} {y} Td (x) Tj ET".encode())

    with open_pdf(one_page_pdf(b"\n".join(commands))) as document:
        program = document.pages[0].get_page_program()
        result = document.extract()

    assert sum(glyph.text == "x" for glyph in program.glyphs) == 4
    assert result.text.split() == ["Diagram", "nodes", "x", "x", "x", "x"]
