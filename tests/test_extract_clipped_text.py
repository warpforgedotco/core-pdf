# SPDX-License-Identifier: AGPL-3.0-only
"""Clipping groups must not turn duplicate removal into unrelated text loss.

Verified with Poppler 26.07.0 ``pdftotext -layout`` before pinning these expectations:
overlaid duplicates appear once, distant repeats twice, and both unique suffixes and
separate lines survive. The fixtures use small fonts to keep every glyph on the page.
"""

import pytest

from core_pdf import PdfDocument
from tests.helpers.pdf_bytes import one_page_pdf


@pytest.mark.parametrize("variant", ["duplicate", "mixed", "distant", "extended", "punctuation"])
def test_extraction_preserves_distinct_text_in_a_duplicate_clipping_group(variant: str) -> None:
    repeated = " ".join(f"word{index}" for index in range(24))
    unique = "UNIQUE CLIPPED CONTENT MUST SURVIVE"

    def text(value: str, y: int) -> bytes:
        return f"BT /F1 5 Tf 10 {y} Td ({value}) Tj ET".encode()

    primary = b"q 0 0 612 792 re W n " + text(repeated, 700) + b" Q "
    match variant:
        case "mixed":
            clipped = text(repeated, 700) + b" " + text(unique, 650)
        case "distant":
            clipped = text(repeated, 650)
        case "extended":
            clipped = text(repeated + " " + unique, 700)
        case "punctuation":
            clipped = text(repeated + " !", 700)
        case _:
            clipped = text(repeated, 700)
    pdf = one_page_pdf(primary + b"q 0 600 612 160 re W n " + clipped + b" Q")

    with PdfDocument(pdf) as document:
        extracted = document.extract().text

    expected = repeated
    if variant in {"mixed", "extended"}:
        expected += " " + unique
    elif variant == "distant":
        expected += " " + repeated
    elif variant == "punctuation":
        expected += " !"
    assert extracted.split() == expected.split()


def test_duplicate_clipping_preserves_different_word_boundaries() -> None:
    # Poppler 26.07.0 pdftotext -raw preserves both paragraphs; removing
    # whitespace made "now here" incorrectly compare equal to "nowhere".
    primary = " ".join(["nowhere"] * 24)
    alternate = " ".join(["now here"] * 24)

    def text(value: str) -> bytes:
        return f"BT /F1 4 Tf 10 700 Td ({value}) Tj ET".encode()

    content = (
        b"q 0 0 612 792 re W n "
        + text(primary)
        + b" Q q 0 600 612 160 re W n "
        + text(alternate)
        + b" Q"
    )
    with PdfDocument(one_page_pdf(content)) as document:
        words = document.extract().text.split()

    assert words.count("nowhere") == 24
    assert words.count("now") == 24
    assert words.count("here") == 24
