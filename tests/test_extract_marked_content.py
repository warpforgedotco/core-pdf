# SPDX-License-Identifier: AGPL-3.0-only
"""Marked-content replacements must retain neighboring text and its provenance."""

import pytest

from core_pdf.impl._impl.extract.capture import capture_page
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, one_page_pdf, open_pdf, stream_obj


def internal_tagged_pdf(content: bytes) -> bytes:
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 6 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
            stream_obj(b"BT /F1 12 Tf 10 40 Td " + content + b" ET"),
            HELVETICA,
            b"<< /Type /StructTreeRoot /K [7 0 R 8 0 R] "
            b"/ParentTree << /Nums [0 [7 0 R 8 0 R]] >> >>",
            b"<< /Type /StructElem /S /Span /P 6 0 R /Pg 3 0 R /K 0 /ActualText (Alpha) >>",
            b"<< /Type /StructElem /S /Span /P 6 0 R /Pg 3 0 R /K 1 /ActualText (Beta) >>",
        ]
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"/Span << /MCID 0 >> BDC (A) Tj EMC (B) Tj", ("Alpha", "B")),
        (b"(B) Tj /Span << /MCID 0 >> BDC (A) Tj EMC", ("B", "Alpha")),
        (
            b"/Span << /MCID 0 >> BDC (A) Tj EMC /Span << /MCID 1 >> BDC (B) Tj EMC",
            ("Alpha", "Beta"),
        ),
        (
            b"/Span << /MCID 0 >> BDC (A) Tj /Span BMC (B) Tj EMC (C) Tj EMC (D) Tj",
            ("Alpha", "D"),
        ),
    ],
)
def test_structure_actual_text_preserves_adjacent_and_nested_marked_content(
    content: bytes, expected: tuple[str, ...]
) -> None:
    with open_pdf(internal_tagged_pdf(content)) as document:
        capture = capture_page(document.pages[0])
        assert capture.observations.text == expected
        assert "".join(document.extract().text.split()) == "".join(expected)


@pytest.mark.parametrize("replacement", [b"replacement", b""])
def test_inline_actual_text_keeps_neighboring_text_out_of_its_provenance(
    replacement: bytes,
) -> None:
    content = (
        b"BT /F1 12 Tf 10 40 Td (A) Tj /Span << /ActualText ("
        + replacement
        + b") >> BDC (B) Tj EMC (C) Tj ET"
    )
    with open_pdf(one_page_pdf(content)) as document:
        page = document.pages[0]
        runs = page.get_page_program().runs
        assert tuple(run.text for run in runs) == ("A", replacement.decode(), "C")
        assert ("unicode_source", "actual_text") not in runs[0].provenance
        assert ("unicode_source", "actual_text") in runs[1].provenance
        assert ("unicode_source", "actual_text") not in runs[2].provenance
        capture = capture_page(page)
        assert capture.evidence.glyphs.actual_text_characters == len(replacement)
        assert "".join(document.extract().text.split()) == "A" + replacement.decode() + "C"
