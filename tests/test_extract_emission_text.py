# SPDX-License-Identifier: AGPL-3.0-only
"""Decoded PDF text must survive emission regardless of spelling or symbol count."""

import pytest

from tests.helpers.pdf_bytes import one_page_pdf, open_pdf


@pytest.mark.parametrize(
    "text",
    [
        "> 5",
        "~",
        "+",
        "$",
        "*",
        "[ 01 ]",
        "x[0]",
        "able to read",
        "• 42 87",
        "7!19/71 T[34 G! (%! Warning!",
        "ence on global dynamics",
        "tions within states",
        "ating shifts continue",
        "ducted studies",
        "r • r-",
        "2Y--> -- a IIIISCI...........................",
        "76391*11 IOIIlo6 I * 9 2*0 118)96 '1'1322) '1'19)20 IZZO 1911*2.1,z,z CSM/l\":OST L*O*Io",
    ],
)
def test_document_extraction_preserves_independently_verified_decoded_text(text: str) -> None:
    # These exact strings were extracted and visibly painted by Poppler 26.07.0
    # before changing expectations: pdftotext -layout reference.pdf - and
    # pdftoppm -singlefile -r 100 -png reference.pdf reference. Spacing may differ
    # in Poppler's text output, but every authored operator/identifier survives.
    escaped = (
        text.encode("cp1252").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    )
    pdf = one_page_pdf(
        b"BT /F1 10 Tf 36 740 Td (" + escaped + b") Tj ET",
        font=b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    )
    with open_pdf(pdf) as document:
        result = document.extract()

    # Existing glyph-spacing reconstruction can close spaces before brackets;
    # this regression checks preservation of decoded characters, independently.
    assert "".join(result.text.split()) == "".join(text.split())
    assert result.pages[0].blocks
