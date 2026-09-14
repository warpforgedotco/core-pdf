# SPDX-License-Identifier: AGPL-3.0-only
"""Reader-tolerant page box recovery on the public document API."""

from __future__ import annotations

from core_pdf import PdfDocument

EXPONENT_MEDIA_BOX_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 1e2 200] >> endobj\n"
    b"trailer << /Root 1 0 R >>\n"
    b"%%EOF\n"
)


def test_page_box_accepts_exponent_token_retained_by_the_reader() -> None:
    # The recovery lexer keeps ``1e2`` as a bare token that the reader's box
    # parser coerces; the strict spec parser would drop the whole box instead.
    with PdfDocument(EXPONENT_MEDIA_BOX_PDF) as document:
        page = document.pages[0]
        assert page.media_box == (0.0, 0.0, 100.0, 200.0)
        assert (page.width, page.height) == (100.0, 200.0)
