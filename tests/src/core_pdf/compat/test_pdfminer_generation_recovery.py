# SPDX-License-Identifier: AGPL-3.0-only
"""Invalid generations cannot hide valid headers during compatibility recovery."""

from core_pdf import PdfDocument
from core_pdf.api.compat.pdfminer import internal_pdfminer_resolvable_pages


def test_invalid_later_header_does_not_hide_valid_page_in_pdfminer_fallback() -> None:
    data = (
        b"%PDF-1.7\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >> endobj\n"
        b"3 65536 obj << /Type /NotAPage >> endobj\n"
        b"trailer << /Root 1 0 R /Size 4 >>\n%%EOF\n"
    )
    with PdfDocument.open(data) as document:
        assert len(document.pages) == 1
        assert [index for index, _ in internal_pdfminer_resolvable_pages(document)] == [0]
