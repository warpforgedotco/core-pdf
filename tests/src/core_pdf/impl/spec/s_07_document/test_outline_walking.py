# SPDX-License-Identifier: AGPL-3.0-only
"""Outline traversal (12.3.3) resolves siblings and children shallowly."""

from __future__ import annotations

from io import BytesIO

from core_pdf.impl.spec.s_07_document.document import PdfDocument
from tests.helpers.pdf_bytes import assemble_pdf, stream_obj


def test_outline_links_are_resolved_shallowly() -> None:
    pdf = assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /Outlines 5 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            stream_obj(b""),
            b"<< /Type /Outlines /First 6 0 R >>",
            b"<< /Title (First) /First 7 0 R >>",
            b"<< /Title (Child) /Dest [3 0 R /Fit] /Next 8 0 R >>",
            b"<< /Title (Sibling) >>",
        ]
    )

    with PdfDocument(BytesIO(pdf)) as document:
        result = document.iter_outlines()

    assert [item.title for item in result] == ["First", "Child", "Sibling"]
    assert [item.level for item in result] == [0, 1, 1]
    assert result[1].page_index == 0
