# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from io import BytesIO

from core_pdf.impl.spec.s_07_document.document import PdfDocument
from tests.helpers.pdf_bytes import assemble_pdf, stream_obj


def test_structure_root_keeps_catalog_object_identity() -> None:
    pdf = assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 5 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            stream_obj(b""),
            b"<< /Type /StructTreeRoot >>",
        ]
    )

    with PdfDocument(BytesIO(pdf)) as document:
        structure = document.structure
        root = document.resolver.resolve(document.catalog()["StructTreeRoot"])

        assert structure is not None
        assert structure.props is root
        rebuilt = document.structure
        assert rebuilt is not None
        assert rebuilt.props is root
