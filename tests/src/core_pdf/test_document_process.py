# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from pathlib import Path

from core_pdf.cli import process_pdf


def test_pdf_document_process_pdf(tmp_path: Path) -> None:
    from core_pdf.impl.engine.writing import serialize_pdf_file
    from core_pdf.impl.objects import PdfName, PdfReference, PdfStream

    objects = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        2: {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): [PdfReference(3)],
            PdfName.of("Count"): 1,
        },
        3: {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 10, 10],
            PdfName.of("Contents"): PdfReference(4),
        },
        4: PdfStream({}, b"q\nQ\n"),
    }
    pdf_bytes = serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(pdf_bytes)

    # 1. Parse without writing files
    res = process_pdf(pdf_file, "markdown")
    assert res is None

    # 2. Write file next to PDF
    res = process_pdf(pdf_file, "markdown", write_files=True)
    assert res is not None
    assert res == tmp_path / "test.md"
    assert res.exists()

    # 3. Write file to custom output dir
    out_dir = tmp_path / "out"
    res = process_pdf(pdf_file, "markdown", output_dir=out_dir)
    assert res is not None
    assert res == out_dir / "test.md"
    assert res.exists()
