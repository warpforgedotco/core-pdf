from __future__ import annotations

import sys
from io import BytesIO

from core_pdf.integrations.pdfminer.six.pdfdocument import (  # ty: ignore[unresolved-import]
    PDFDocument,
)
from core_pdf.integrations.pdfminer.six.pdfparser import (  # ty: ignore[unresolved-import]
    PDFParser,
)


def make_pdf_with_xref_chain(length: int) -> bytes:
    header = b"%PDF-1.4\n"
    catalog_offset = len(header)
    catalog = b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    parts = [header, catalog]
    size = len(header) + len(catalog)
    previous_offset: int | None = None
    xref_offsets = []

    for _ in range(length):
        xref_offsets.append(size)
        previous = b"" if previous_offset is None else b" /Prev " + str(previous_offset).encode()
        section = (
            b"xref\n0 2\n0000000000 65535 f \n"
            + f"{catalog_offset:010d} 00000 n \n".encode()
            + b"trailer\n<< /Size 2 /Root 1 0 R"
            + previous
            + b" >>\n"
        )
        parts.append(section)
        size += len(section)
        previous_offset = xref_offsets[-1]

    parts.append(b"startxref\n" + str(xref_offsets[-1]).encode() + b"\n%%EOF\n")
    return b"".join(parts)


def test_xref_chain_can_exceed_python_recursion_limit() -> None:
    chain_length = sys.getrecursionlimit() + 100
    parser = PDFParser(BytesIO(make_pdf_with_xref_chain(chain_length)))

    document = PDFDocument(parser, fallback=False)

    assert len(document.xrefs) == chain_length
