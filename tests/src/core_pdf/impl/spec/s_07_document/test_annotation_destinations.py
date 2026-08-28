# SPDX-License-Identifier: AGPL-3.0-only
"""Annotation destinations must reach the structured document as values.

A URI action normally stores its target behind an indirect reference.  If that
reference survives into the structured IR, JSON serialization falls back to
``str()`` on the reference object, so the URL is lost and -- before
``PdfReference`` had a ``__repr__`` -- the output carried a live memory address
and differed between runs of the same file.
"""

from __future__ import annotations

import io
import json

from core_pdf import PdfDocument
from core_pdf.impl.primitives import PdfReference

URL = b"https://example.invalid/target#frag"


def assemble(objects: list[bytes]) -> bytes:
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def link_pdf() -> bytes:
    """A link whose /URI is an indirect reference to the string object."""
    content = b"BT /F1 12 Tf 50 700 Td (Body) Tj ET\n"
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> /Annots [6 0 R] >>",
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Annot /Subtype /Link /Rect [50 690 200 710] /A << /S /URI /URI 7 0 R >> >>",
            b"(" + URL + b")",
        ]
    )


def destinations(pdf_bytes: bytes) -> list[object]:
    with PdfDocument.open(io.BytesIO(pdf_bytes)) as document:
        payload = json.loads(document.extract().to_json())
    page_id = payload["pages"][0]["id"]
    return [
        annotation["destination"]
        for annotation in payload["annotations"]
        if annotation["page_id"] == page_id
    ]


def test_indirect_uri_reaches_structured_output_as_the_url() -> None:
    assert destinations(link_pdf()) == [
        {"S": "URI", "URI": URL.decode()},
    ]


def test_structured_output_is_identical_across_runs() -> None:
    pdf_bytes = link_pdf()

    # Distinct PdfReference objects live at distinct addresses, so a repr that
    # leaked one made the same file serialize differently each time.
    assert destinations(pdf_bytes) == destinations(pdf_bytes)


def test_pdf_reference_renders_as_pdf_syntax_not_an_address() -> None:
    reference = PdfReference(7, 0)

    assert str(reference) == "7 0 R"
    assert repr(reference) == "PdfReference(7, 0)"
    assert "object at 0x" not in str(reference)
