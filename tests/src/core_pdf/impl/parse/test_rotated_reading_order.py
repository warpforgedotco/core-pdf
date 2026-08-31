# SPDX-License-Identifier: AGPL-3.0-only
"""Reading order is decided in the frame the page is displayed in.

/Rotate is not decoration: a page carrying 180 stores its first line at the
bottom of unrotated space. Ordering there walks the page backwards, and a real
document came out reading its outline items l, k, j, i, h.
"""

from __future__ import annotations

import io

import pytest

from core_pdf import PdfDocument


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


def rotated_pdf(rotate: int) -> bytes:
    """Three lines whose display order depends on the page rotation.

    In unrotated space ALPHA sits at the top and CHARLIE at the bottom, so an
    unrotated page reads ALPHA, BRAVO, CHARLIE and a page turned 180 reads
    them the other way about.
    """
    content = (
        b"BT /F1 24 Tf 100 700 Td (ALPHA) Tj ET\n"
        b"BT /F1 24 Tf 100 400 Td (BRAVO) Tj ET\n"
        b"BT /F1 24 Tf 100 100 Td (CHARLIE) Tj ET\n"
    )
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate %d "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>" % rotate,
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def order_of(data: bytes) -> list[str]:
    with PdfDocument.open(io.BytesIO(data)) as document:
        text = document.extract().text
    return [word for word in text.split() if word in {"ALPHA", "BRAVO", "CHARLIE"}]


def test_unrotated_page_reads_down_the_page() -> None:
    assert order_of(rotated_pdf(0)) == ["ALPHA", "BRAVO", "CHARLIE"]


def test_page_rotated_180_reads_in_display_order() -> None:
    # Turned upside down, the line stored lowest is the one a reader sees first.
    assert order_of(rotated_pdf(180)) == ["CHARLIE", "BRAVO", "ALPHA"]


@pytest.mark.parametrize("rotate", [90, 270])
def test_every_rotation_still_reports_all_the_text(rotate: int) -> None:
    assert sorted(order_of(rotated_pdf(rotate))) == ["ALPHA", "BRAVO", "CHARLIE"]
