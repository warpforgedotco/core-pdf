# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io

import pytest

from core_pdf import PdfDocument


def type0_font(encoding: str, descendant: int) -> bytes:
    return (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiMin-W3 "
        + f"/Encoding /{encoding} /DescendantFonts [{descendant} 0 R] >>".encode()
    )


def cid_font() -> bytes:
    return (
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiMin-W3 "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> "
        b"/FontDescriptor 7 0 R /DW 1000 >>"
    )


def japanese_pdf_without_to_unicode() -> bytes:
    content = (
        b"BT /F1 24 Tf 72 700 Td <93fa967b8cea82a982c8834a83698abf8e9a> Tj ET\n"
        b"BT /F2 24 Tf 400 700 Td <8169816a> Tj ET\n"
        b"BT /F2 24 Tf 10 Ts 400 600 Td <8169> Tj ET\n"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R /F2 8 0 R >> >> /Contents 6 0 R >>"
        ),
        type0_font("90ms-RKSJ-H", 5),
        cid_font(),
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
        (
            b"<< /Type /FontDescriptor /FontName /HeiseiMin-W3 /Flags 6 "
            b"/FontBBox [0 -257 1000 899] /ItalicAngle 0 /Ascent 859 "
            b"/Descent -141 /CapHeight 859 /StemV 78 >>"
        ),
        type0_font("90ms-RKSJ-V", 9),
        cid_font(),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def test_extracts_horizontal_and_vertical_japanese_without_to_unicode() -> None:
    with PdfDocument.open(io.BytesIO(japanese_pdf_without_to_unicode())) as document:
        page = document.pages[0]
        result = page.extract()
        text = "\n".join(line.text for block in result.blocks for line in block.lines)
        runs = page.text_diagnostics().runs
        geometry = page.extract_geometry_summary()

    assert "日本語かなカナ漢字" in text
    assert "（" in text
    assert "）" in text
    assert [run.text for run in runs] == ["日本語かなカナ漢字", "（）", "（"]
    vertical = runs[1]
    risen_vertical = runs[2]
    assert vertical.is_vertical is True
    assert risen_vertical.is_vertical is True
    assert vertical.bbox[2] > vertical.bbox[0]
    assert vertical.bbox[3] > vertical.bbox[1]
    assert risen_vertical.bbox[0] == pytest.approx(vertical.bbox[0])
    assert risen_vertical.bbox[2] == pytest.approx(vertical.bbox[2])
    assert vertical.geometry_issues == ()
    assert risen_vertical.geometry_issues == ()
    assert geometry.error_count == 0
