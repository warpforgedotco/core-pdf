# SPDX-License-Identifier: AGPL-3.0-only
"""Whether an unpainted text layer is the page's real text is a whole-page property.

Render mode 3 and sub-0.1pt text paint nothing. A scan carrying an OCR layer is
made entirely of such text and must still extract; a normal page's hidden
watermark must not. Deciding that per show-text operator, from the runs captured
so far, made the answer depend on the order the operators appear in -- the first
text object on a page has no preceding runs to look at. These pin the decision to
the page instead.
"""

from __future__ import annotations

import io

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


def stream_obj(data: bytes, extra: bytes = b"") -> bytes:
    return f"<< /Length {len(data)} {extra.decode()} >>\nstream\n".encode() + data + b"\nendstream"


def page_pdf(content: bytes) -> bytes:
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            stream_obj(content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


HIDDEN_ALPHA = b"BT /F1 12 Tf 3 Tr 50 700 Td (HiddenAlpha) Tj ET\n"
HIDDEN_BETA = b"BT /F1 12 Tf 3 Tr 50 600 Td (HiddenBeta) Tj ET\n"
VISIBLE = b"BT /F1 12 Tf 0 Tr 50 500 Td (VisibleText) Tj ET\n"


def extracted_text(content: bytes) -> str:
    with PdfDocument.open(io.BytesIO(page_pdf(content))) as document:
        return document.pages[0].extract().text


def run_visibility(content: bytes) -> dict[str, bool]:
    with PdfDocument.open(io.BytesIO(page_pdf(content))) as document:
        runs = document.pages[0].text_diagnostics().runs
        return {run.text: run.visible for run in runs}


def test_hidden_text_before_visible_text_is_still_hidden() -> None:
    """The first text object on the page has no preceding runs to judge against."""
    assert extracted_text(HIDDEN_ALPHA + VISIBLE + HIDDEN_BETA) == "VisibleText"


def test_hidden_text_visibility_does_not_depend_on_operator_order() -> None:
    leading = run_visibility(HIDDEN_ALPHA + VISIBLE + HIDDEN_BETA)
    trailing = run_visibility(VISIBLE + HIDDEN_ALPHA + HIDDEN_BETA)
    assert leading == trailing
    assert leading == {"HiddenAlpha": False, "VisibleText": True, "HiddenBeta": False}


def test_an_unpainted_layer_survives_being_split_across_text_objects() -> None:
    """One BT per line is ordinary in an OCR layer; every line must still extract."""
    split = extracted_text(HIDDEN_ALPHA + HIDDEN_BETA)
    single = extracted_text(
        b"BT /F1 12 Tf 3 Tr 50 700 Td (HiddenAlpha) Tj 0 -20 Td (HiddenBeta) Tj ET\n"
    )
    assert "HiddenAlpha" in split
    assert "HiddenBeta" in split
    assert "HiddenAlpha" in single
    assert "HiddenBeta" in single


def test_painted_text_is_unaffected() -> None:
    content = VISIBLE + b"BT /F1 12 Tf 0 Tr 50 400 Td (MoreVisible) Tj ET\n"
    assert run_visibility(content) == {"VisibleText": True, "MoreVisible": True}
