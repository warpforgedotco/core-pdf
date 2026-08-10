# SPDX-License-Identifier: AGPL-3.0-only
"""Extracted text keeps the characters the fonts actually produced."""

from __future__ import annotations

import io

from core_pdf import PdfDocument
from core_pdf.impl.engine.spec.s_07_content.text_helpers import normalize_extracted_text


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


def page_text(data: bytes) -> str:
    with PdfDocument.open(io.BytesIO(data)) as document:
        return "".join(run.text for run in document.pages[0].text_diagnostics().runs)


def form_feed_pdf() -> bytes:
    """WinAnsi code 014, which the font maps to a form feed and nothing else."""
    content = b"BT /F1 12 Tf 50 400 Td (A\x0cB) Tj ET\n"
    return assemble(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            stream_obj(content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
    )


def test_form_feed_is_not_rewritten_to_a_ligature() -> None:
    # Code 014 used to be rewritten to U+FB01 for every font, which invented an
    # "fi" wherever a document legitimately emitted a form feed.
    text = page_text(form_feed_pdf())
    assert "ﬁ" not in text
    assert "\x0c" in text


def test_lone_surrogates_are_dropped() -> None:
    # Surrogates cannot be encoded to UTF-8, so they must not survive.
    assert normalize_extracted_text("a\ud800b") == "ab"


def test_ordinary_text_is_returned_unchanged() -> None:
    assert normalize_extracted_text("plain ascii") == "plain ascii"
    assert normalize_extracted_text("caf\u00e9 \u2014 na\u00efve") == "caf\u00e9 \u2014 na\u00efve"
