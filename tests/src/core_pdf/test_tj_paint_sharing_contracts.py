"""The strings of one TJ array share the paint their glyphs record."""

import re
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.glyphs import GlyphPaint
from core_pdf.impl.capture.recording import TextState
from core_pdf.impl.capture.tolerant_state import RecoveringTextState

CONTENT = (
    b"BT /F1 12 Tf 1 0 0 rg 0 0 1 RG 2 w 20 100 Td "
    b"[(Ab) -250 (cd) 120 (ef)] TJ 0 1 0 rg [(gh) -50 (ij)] TJ ET"
)


def one_page_pdf(content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    data = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number, body in objects.items():
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, 6))
    data += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(data)


def captured_glyphs() -> list[str]:
    with PdfDocument(one_page_pdf(CONTENT)) as document:
        glyphs = document.pages[0].get_page_program().glyphs
        return [re.sub(r"0x[0-9a-f]+", "", repr(glyph)) for glyph in glyphs]


def test_each_tj_array_builds_its_paint_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[GlyphPaint] = []
    original = TextState.glyph_paint

    def counted(self: Any, fill: Any) -> GlyphPaint:
        paint = original(self, fill)
        built.append(paint)
        return paint

    monkeypatch.setattr(TextState, "glyph_paint", counted)
    glyphs = captured_glyphs()
    assert len(glyphs) == 10
    assert len(built) == 2
    assert [paint.fill for paint in built] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def test_shared_paint_captures_what_a_paint_per_string_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = captured_glyphs()
    monkeypatch.setattr(TextState, "append_tj_array", RecoveringTextState.append_tj_array)
    assert captured_glyphs() == shared
