"""Path operators the content scanner applies itself capture what their handlers did."""

import re
from copy import replace
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture import recording

ADDRESS = re.compile(r"0x[0-9a-f]+")

CONTENTS = [
    pytest.param(b"10 10 m 50 10 l 50 50 l h S", id="lines"),
    pytest.param(b"10 10 40 20 re f 0 0 m 1.5 -0 l S", id="rect-and-negative-zero"),
    pytest.param(b"10 10 m 20 30 40 30 50 10 c 60 0 70 20 v 80 40 90 10 y h f", id="curves"),
    pytest.param(b"20 30 40 30 50 10 c 60 60 l 5 5 m 7 7 l S", id="curve-without-point"),
    pytest.param(b"60 60 l 1 2 3 4 v 1 2 3 4 y h 5 5 m 6 6 l S", id="ops-without-point"),
    pytest.param(b"2 0 0 2 10 10 cm 0.5 i 10 10 m 20 30 40 30 50 10 c f", id="ctm-flatness"),
    pytest.param(b"10 10 m 20 l 30 30 40 l 1 /N 2 m 5 5 5 5 5 re S", id="bad-operands"),
    pytest.param(
        b"q 10 10 m 20 20 l Q 30 30 l S BT /F1 9 Tf 1 1 Td (x) Tj ET 1 1 m 2 2 l S",
        id="state-around-text",
    ),
    pytest.param(b"10 10 m 1e3 5 l 99999999999999999 1 l 5 5 l S", id="unscannable-numbers"),
]


def one_page_pdf(content: bytes) -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
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


def captured(data: bytes) -> list[object]:
    with PdfDocument(data) as document:
        body = document.pages[0].get_page_program().body
        drawings = [
            (
                ADDRESS.sub("0x", repr(replace(d, path=None))),
                None if d.path is None else [s.points for s in d.path.subpaths],
            )
            for d in body.drawings
        ]
        return [drawings, body.lines.array.tobytes(), ADDRESS.sub("0x", repr(body.glyphs))]


@pytest.mark.parametrize("content", CONTENTS)
def test_native_paths_capture_as_the_handlers_do(
    content: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = one_page_pdf(content)
    applied: list[bool] = []
    original = recording.applies_paths_natively

    def recorded(handlers: Any, state: object) -> bool:
        result = original(handlers, state)
        applied.append(result)
        return result

    monkeypatch.setattr(recording, "applies_paths_natively", recorded)
    native = captured(data)
    assert any(applied)
    monkeypatch.setattr(recording, "applies_paths_natively", lambda *_: False)
    assert captured(data) == native
