"""A text layout reused across strings and operators captures what a fresh one does."""

import re
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.recording import TextState
from tests.src.core_pdf.pdf_bytes import one_page_pdf

CONTENTS = [
    pytest.param(b"BT /F1 12 Tf 10 700 Td [(Wa) -120 (ve) 30 (s)] TJ ET", id="tj"),
    pytest.param(
        b"BT /F1 12 Tf 10 700 Td (one) Tj 0 -14 Td (two) Tj T* (three) Tj"
        b" 14 TL (four) ' 2 Tw (five six) Tj ET",
        id="line-moves",
    ),
    pytest.param(
        b"BT /F1 12 Tf 1 0 0 1 0 0 Tm [(A) -50 (B)] TJ 1 Tc [(C) 20 (D)] TJ"
        b" 90 Tz (E) Tj 3 Ts (F) Tj /F1 9 Tf (G) Tj ET",
        id="text-state",
    ),
    pytest.param(
        b"q -1 0 0 1 300 0 cm BT /F1 10 Tf [(ab) 100 (cd)] TJ 5 5 Td [(e) 7 (f)] TJ ET Q",
        id="identity-text-matrix",
    ),
    pytest.param(
        b"BT /F1 12 Tf 10 600 Td /Span <</MCID 3>> BDC [(x) 10 (y)] TJ EMC [(z) 10 (w)] TJ"
        b" 1 0 0 rg (r) Tj 3 Tr (i) Tj ET BT (n) Tj ET",
        id="marked-and-paint",
    ),
]


def captured(content: bytes) -> tuple[str, ...]:
    # Each document has its own font decoders, so their addresses are masked.
    with PdfDocument(one_page_pdf(content, width=600, height=800)) as document:
        program = document.pages[0].get_page_program()
        return tuple(
            re.sub(r" at 0x[0-9a-f]+", "", repr(part))
            for part in (program.glyphs, program.runs, program.drawings)
        )


@pytest.mark.parametrize("content", CONTENTS)
def test_reused_layouts_capture_what_fresh_ones_do(
    content: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[object] = []
    original_layout = TextState.new_text_layout

    def counted(self: TextState, *args: Any) -> Any:
        layout = original_layout(self, *args)
        built.append(layout)
        return layout

    monkeypatch.setattr(TextState, "new_text_layout", counted)
    reused = captured(content)
    reuses = len(built)
    original_show = TextState.show_text

    def fresh(self: TextState, *args: Any, **kwargs: Any) -> None:
        self.text_layout = None
        original_show(self, *args, **kwargs)

    monkeypatch.setattr(TextState, "show_text", fresh)
    built.clear()
    assert captured(content) == reused
    assert len(built) >= reuses


def test_a_tj_array_builds_one_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    original_layout = TextState.new_text_layout

    def counted(self: TextState, *args: Any) -> Any:
        layout = original_layout(self, *args)
        built.append(layout)
        return layout

    monkeypatch.setattr(TextState, "new_text_layout", counted)
    captured(b"BT /F1 12 Tf 10 700 Td [(a) 1 (b) 2 (c) 3 (d)] TJ (e) Tj ET")
    assert len(built) == 1
