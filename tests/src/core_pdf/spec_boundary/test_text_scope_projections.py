# SPDX-License-Identifier: AGPL-3.0-only
"""Text transparency boundaries travel separately from extraction projections."""

from types import SimpleNamespace
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.pdfminer._capture import internal_pdfminer_page_program
from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.program import CapturedProgram
from core_pdf.impl._impl.capture.records import CapturedTextBoundary, TilingPattern
from core_pdf.impl._impl.document.recovery.resolver import ObjectResolver
from core_pdf.impl._impl.extract.capture import capture_page
from core_pdf.impl._impl.render import patterns
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf_spec.types import PdfName, PdfReference

internal_FONT = b"/Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> "


def internal_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    return (
        b"<< "
        + dictionary
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_pdf(
    content: bytes,
    *,
    resources: bytes = b"",
    page_entries: bytes = b"",
    extras: tuple[bytes, ...] = (),
) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 60 60] /Resources << "
        + internal_FONT
        + resources
        + b" >> /Contents 4 0 R "
        + page_entries
        + b" >>",
        internal_stream(content),
        *extras,
    ]
    data = b"%PDF-1.7\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        data
        + f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


@pytest.mark.parametrize("knockout", [False, True])
def test_text_boundaries_do_not_enter_public_drawings_or_change_text_seqnos(
    knockout: bool,
) -> None:
    data = internal_pdf(
        b"0 0 1 1 re f /TextState gs BT /F1 8 Tf 2 10 Td (ABC) Tj ET 0 0 m 5 5 l S",
        resources=f"/ExtGState << /TextState << /TK {str(knockout).lower()} >> >>".encode(),
    )
    with PdfDocument(data) as document:
        page = document.pages[0]
        program = page.get_page_program()
        boundaries = program.body.text_boundaries
        assert [boundary.knockout for boundary in boundaries if boundary.kind == "begin"] == [
            knockout
        ]
        assert [drawing.kind for drawing in program.drawings] == ["fill", "stroke"]
        assert [drawing.seqno for drawing in program.drawings] == [0, 2]
        assert [drawing.kind for drawing in page.get_drawings()] == ["fill", "stroke"]
        assert [run.seqno for run in page.text_diagnostics().runs] == [1]
        assert page.extract().text == "ABC"
        assert len(program.glyphs) == 3
        assert len(program.runs) == 1
        assert sum(
            isinstance(command, CapturedTextBoundary) for command in program.commands
        ) == len(boundaries)


def test_text_controls_leave_native_and_ocr_evidence_counts_unchanged() -> None:
    ocr_capture = pytest.importorskip("core_pdf_ocr.impl.extract.capture")
    data = internal_pdf(b"0 0 1 1 re f BT /F1 8 Tf 2 10 Td (ABC) Tj ET 0 0 m 5 5 l S")
    with PdfDocument(data) as document:
        captured = capture_page(document.pages[0])
        enriched = ocr_capture.internal_enrich_capture(captured)
        assert captured.program.text_boundaries
        assert captured.evidence.native_characters == enriched.evidence.native_characters == 3
        assert captured.evidence.glyphs.glyph_count == enriched.evidence.glyphs.glyph_count == 3
        assert captured.evidence.image_count == enriched.evidence.image_count == 0
        assert len(enriched.program.drawings) == 2
        assert enriched.evidence.vector_complexity == ocr_capture.internal_vector_complexity(
            captured.program.drawings, captured.program.lines
        )
        assert enriched.program is captured.program


def test_annotation_text_boundaries_are_partitioned_by_appearance_scope() -> None:
    appearance_dictionary = (
        b"/Type /XObject /Subtype /Form /BBox [0 0 30 15] /Resources << " + internal_FONT + b" >>"
    )
    data = internal_pdf(
        b"BT /F1 8 Tf 2 45 Td (BODY) Tj ET",
        page_entries=(
            b"/Annots [ << /Type /Annot /Subtype /Stamp /Rect [0 20 30 35] /AP << /N 5 0 R >> >> "
            b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 30 15] /AP << /N 6 0 R >> >> ]"
        ),
        extras=(
            internal_stream(b"BT /F1 8 Tf 2 4 Td (ONE) Tj ET", appearance_dictionary),
            internal_stream(b"BT /F1 8 Tf 2 4 Td (TWO) Tj ET", appearance_dictionary),
        ),
    )
    with PdfDocument(data) as document:
        program = document.pages[0].get_page_program()
        assert len(program.appearances) == 2
        scopes = (program.body, *(appearance.program for appearance in program.appearances))
        assert ["".join(run.text for run in scope.runs) for scope in scopes] == [
            "BODY",
            "ONE",
            "TWO",
        ]
        for scope in scopes:
            assert [
                event.kind for event in scope.text_boundaries if event.kind in {"begin", "end"}
            ] == [
                "begin",
                "end",
            ]
        boundaries = tuple(event for scope in scopes for event in scope.text_boundaries)
        assert program.text_boundaries == boundaries
        assert len({id(event) for event in boundaries}) == len(boundaries)


def test_pdfminer_capture_retains_text_boundaries_without_new_drawings() -> None:
    data = internal_pdf(b"BT /F1 8 Tf 2 10 Td (ABC) Tj ET 0 0 1 1 re f")
    with PdfDocument(data) as document:
        program = internal_pdfminer_page_program(document.pages[0])
        assert [
            event.kind for event in program.text_boundaries if event.kind in {"begin", "end"}
        ] == [
            "begin",
            "end",
        ]
        assert "".join(run.text for run in program.runs) == "ABC"
        assert [drawing.kind for drawing in program.drawings] == ["fill"]


def test_pattern_render_program_retains_its_captured_text_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pattern = internal_stream(
        b"BT /F1 8 Tf 2 4 Td (A) Tj ET",
        b"/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 15 15] /XStep 15 /YStep 15 "
        b"/Resources << " + internal_FONT + b" >>",
    )
    data = internal_pdf(
        b"/Pattern cs /Pt scn 0 0 30 15 re f",
        resources=b"/Pattern << /Pt 5 0 R >>",
        extras=(pattern,),
    )
    captured_programs: list[CapturedProgram] = []

    def record_program(*args: Any, **kwargs: Any) -> CapturedProgram:
        result = CapturedProgram(*args, **kwargs)
        captured_programs.append(result)
        return result

    monkeypatch.setattr(patterns, "CapturedProgram", record_program)
    with PdfDocument(data) as document:
        page = document.pages[0]
        captured_pattern = page.get_page_program().drawings[0].fill_pattern
        assert isinstance(captured_pattern, TilingPattern)
        assert any(event.kind == "begin" for event in captured_pattern.text_boundaries)
        page.render().rasterize()
        assert captured_programs
        assert captured_programs[0].text_boundaries == tuple(captured_pattern.text_boundaries)
        assert len(page.get_drawings()) == 1
        assert page.extract().text == ""


@pytest.mark.parametrize("initial_knockout", [False, True])
def test_tolerant_pattern_uses_defining_stream_initial_text_knockout(
    initial_knockout: bool,
) -> None:
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.graphics.text_knockout = initial_knockout
    pattern = PdfStream(
        raw_data=b"BT ET 0 0 1 1 re f",
        dictionary={
            "PatternType": 1,
            "PaintType": 1,
            "TilingType": 1,
            "BBox": [0, 0, 1, 1],
            "XStep": 1,
            "YStep": 1,
        },
    )
    state.stream_executor.consume(
        PdfStream(raw_data=b"/Changed gs /Pattern cs /Pt scn 0 0 1 1 re f"),
        {
            "Pattern": {"Pt": pattern},
            "ExtGState": {"Changed": {"TK": not initial_knockout}},
            "ColorSpace": {"Pattern": PdfName.of("Pattern")},
        },
        IDENTITY_MATRIX,
        0,
    )
    captured_pattern = state.drawings[0].fill_pattern
    assert isinstance(captured_pattern, TilingPattern)
    assert [
        event.knockout for event in captured_pattern.text_boundaries if event.kind == "begin"
    ] == [initial_knockout]


def test_reader_ignores_unreadable_TK_reference_inside_text_before_resolving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = PdfReference(999, 0)
    original_resolve = ObjectResolver.resolve

    def resolve(self: ObjectResolver, value: object) -> object:
        if value is unreadable:
            raise AssertionError("ignored TK was resolved")
        return original_resolve(self, value)

    monkeypatch.setattr(ObjectResolver, "resolve", resolve)
    resolver = ObjectResolver(b"", {})
    state = TextState(SimpleNamespace(resolver=resolver, resolve=resolver.resolve))
    state.resources = {"ExtGState": {"Control": {"TK": unreadable, "ca": 0.25}}}
    state.op_BT((), 0)
    state.op_gs((PdfName.of("Control"),), 0)
    assert state.graphics.text_knockout
    assert state.graphics.fill_opacity == 0.25
    state.op_ET((), 0)
    with pytest.raises(AssertionError, match="ignored TK was resolved"):
        state.op_gs((PdfName.of("Control"),), 0)
