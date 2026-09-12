# SPDX-License-Identifier: AGPL-3.0-only
"""Form raster clip scopes are not extracted drawing or text-layout content."""

from dataclasses import replace
from io import BytesIO
from typing import Any

import pytest

from core_pdf import PdfDocument, PdfPage
from core_pdf.api.compat import pdfminer, pdfplumber
from core_pdf.api.compat.pdfminer import _projection
from core_pdf.api.compat.pdfminer._capture import internal_pdfminer_page_program
from core_pdf.impl._impl.capture.records import (
    CapturedDrawing,
    CapturedPath,
    CapturedSubpath,
    marker_drawing,
)
from core_pdf.impl._impl.extract.contracts import ObservationBatch
from core_pdf_ocr.impl.extract.capture import (
    internal_uncovered_vector_area,
    internal_vector_complexity,
)


def internal_stream(content: bytes, entries: bytes = b"") -> bytes:
    return (
        b"<< "
        + entries
        + f" /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream"
    )


def internal_document() -> bytes:
    bodies = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 80 80] "
        b"/Resources << /XObject << /Form 5 0 R >> >> /Contents 4 0 R >>",
        internal_stream(b"/Form Do"),
        internal_stream(
            b"2 2 6 4 re f BT /F 10 Tf 10 20 Td (AB) Tj ET",
            b"/Type /XObject /Subtype /Form /BBox [0 0 60 60] "
            b"/Resources << /Font << /F 6 0 R >> >>",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )
    result = b"%PDF-1.7\n"
    offsets = [0]
    for number, body in enumerate(bodies, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer << /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def internal_scope(seqno: int = 0) -> CapturedDrawing:
    return CapturedDrawing(
        kind="scope-begin",
        seqno=seqno,
        fill=None,
        fill_opacity=None,
        line_width=0.0,
        path=CapturedPath([CapturedSubpath([(0, 0), (60, 0), (60, 60), (0, 60)], closed=True)]),
    )


def test_public_drawing_projection_omits_form_clip_geometry() -> None:
    scope = internal_scope()
    paint = replace(scope, kind="fill", seqno=1, fill=(1.0, 0.0, 0.0))
    records = PdfPage.internal_drawing_records((scope, paint, marker_drawing("scope-end", 2)))
    assert len(records) == 1
    assert records[0].kind == "fill"
    assert records[0].seqno == 1
    assert records[0].rect == (0.0, 0.0, 60.0, 60.0)
    assert records[0].path is paint.path


def test_form_clip_does_not_become_a_pdfplumber_curve_or_object() -> None:
    with pdfplumber.open(BytesIO(internal_document())) as document:
        page = document.pages[0]
        assert set(page.objects) == {"char", "rect", "curve"}
        assert len(page.rects) == len(page.curves) == 1
        assert page.rects[0]["x0"] == 2
        assert page.rects[0]["x1"] == 8
        assert "".join(char["text"] for char in page.chars) == "AB"


@pytest.mark.parametrize("kind", ["scope-begin", "scope-end", "stroke"])
def test_pdfminer_form_snippets_ignore_only_scope_records(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with PdfDocument(internal_document()) as document:
        page = document.pages[0]
        original = internal_pdfminer_page_program(page)
        assert len(original.glyphs) == 2
        # Leave sequence room between the two letters for a scope control.
        glyphs = tuple(
            replace(glyph, seqno=index * 3) for index, glyph in enumerate(original.glyphs)
        )
        control = replace(internal_scope(1), kind=kind, xobject_depth=1)
        products = replace(original, glyphs=glyphs, drawings=(control,))
        monkeypatch.setattr(_projection, "internal_pdfminer_page_program", lambda _page: products)
        projected = _projection.internal_project_page(page, pdfminer.LAParams())
        figures = [item for item in projected if isinstance(item, pdfminer.LTFigure)]
        assert len(figures) == 1
        assert figures[0].text_snippets == (("A", "B") if kind == "stroke" else ("AB",))


def test_form_scopes_preserve_native_structured_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    with PdfDocument(internal_document()) as document:
        page = document.pages[0]
        program = page.get_page_program()
        expected = page.extract()
        assert expected.blocks
        # A thin scope box near the text must not become a decoration or table line.
        scope = internal_scope()
        scope.path = CapturedPath([CapturedSubpath([(10, 18), (30, 18)], closed=False)])
        body = replace(program.body, drawings=(*program.body.drawings, scope))
        with_scope = replace(program, body=body)

        def internal_program(_page: PdfPage, **_options: Any) -> Any:
            return with_scope

        monkeypatch.setattr(PdfPage, "get_page_program", internal_program)
        assert page.extract() == expected


@pytest.mark.parametrize("paint_count", [60, 179, 180])
def test_form_scopes_do_not_escalate_ocr_vector_evidence(paint_count: int) -> None:
    observations = ObservationBatch.from_columns(["A"], [(70.0, 70.0, 75.0, 75.0)], source=0)
    scope = internal_scope()
    paint = replace(scope, kind="fill", seqno=1, fill=(0.0, 0.0, 0.0))
    drawings = (paint,) * paint_count
    with_scopes = (scope, paint, marker_drawing("scope-end", 2)) * paint_count
    expected = internal_uncovered_vector_area(drawings, observations)
    assert expected == (None if paint_count < 180 else paint_count * 3600.0)
    assert internal_uncovered_vector_area(with_scopes, observations) == expected
    assert internal_vector_complexity(with_scopes, ()) == internal_vector_complexity(drawings, ())
