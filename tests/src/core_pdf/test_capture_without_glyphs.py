from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import CaptureOptions
from tests.src.core_pdf.extraction_snapshot import FIXTURES

WITHOUT_GLYPHS = CaptureOptions(
    ink_bounds=False, text_runs=False, render_details=False, glyphs=False
)


@pytest.mark.parametrize(
    "document",
    [
        "pdfplumber/examples/pdfs/ca-warn-report.pdf",
        "pdfminer.six/samples/nonfree/i1040nr.pdf",
        "pypdf/resources/issue-301.pdf",
    ],
)
def test_drawings_keep_their_sequence_without_glyphs(document: str) -> None:
    path = FIXTURES / document
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("fixture corpus not initialized")
    with PdfDocument(path.read_bytes()) as pdf:
        page = pdf.pages[0]
        full = page.get_page_program()
        lean = page.get_page_program(options=WITHOUT_GLYPHS)
    assert not lean.glyphs
    assert not lean.runs
    assert [(d.seqno, d.kind, d.rect) for d in lean.drawings] == [
        (d.seqno, d.kind, d.rect) for d in full.drawings
    ]


def test_painted_paths_are_the_records_the_constructor_builds() -> None:
    from core_pdf.impl.capture_recording import DRAWING_LAYOUT, PATH_PAINT_DEFAULTS
    from core_pdf.impl.capture_records import CapturedDrawing

    values: dict[str, Any] = {name: object() for name in CapturedDrawing.__fields__}
    values.update(PATH_PAINT_DEFAULTS)
    values["items"] = ()
    built = DRAWING_LAYOUT.build(tuple(values[name] for name in CapturedDrawing.__fields__))
    constructed = CapturedDrawing(**values)
    for name in CapturedDrawing.__fields__:
        assert getattr(built, name) is getattr(constructed, name), name


def test_glyphs_off_keeps_runs_but_records_no_glyphs() -> None:
    path = FIXTURES / "pdfminer.six/samples/nonfree/i1040nr.pdf"
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("fixture corpus not initialized")
    with PdfDocument(path.read_bytes()) as pdf:
        page = pdf.pages[0]
        full = page.get_page_program()
        runs_only = page.get_page_program(options=CaptureOptions(glyphs=False))
    assert full.glyphs
    assert not runs_only.glyphs
    assert [(run.text, run.advance_bbox) for run in runs_only.runs] == [
        (run.text, run.advance_bbox) for run in full.runs
    ]
