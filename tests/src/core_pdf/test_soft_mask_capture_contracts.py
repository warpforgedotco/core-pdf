"""A program captured for extraction records no soft masks; one captured for drawing does."""

import re
from copy import replace
from pathlib import Path

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import DEFAULT_CAPTURE, EXTRACTION_CAPTURE
from core_pdf.impl.capture_records import CapturedDrawing

FIXTURE = Path(__file__).parents[3] / "tests/fixtures/PyMuPDF/tests/resources/test_3450.pdf"

ADDRESS = re.compile(r"0x[0-9a-f]+")

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="reference corpus not initialized")


def test_extraction_skips_the_soft_masks_drawing_captures() -> None:
    with PdfDocument(FIXTURE.read_bytes()) as document:
        page = document.pages[0]
        drawn = page.get_page_program(options=DEFAULT_CAPTURE).body
        extracted = page.get_page_program(options=EXTRACTION_CAPTURE).body
    masked = [d for d in drawn.drawings if d.graphics_soft_mask is not None]
    assert masked
    assert all(d.graphics_soft_mask is None for d in extracted.drawings)

    # Everything else about the drawings is what the drawing capture made.
    # Paths and image sources compare by identity, so each capture's own
    # objects are compared by points and by repr, addresses masked.
    def plain(drawing: CapturedDrawing) -> tuple[object, ...]:
        path = drawing.path
        points = None if path is None else [s.points for s in path.subpaths]
        rest = repr(replace(drawing, graphics_soft_mask=None, path=None))
        return (ADDRESS.sub("0x", rest), points)

    assert [plain(d) for d in drawn.drawings] == [plain(d) for d in extracted.drawings]
