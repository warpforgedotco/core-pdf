# SPDX-License-Identifier: AGPL-3.0-only
"""A Form owns its graphics saves and clipping until the invocation returns.

qpdf 12.3.2 --check accepts these PDF containers. Poppler 26.07.0 pdftoppm -r 72
shows a red 10x10 rectangle for all four Form endings, with a diagnostic for the
unmatched Q and unterminated array. Verified before pinning the raster sample.
"""

import pytest

from core_pdf import PdfDocument
from tests.helpers.pdf_bytes import one_page_pdf, stream_obj


@pytest.mark.parametrize(
    "child",
    [b"Q", b"q 0 0 2 2 re W n", b"0 0 2 2 re W n", b"q 0 0 2 2 re W n ["],
    ids=["unmatched-restore", "unclosed-clip-save", "clip-without-save", "clip-parse-error"],
)
def test_form_state_cannot_consume_parent_saves_or_clip_later_page_paint(child: bytes) -> None:
    pdf = one_page_pdf(
        b"1 0 0 rg q 0 1 0 rg /Child Do Q 0 0 10 10 re f",
        resources=b"<< /XObject << /Child 6 0 R >> >>",
        extra_objects=[stream_obj(child, b"/Type /XObject /Subtype /Form /BBox [0 0 100 100]")],
    )
    with PdfDocument(pdf) as document:
        page = document.pages[0]
        program = page.get_page_program()
        assert program.drawings[-1].fill == (1.0, 0.0, 0.0)
        kinds = [drawing.kind for drawing in program.drawings]
        assert kinds.count("state-push") == kinds.count("state-pop")
        raster = page.render().rasterize()

    offset = (787 * raster.width + 5) * raster.channels
    assert bytes(raster.pixels)[offset : offset + 4] == b"\xff\x00\x00\xff"
