# SPDX-License-Identifier: AGPL-3.0-only
"""Contemporary Type 3 render-mode rules preserve advance and the active clip."""

import numpy
import pytest

from core_pdf import PdfDocument


def internal_stream(content: bytes) -> bytes:
    return f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"


def internal_document(version: str, render_mode: int) -> bytes:
    content = (
        b"1 1 1 rg 0 0 32 24 re f q 2 0 28 24 re W n "
        + f"BT /F 8 Tf {render_mode} Tr 1 0 0 1 4 4 Tm (A) Tj (B) Tj ET ".encode()
        + b"0 1 0 rg 0 0 32 2 re f Q 0 0 1 rg 0 20 32 2 re f"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 24] "
        b"/Resources << /Font << /F 5 0 R >> >> /Contents 4 0 R >>",
        internal_stream(content),
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] "
        b"/FontMatrix [0.001 0 0 0.001 0 0] /FirstChar 65 /LastChar 66 "
        b"/Widths [1000 1000] /Encoding << /Type /Encoding /Differences [65 /A /B] >> "
        b"/CharProcs << /A 6 0 R /B 7 0 R >> /Resources << >> >>",
        internal_stream(b"1000 0 d0 1 0 0 rg 0 0 1000 1000 re f"),
        internal_stream(b"1000 0 d0 0 0 1 rg 0 0 1000 1000 re f"),
    ]
    data = f"%PDF-{version}\n".encode()
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


@pytest.mark.parametrize("version", ["1.7", "2.0"])
@pytest.mark.parametrize("render_mode", range(8))
def test_type3_rendering_modes_share_paint_advance_and_no_clipping_rules(
    version: str, render_mode: int
) -> None:
    # ISO 32000-2 9.3.6: modes 3/7 suppress Type 3 painting, while all
    # other modes execute CharProcs; none of them adds glyph-based clipping.
    # This is the contemporary shared correction to ISO 32000-1's wording.
    with PdfDocument(internal_document(version, render_mode)) as document:
        page = document.pages[0]
        pixels = page.render().rasterize().array()
        paints = render_mode not in {3, 7}
        expected = numpy.full((24, 32, 4), 255, dtype=numpy.uint8)
        if paints:
            expected[12:20, 4:12] = [255, 0, 0, 255]
            expected[12:20, 12:20] = [0, 0, 255, 255]
        # The green band remains constrained only by the original path clip;
        # the blue band after Q verifies restoration of that parent clip.
        expected[22:24, 2:30] = [0, 255, 0, 255]
        expected[2:4] = [0, 0, 255, 255]
        numpy.testing.assert_array_equal(pixels, expected)

        assert "".join(run.text for run in page.chars) == "AB"
        first, second = page.get_page_program().glyphs
        assert first.visible is second.visible is paints
        assert first.advance_bbox == pytest.approx((4, 4, 12, 12))
        assert second.advance_bbox == pytest.approx((12, 4, 20, 12))
        assert not first.clip_glyph
        assert not second.clip_glyph
