# SPDX-License-Identifier: AGPL-3.0-only

"""Capture skips the rasterizer's per-glyph payload when the caller only wants
text. The danger in that is silent: a program captured without it has no glyph
transforms, so every glyph reports no paint and the page renders blank. These
pin the two halves -- that the flag is recorded on the program, and that
compose_page refuses such a program instead of drawing nothing."""

from types import SimpleNamespace

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.program import PageProgram
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page


def test_a_program_defaults_to_carrying_render_detail() -> None:
    assert PageProgram().render_details is True


def test_compose_page_refuses_a_text_only_program() -> None:
    with pytest.raises(ValueError, match="render_details=True"):
        # A page-shaped stub: the check runs before anything is read off it,
        # which is the point of putting it first.
        stub = SimpleNamespace(width=0.0, height=0.0, media_box=None, user_unit=1.0)
        compose_page(stub, RenderOptions(), page_program=PageProgram(render_details=False))


def test_extraction_captures_without_render_detail(text_pdf_bytes: bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        text_only = page.get_page_program(render_details=False)
        full = page.get_page_program()

    assert text_only.render_details is False
    assert full.render_details is True
    # Same text, same boxes: only the rasterizer's payload differs.
    assert [glyph.text for glyph in text_only.glyphs] == [glyph.text for glyph in full.glyphs]
    assert [glyph.advance_bbox for glyph in text_only.glyphs] == [
        glyph.advance_bbox for glyph in full.glyphs
    ]
    assert all(glyph.glyph_transform is None for glyph in text_only.glyphs)
    assert any(glyph.glyph_transform is not None for glyph in full.glyphs)
    assert all(glyph.bitmap_code is None for glyph in text_only.glyphs)


def test_rendering_still_works_from_a_full_capture(text_pdf_bytes: bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        program = page.get_page_program()
        rendered = compose_page(page, RenderOptions(), page_program=program)
    assert rendered is not None


def test_a_text_only_capture_splits_glyphs_the_same_way(text_pdf_bytes: bytes) -> None:
    """The flag is about the render payload, not about what the text is.

    `suspicious` decides whether a multi-character glyph is split into one
    observation per character. It once hung off the render flag, which split
    "A/B" into three observations for a text-only capture and left it as one
    otherwise -- a silent extraction difference from a rendering switch.
    """
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        text_only = page.get_page_program(render_details=False)
        full = page.get_page_program()

    assert len(text_only.glyphs) == len(full.glyphs)
    assert [glyph.text for glyph in text_only.glyphs] == [glyph.text for glyph in full.glyphs]
    assert [glyph.cluster_id for glyph in text_only.glyphs] == [
        glyph.cluster_id for glyph in full.glyphs
    ]
