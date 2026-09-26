# SPDX-License-Identifier: AGPL-3.0-only

"""Capture skips the rasterizer's per-glyph payload when the caller only wants
text. The danger in that is silent: a program captured without it has no glyph
transforms, so every glyph reports no paint and the page renders blank. These
pin the two halves -- that the options are recorded on the program, and that
its commands, the only thing a renderer draws from, refuse such a program
instead of drawing nothing."""

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture_program import (
    EXTRACTION_CAPTURE,
    CapturedProgram,
    CaptureOptions,
    PageProgram,
)
from core_pdf.impl.render_model import RenderOptions
from core_pdf.impl.render_page import compose_page


def test_a_program_defaults_to_carrying_render_detail() -> None:
    assert PageProgram().options == CaptureOptions()
    assert PageProgram().options.render_details is True


def test_a_text_only_program_refuses_to_give_commands() -> None:
    with pytest.raises(ValueError, match="render_details=True"):
        _ = PageProgram(CapturedProgram(options=EXTRACTION_CAPTURE)).commands


def test_compose_page_refuses_a_text_only_program(text_pdf_bytes: bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        program = page.get_page_program(options=EXTRACTION_CAPTURE)
        with pytest.raises(ValueError, match="render_details=True"):
            compose_page(page, RenderOptions(), page_program=program)


def test_extraction_captures_without_render_detail(text_pdf_bytes: bytes) -> None:
    with PdfDocument(text_pdf_bytes) as document:
        page = document.pages[0]
        text_only = page.get_page_program(options=EXTRACTION_CAPTURE)
        full = page.get_page_program()

    assert text_only.options.render_details is False
    assert full.options.render_details is True
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
        text_only = page.get_page_program(options=EXTRACTION_CAPTURE)
        full = page.get_page_program()

    assert len(text_only.glyphs) == len(full.glyphs)
    assert [glyph.text for glyph in text_only.glyphs] == [glyph.text for glyph in full.glyphs]
    assert [glyph.cluster_id for glyph in text_only.glyphs] == [
        glyph.cluster_id for glyph in full.glyphs
    ]


@pytest.mark.parametrize("want_transform", [True, False])
def test_a_vertical_run_honours_the_render_details_gate(want_transform: bool) -> None:
    """The vertical path used to build a transform tuple per glyph whatever the
    flag said, so a text-only capture of a vertical page still carried the
    render payload the flag exists to leave out."""
    from core_pdf.impl.capture_glyph_geometry import vertical_glyph_geometry

    _advance, _baseline, transforms, _ink, _visible, _bitmap = vertical_glyph_geometry(
        [0.0, 12.0],
        [12.0, 12.0],
        [(0.0, 0.0), (0.0, -12.0)],
        basis=(100.0, 700.0, 1.0, 0.0, 0.0, 1.0),
        font_ascent=0.88,
        font_descent=-0.12,
        rise=0.0,
        font_scale=12.0,
        advance_scale=12.0,
        clip_primary=None,
        clip_page=None,
        visible=True,
        want_bitmap=[0, 0],
        want_transform=want_transform,
    )
    assert len(transforms) == 2
    if want_transform:
        assert all(t is not None and len(t) == 6 for t in transforms)
    else:
        assert transforms == [None, None]
