# SPDX-License-Identifier: AGPL-3.0-only

"""Capture skips the rasterizer's per-glyph payload when the caller only wants
text. The danger in that is silent: a program captured without it has no glyph
transforms, so every glyph reports no paint and the page renders blank. These
pin the two halves -- that the flag is recorded on the program, and that
compose_page refuses such a program instead of drawing nothing."""

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.capture.program import PageProgram
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page


def test_a_program_defaults_to_carrying_render_detail() -> None:
    assert PageProgram().render_details is True


class UnreadPage:
    """A page whose every member fails, so reading any of them fails the test."""

    width = height = 0.0
    media_box = None

    def get_page_program(self, *, fields=None, annotations=None):
        raise AssertionError("read the page")

    def get_fields(self):
        raise AssertionError("read the page")

    def get_annotations(self):
        raise AssertionError("read the page")

    def resolve_transparency_group_alpha(self):
        raise AssertionError("read the page")


def test_compose_page_refuses_a_text_only_program() -> None:
    with pytest.raises(ValueError, match="render_details=True"):
        # A page-shaped stub: the check runs before anything is read off it,
        # which is the point of putting it first.
        compose_page(UnreadPage(), RenderOptions(), page_program=PageProgram(render_details=False))


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


@pytest.mark.parametrize("want_transform", [True, False])
def test_a_vertical_run_honours_the_render_details_gate(want_transform: bool) -> None:
    """The vertical path used to build a transform tuple per glyph whatever the
    flag said, so a text-only capture of a vertical page still carried the
    render payload the flag exists to leave out."""
    from core_pdf.impl.capture.glyph_geometry import vertical_glyph_geometry

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
