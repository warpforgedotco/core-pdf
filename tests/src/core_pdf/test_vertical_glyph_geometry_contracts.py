# SPDX-License-Identifier: AGPL-3.0-only

"""Vertical writing keeps a scalar geometry pass; the compiled kernel models
only horizontal runs. These drive it directly, because a vertical run that is
also clipped is rare enough in the corpus that the clip branch would otherwise
go unexercised."""

from core_pdf.impl.capture.glyph_geometry import GlyphGeometry, vertical_glyph_geometry
from core_pdf.impl.types import Rectangle

BASIS = (100.0, 700.0, 12.0, 0.0, 0.0, 12.0)


def run(
    *,
    clip_primary: Rectangle | None = None,
    clip_page: Rectangle | None = None,
    visible: bool = True,
    want_bitmap: tuple[int, int] = (1, 1),
) -> GlyphGeometry:
    # Spelled out rather than splatted from a dict: a **kwargs dict widens
    # every value to a union and the checkers cannot match them to parameters.
    return vertical_glyph_geometry(
        [0.0, 12.0],
        [12.0, 12.0],
        [(0.0, 0.0), (0.0, 0.0)],
        basis=BASIS,
        font_ascent=0.88,
        font_descent=-0.12,
        rise=0.0,
        font_scale=0.012,
        advance_scale=0.012,
        clip_primary=clip_primary,
        clip_page=clip_page,
        visible=visible,
        want_bitmap=list(want_bitmap),
    )


def test_unclipped_vertical_glyphs_are_visible_with_default_bitmaps() -> None:
    advance, baseline, transform, ink, vis, bitmap = run()
    assert vis == [1, 1]
    assert ink == advance, "a vertical glyph has no font bbox, so ink is the advance box"
    assert len(baseline) == 2
    # Vertical always builds a transform: only the horizontal kernel skips it.
    assert all(t is not None and len(t) == 6 for t in transform)
    # The kernel falls back to the default bitmap for a missing bbox.
    assert bitmap == [24, 32, 24, 32]


def test_a_glyph_outside_the_primary_clip_is_not_visible() -> None:
    advance, *_rest, vis, _bitmap = run(clip_primary=(-500.0, -500.0, -400.0, -400.0))
    assert vis == [0, 0]


def test_a_glyph_outside_the_page_clip_is_not_visible() -> None:
    *_rest, vis, _bitmap = run(clip_page=(-500.0, -500.0, -400.0, -400.0))
    assert vis == [0, 0]


def test_a_clip_that_contains_the_run_leaves_it_visible() -> None:
    *_rest, vis, _bitmap = run(clip_primary=(-1000.0, -1000.0, 1000.0, 1000.0))
    assert vis == [1, 1]


def test_an_invisible_run_stays_invisible_whatever_the_clip() -> None:
    *_rest, vis, _bitmap = run(visible=False, clip_primary=(-1000.0, -1000.0, 1000.0, 1000.0))
    assert vis == [0, 0]


def test_bitmap_dimensions_are_only_filled_where_asked() -> None:
    *_rest, bitmap = run(want_bitmap=(0, 1))
    assert bitmap == [0, 0, 24, 32]
