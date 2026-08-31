from core_pdf.impl.model.glyphs import GlyphObservation
from core_pdf.impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.capture import (
    apply_glyph_geometry_to_run,
    glyph_bitmap_dimensions,
)


def internal_run() -> TextRun:
    return TextRun("AB", 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 12.0, 4.0, 0, 0, 0)


def test_apply_glyph_geometry_reduces_bounds_and_confidence() -> None:
    run = internal_run()
    glyphs = (
        GlyphObservation("A", (1.0, 2.0, 3.0, 5.0), (0.0, 1.0, 4.0, 6.0), 0, confidence=0.9),
        GlyphObservation("B", (-2.0, 3.0, 7.0, 8.0), (-1.0, -3.0, 6.0, 7.0), 1, confidence=0.7),
    )

    apply_glyph_geometry_to_run(run, glyphs)

    assert run.advance_bbox == (-1.0, -3.0, 6.0, 7.0)
    assert run.ink_bbox == (-2.0, 2.0, 7.0, 8.0)
    assert run.confidence == 0.7


def test_apply_empty_glyph_geometry_preserves_run_bounds() -> None:
    run = internal_run()
    original = (run.advance_bbox, run.ink_bbox, run.confidence)

    apply_glyph_geometry_to_run(run, ())

    assert (run.advance_bbox, run.ink_bbox, run.confidence) == original


def test_glyph_bitmap_dimensions_derive_from_bbox_aspect_and_font_size() -> None:
    assert glyph_bitmap_dimensions((0.0, -0.2, 0.6, 0.8), 12.0) == (18, 30)


def test_glyph_bitmap_dimensions_preserves_degenerate_fallback() -> None:
    assert glyph_bitmap_dimensions((1.0, 1.0, 1.0, 2.0), 12.0) == (24, 32)
