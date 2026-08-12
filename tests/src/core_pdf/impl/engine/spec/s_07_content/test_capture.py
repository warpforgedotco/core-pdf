from core_pdf.impl.engine.layout.glyphs import GlyphObservation
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.spec.s_07_content.capture import (
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


def test_glyph_bitmap_dimensions_reuses_semantically_identical_geometry() -> None:
    glyph_bitmap_dimensions.cache_clear()
    bbox = (0.0, -0.2, 0.6, 0.8)

    assert glyph_bitmap_dimensions(bbox, 12.0) == (18, 30)
    assert glyph_bitmap_dimensions(bbox, 12.0) == (18, 30)

    cache_info = glyph_bitmap_dimensions.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_glyph_bitmap_dimensions_preserves_degenerate_fallback() -> None:
    assert glyph_bitmap_dimensions((1.0, 1.0, 1.0, 2.0), 12.0) == (24, 32)
