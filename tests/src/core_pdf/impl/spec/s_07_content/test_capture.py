from core_pdf.impl.spec.s_07_content.capture import glyph_bitmap_dimensions


def test_glyph_bitmap_dimensions_derive_from_bbox_aspect_and_font_size() -> None:
    assert glyph_bitmap_dimensions((0.0, -0.2, 0.6, 0.8), 12.0) == (18, 30)


def test_glyph_bitmap_dimensions_preserves_degenerate_fallback() -> None:
    assert glyph_bitmap_dimensions((1.0, 1.0, 1.0, 2.0), 12.0) == (24, 32)
