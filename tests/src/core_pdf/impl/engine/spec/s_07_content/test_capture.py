from core_pdf.impl.engine.spec.s_07_content.capture import glyph_bitmap_dimensions


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
