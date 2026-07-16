from core_pdf.impl.engine.layout.glyphs import glyph_unicode_confidence


def test_unicode_confidence_is_independent_of_paint_visibility() -> None:
    assert glyph_unicode_confidence("A", "to_unicode", visible=False) == 1.0


def test_hidden_unsupported_glyph_still_has_low_unicode_confidence() -> None:
    assert glyph_unicode_confidence("\ue000", "to_unicode", visible=False) == 0.20
