from typing import cast

import core_pdf.impl.model.glyphs as glyphs_module
from core_pdf.impl.model.glyphs import glyph_unicode_confidence


def test_confidence_is_memoised_and_bounded(monkeypatch) -> None:
    monkeypatch.setattr(glyphs_module, "CONFIDENCE_CACHE", {})
    monkeypatch.setattr(glyphs_module, "CONFIDENCE_CACHE_LIMIT", 2)
    first = glyph_unicode_confidence("a", "tounicode", ())
    assert glyph_unicode_confidence("a", "tounicode", ()) == first
    assert len(glyphs_module.CONFIDENCE_CACHE) == 1
    glyph_unicode_confidence("b", "tounicode", ())
    glyph_unicode_confidence("c", "tounicode", ())
    assert len(glyphs_module.CONFIDENCE_CACHE) == 1
    assert {("c", "tounicode", ()): first} == glyphs_module.CONFIDENCE_CACHE


def test_unhashable_alternates_bypass_the_cache(monkeypatch) -> None:
    monkeypatch.setattr(glyphs_module, "CONFIDENCE_CACHE", {})
    value = glyph_unicode_confidence("a", "tounicode", cast(tuple[str, ...], ["x"]))
    assert value == glyph_unicode_confidence("a", "tounicode", ("x",))
    assert list(glyphs_module.CONFIDENCE_CACHE) == [("a", "tounicode", ("x",))]
