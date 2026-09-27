from core_pdf.impl import glyphs
from core_pdf.impl.caches import BoundedDict
from core_pdf.impl.glyphs import glyph_unicode_confidence


def test_confidence_is_memoised_and_bounded(monkeypatch) -> None:
    monkeypatch.setattr(glyphs, "CONFIDENCE_CACHE", BoundedDict(2))
    first = glyph_unicode_confidence("a", "tounicode", ())
    assert glyph_unicode_confidence("a", "tounicode", ()) == first
    assert len(glyphs.CONFIDENCE_CACHE) == 1
    glyph_unicode_confidence("b", "tounicode", ())
    glyph_unicode_confidence("c", "tounicode", ())
    assert len(glyphs.CONFIDENCE_CACHE) == 1
    assert {("c", "tounicode", ()): first} == glyphs.CONFIDENCE_CACHE


def test_unhashable_alternates_bypass_the_cache(monkeypatch) -> None:
    monkeypatch.setattr(glyphs, "CONFIDENCE_CACHE", BoundedDict(8))
    value = glyph_unicode_confidence("a", "tounicode", (["x"]))  # ty: ignore[invalid-argument-type]
    assert value == glyph_unicode_confidence("a", "tounicode", ("x",))
    assert list(glyphs.CONFIDENCE_CACHE) == [("a", "tounicode", ("x",))]
