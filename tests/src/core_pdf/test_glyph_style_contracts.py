"""A glyph observation shares its operation's style with the operation's other glyphs."""

from copy import replace

import pytest

from core_pdf.impl.glyphs import STYLE_FIELDS, GlyphObservation, GlyphStyle

BOX = (0.0, 0.0, 1.0, 1.0)


def style(**changes: object) -> GlyphStyle:
    values: dict[str, object] = {
        "font_size": 12.0,
        "rotation_angle": 0,
        "fill": (0.0, 0.0, 0.0),
        "font_decoder": None,
        "effective_font_size": 12.0,
        "effective_font_height": 12.0,
        "provenance": (("source", "native_text"),),
        "text_render_mode": 0,
        "fill_opacity": 1.0,
        "stroke_color": None,
        "stroke_opacity": 1.0,
        "line_width": 1.0,
        "blend_mode": None,
        "soft_mask_alpha": None,
        "text_object_id": 3,
        "line_cap": 0,
        "line_join": 0,
        "dash_pattern": None,
        "clip_glyph": False,
        "alpha_is_shape": False,
        "paint_from_program": False,
        "graphics_soft_mask": None,
    }
    values.update(changes)
    return GlyphStyle(**values)


def styled(shared: GlyphStyle, text: str) -> GlyphObservation:
    return GlyphObservation.styled(
        shared,
        text,
        BOX,
        BOX,
        1,
        text.encode(),
        ord(text),
        None,
        None,
        "Helvetica",
        None,
        True,
        1.0,
        "to_unicode",
        (),
        (),
        0,
        0,
        None,
        None,
        True,
        (1, 0),
    )


def test_every_style_field_reads_through_to_the_shared_style() -> None:
    shared = style()
    glyph = styled(shared, "a")
    for name in STYLE_FIELDS:
        assert getattr(glyph, name) == getattr(shared, name)


def test_setting_a_style_field_changes_only_that_glyph() -> None:
    shared = style()
    first, second = styled(shared, "a"), styled(shared, "b")
    first.fill = (1.0, 0.0, 0.0)
    assert first.fill == (1.0, 0.0, 0.0)
    assert second.fill == (0.0, 0.0, 0.0)
    assert shared.fill == (0.0, 0.0, 0.0)
    assert second.style is shared


def test_replace_takes_style_and_own_fields_together() -> None:
    glyph = styled(style(), "a")
    copied = replace(glyph, text="b", font_size=9.0)
    assert (copied.text, copied.font_size) == ("b", 9.0)
    assert (glyph.text, glyph.font_size) == ("a", 12.0)
    assert copied.font_name == glyph.font_name
    with pytest.raises(TypeError, match="unexpected keyword"):
        replace(glyph, not_a_field=1)


def test_the_keyword_constructor_still_accepts_every_field() -> None:
    glyph = GlyphObservation("a", BOX, BOX, 1, font_size=7.0, fill=(1.0,), text_object_id=4)
    assert (glyph.font_size, glyph.fill, glyph.text_object_id) == (7.0, (1.0,), 4)
    assert repr(glyph).startswith("GlyphObservation(text='a', ")
    assert "font_size=7.0" in repr(glyph)
    assert "style=" not in repr(glyph)
