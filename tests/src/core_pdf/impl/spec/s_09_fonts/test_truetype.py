from typing import Any, Protocol, cast

from core_pdf.impl.spec.s_09_fonts.font_program_truetype import TrueTypeFontProgram


def test_corrupt_post_table_uses_synthetic_glyph_order() -> None:
    from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
        internal_ensure_glyph_order,
    )

    class Maxp:
        numGlyphs = 3

    class Font:
        glyph_order: list[str] | None = None

        def getGlyphOrder(self) -> list[str]:
            if self.glyph_order is None:
                raise ValueError("unsupported post table")
            return self.glyph_order

        def __getitem__(self, key: str) -> Maxp:
            assert key == "maxp"
            return Maxp()

        def setGlyphOrder(self, glyph_order: list[str]) -> None:
            self.glyph_order = glyph_order

    font = Font()

    internal_ensure_glyph_order(cast(Any, font))

    assert font.glyph_order == [".notdef", "glyph00001", "glyph00002"]


class internal_Pen(Protocol):
    def moveTo(self, point: tuple[int, int]) -> None: ...

    def lineTo(self, point: tuple[int, int]) -> None: ...

    def closePath(self) -> None: ...


def test_explicit_cid_to_gid_map_returns_notdef_outside_stream() -> None:
    font = object.__new__(TrueTypeFontProgram)
    font.cid_to_gid = b"\x00\x07"

    assert font.glyph_id_for_code(0) == 7
    assert font.glyph_id_for_code(1) == 0


def test_glyph_contours_reuses_glyph_set_and_returns_fresh_lists() -> None:
    class FakeGlyph:
        def draw(self, pen: internal_Pen) -> None:
            pen.moveTo((0, 0))
            pen.lineTo((100, 0))
            pen.lineTo((100, 100))
            pen.closePath()

    class FakeFont:
        glyph_set_calls = 0

        def getGlyphName(self, gid: int) -> str:
            assert gid == 0
            return "triangle"

        def getGlyphSet(self) -> dict[str, FakeGlyph]:
            self.glyph_set_calls += 1
            return {"triangle": FakeGlyph()}

    font = object.__new__(TrueTypeFontProgram)
    fake_font = FakeFont()
    font.font = cast(Any, fake_font)
    font.internal_glyph_set = None
    font.internal_glyph_contour_cache = {}

    first = font.glyph_contours(0)
    second = font.glyph_contours(0)

    assert first == second
    assert first is not second
    assert fake_font.glyph_set_calls == 1
    assert len(font.internal_glyph_contour_cache) == 1


def test_glyph_bbox_uses_scaled_glyf_bounds_without_decomposing_outline() -> None:
    class Glyph:
        numberOfContours = -1
        xMin = -20
        yMin = -40
        xMax = 600
        yMax = 1400

    class Font:
        def getGlyphName(self, gid: int) -> str:
            assert gid == 7
            return "composite"

        def __getitem__(self, key: str) -> dict[str, Glyph]:
            assert key == "glyf"
            return {"composite": Glyph()}

    font = object.__new__(TrueTypeFontProgram)
    font.cid_to_gid = b"\x00\x07"
    font.cmap = {}
    font.font = cast(Any, Font())
    font.units_per_em = 2000.0
    font.internal_glyph_contour_cache = {}

    assert font.glyph_bbox(0) == (-10.0, -20.0, 300.0, 700.0)
    assert font.internal_glyph_contour_cache == {}


def test_glyph_bbox_returns_none_for_empty_or_malformed_glyph() -> None:
    class EmptyGlyph:
        numberOfContours = 0

    class Font:
        def __init__(self, glyph: object) -> None:
            self.glyph = glyph

        def getGlyphName(self, gid: int) -> str:
            assert gid == 0
            return "empty"

        def __getitem__(self, key: str) -> dict[str, object]:
            assert key == "glyf"
            return {"empty": self.glyph}

    font = object.__new__(TrueTypeFontProgram)
    font.cid_to_gid = None
    font.cmap = {}
    font.units_per_em = 1000.0
    font.font = cast(Any, Font(EmptyGlyph()))

    assert font.glyph_bbox(0) is None

    font.font = cast(Any, Font(object()))

    assert font.glyph_bbox(0) is None


def test_corrupt_font_tables_do_not_escape_as_assertion_errors() -> None:
    """fontTools guards malformed tables with bare `assert`, not just raises.

    A damaged embedded font must degrade to "no usable font program" rather
    than aborting the caller: a real document was left entirely unextractable
    because a corrupt cmap format 4 subtable raised AssertionError out through
    the content interpreter.
    """
    from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
        internal_best_unicode_gid_cmap,
    )
    from core_pdf.impl.spec.s_09_fonts.ligatures import load_ligature_font_tables

    class AssertingFont:
        def __getitem__(self, key: str) -> object:
            raise AssertionError(f"corrupt {key} table")

    assert internal_best_unicode_gid_cmap(cast(Any, AssertingFont())) == {}

    class AssertingCmapFont:
        def __getitem__(self, key: str) -> object:
            class Table:
                def getBestCmap(self) -> object:
                    raise AssertionError("corrupt cmap subtable")

            return Table()

    assert internal_best_unicode_gid_cmap(cast(Any, AssertingCmapFont())) == {}

    # The whole-program entry point degrades rather than propagating.
    assert load_ligature_font_tables(b"not a font at all") is None
    assert load_ligature_font_tables(b"\x00\x01\x00\x00" + b"\xff" * 64) is None


def test_symbol_bracket_pieces_map_to_unicode_not_private_use() -> None:
    """The AGL puts the Symbol font's bracket pieces in Adobe's private-use
    area, but Unicode 3.2 gave them real codepoints. A private-use character
    renders as a blank box and means nothing to a downstream consumer, so
    extraction should prefer the standard ones.
    """
    from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode

    assert glyph_name_to_unicode("bracketlefttp") == "⎡"
    assert glyph_name_to_unicode("bracketleftex") == "⎢"
    assert glyph_name_to_unicode("bracketrightbt") == "⎦"
    assert glyph_name_to_unicode("parenleftex") == "⎜"
    assert glyph_name_to_unicode("braceex") == "⎪"
    assert glyph_name_to_unicode("integralex") == "⎮"
    # Sans-serif variants are a typeface distinction, not a different character.
    assert glyph_name_to_unicode("registersans") == "®"
    for name in ("bracketlefttp", "braceleftmid", "arrowvertex", "trademarksans"):
        assert not (0xE000 <= ord(glyph_name_to_unicode(name)) <= 0xF8FF)
