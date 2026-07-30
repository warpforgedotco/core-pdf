from typing import Protocol

from core_pdf.impl.engine.spec.s_09_fonts.truetype import TrueTypeFontProgram


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
    font.font = fake_font
    font.internal_glyph_set = None
    font.internal_glyph_contour_cache = {}

    first = font.glyph_contours(0)
    second = font.glyph_contours(0)

    assert first == second
    assert first is not second
    assert fake_font.glyph_set_calls == 1
    assert len(font.internal_glyph_contour_cache) == 1
