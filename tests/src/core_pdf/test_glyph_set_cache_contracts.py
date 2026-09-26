"""A font's glyph set is built once per thread, not once per glyph drawn."""

import threading

from core_pdf.impl.fonts.font_program import glyph_set_of


class CountingFont:
    def __init__(self) -> None:
        self.built: list[object] = []

    def getGlyphSet(self) -> object:  # noqa: N802 -- the fontTools name
        glyph_set = object()
        self.built.append(glyph_set)
        return glyph_set


def test_a_thread_reuses_its_glyph_set() -> None:
    font = CountingFont()
    assert glyph_set_of(font) is glyph_set_of(font)
    assert len(font.built) == 1


def test_another_thread_gets_its_own() -> None:
    # Drawing tracks composite depth on the set, so threads must not share one.
    font = CountingFont()
    here = glyph_set_of(font)
    there: list[object] = []
    worker = threading.Thread(target=lambda: there.append(glyph_set_of(font)))
    worker.start()
    worker.join()
    assert there[0] is not here
    assert len(font.built) == 2


def test_fonts_do_not_share_glyph_sets() -> None:
    first, second = CountingFont(), CountingFont()
    assert glyph_set_of(first) is not glyph_set_of(second)
