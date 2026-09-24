# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import pytest

from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.metrics import glyph_advance_vector
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph


class Sink:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class Type3Font:
    is_type3 = True
    is_vertical = False
    font_matrix = Matrix(0.001, 0, 0, 0.001, 0, 0)
    font = {"CharProcs": {"A": PdfStream(raw_data=b""), "space": PdfStream(raw_data=b"")}}

    def glyph_name(self, code: int) -> str:
        return "space" if code == 32 else "A"

    def glyph_advance_vector(self, code: int, **kwargs: Any) -> tuple[float, float]:
        return glyph_advance_vector(500, vertical=False, **kwargs)

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]:
        return tuple(
            DecodedFontGlyph(bytes([code]), code, code, code, chr(code), code) for code in data
        )

    def text_advance_vector(self, data: bytes, **kwargs: Any) -> tuple[float, float]:
        kwargs.pop("glyphs")
        return (
            sum(
                self.glyph_advance_vector(code, encoded_space=code == 32, **kwargs)[0]
                for code in data
            ),
            0,
        )


@pytest.mark.parametrize("render_mode", range(8))
def test_type3_invisible_modes_skip_programs_but_keep_width_spacing_and_scale(
    render_mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ContentInterpreter(ObjectResolver(b"", {}), Sink(), None)  # ty: ignore[invalid-argument-type]
    state.graphics.render_mode = render_mode
    state.graphics.font_size = 12
    state.graphics.char_space = 2
    state.graphics.word_space = 3
    state.graphics.horizontal_scale = 50
    state.text_matrix = Matrix(1, 0, 0, 1, 5, 7)
    origins: list[float] = []
    monkeypatch.setattr(
        type(state.stream_executor),
        "consume",
        lambda executor, stream, resources, ctm, depth: origins.append(ctm.e),
    )
    state.append_text(b"A A", decoder=Type3Font())  # ty: ignore[invalid-argument-type]
    assert origins == ([] if render_mode in {3, 7} else [5, 9, 14.5])
    assert state.text_matrix == Matrix(1, 0, 0, 1, 18.5, 7)
