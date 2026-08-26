"""Security bounds for embedded Type 1 font programs."""

from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_09_fonts import font_program_type1


def test_sparse_type1_subroutine_index_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    private = b"dup 4096 1 RD x"
    monkeypatch.setattr(font_program_type1, "internal_eexec_payload", lambda *_args: private)

    with pytest.raises(ValueError, match="subroutine index exceeds decoder limit"):
        font_program_type1.Type1FontProgram(b"embedded font")


def test_unknown_glyph_name_requires_an_available_notdef() -> None:
    program = object.__new__(font_program_type1.Type1FontProgram)
    program.glyph_name_to_id = {"visible": 0, ".notdef": 1}

    assert program.glyph_id_for_name("visible") == 0
    assert program.glyph_id_for_name("missing") == 1

    program.glyph_name_to_id = {"visible": 0}

    assert program.glyph_id_for_name("missing") is None
