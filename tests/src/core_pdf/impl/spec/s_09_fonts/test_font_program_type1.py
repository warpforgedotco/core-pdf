"""Security bounds for embedded Type 1 font programs."""

from __future__ import annotations

import pytest

from core_pdf.impl.spec.s_09_fonts import font_program_type1


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


def test_type1_binary_entries_preserve_duplicates_and_skip_truncated_payloads() -> None:
    data = b"/A 1 RD x /B 0 -|  /A 1 RD y /truncated 9 RD z"

    entries = list(
        font_program_type1.internal_binary_entries(data, font_program_type1.internal_CHARSTRING_RE)
    )

    assert entries == [(b"A", b"x"), (b"B", b""), (b"A", b"y")]
    assert list(dict(entries).items()) == [(b"A", b"y"), (b"B", b"")]


@pytest.mark.parametrize(
    ("len_iv", "expected"), [(-1, b"abcd\x0e"), (0, b"abcd\x0e"), (4, b"\x0e")]
)
def test_type1_subroutines_and_glyphs_share_charstring_preparation(
    monkeypatch: pytest.MonkeyPatch, len_iv: int, expected: bytes
) -> None:
    private = f"/lenIV {len_iv} def ".encode() + b"dup 1 5 RD abcd\x0e /A 5 RD abcd\x0e"
    decrypt_keys: list[int] = []

    def decrypt(data: bytes, key: int) -> bytes:
        decrypt_keys.append(key)
        return data

    monkeypatch.setattr(font_program_type1, "internal_eexec_payload", lambda *_: private)
    monkeypatch.setattr(font_program_type1, "internal_decrypt", decrypt)
    program = font_program_type1.Type1FontProgram(b"font")

    assert decrypt_keys == [4330, 4330]
    assert program.charstrings["A"].bytecode == expected
    assert program.subrs[1].bytecode == expected
    assert all(subr.subrs is program.subrs for subr in program.subrs)
    assert program.charstrings["A"].subrs is program.subrs
    assert program.subrs[0].bytecode == b"\x0b"
