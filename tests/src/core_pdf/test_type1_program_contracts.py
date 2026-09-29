import pytest

from core_pdf.impl import fonts_program_type1
from core_pdf.impl.fonts_program_type1 import Type1FontProgram

SQUARE = bytes((139, 248, 136, 13, 239, 139, 21, 247, 192, 139, 5, 139, 248, 36, 5, 9, 14))
LEN_IV = 4


def encrypt(plain: bytes, key: int) -> bytes:
    output = bytearray()
    state = key
    for byte in plain:
        cipher = byte ^ (state >> 8)
        output.append(cipher)
        state = ((cipher + state) * 52845 + 22719) & 0xFFFF
    return bytes(output)


def type1_program(len_iv: int | None) -> bytes:
    charstring = SQUARE if len_iv == -1 else encrypt(bytes(LEN_IV) + SQUARE, 4330)
    private = b"dup /Private 8 dict dup begin "
    if len_iv is not None:
        private += b"/lenIV %d def " % len_iv
    private += b"/Subrs 0 array end /CharStrings 1 dict dup begin "
    private += b"/square %d RD " % len(charstring) + charstring + b" ND end"
    clear = (
        b"%!FontType1-1.0: Test\n/FontName /Test def /FontMatrix [0.001 0 0 0.001 0 0] def\n"
        b"currentfile eexec\n"
    )
    return clear + encrypt(bytes(4) + private, 55665)


@pytest.mark.parametrize("len_iv", [None, 4, -1])
def test_every_lenIV_decodes_the_same_outline(len_iv: int | None) -> None:
    program = Type1FontProgram(type1_program(len_iv))
    glyph_id = program.glyph_id_for_name("square")
    assert glyph_id is not None
    reference = Type1FontProgram(type1_program(None))
    contours = program.normalized_glyph_contours(glyph_id)
    assert contours
    assert contours == reference.normalized_glyph_contours(glyph_id)


@pytest.mark.parametrize("len_iv", [None, 4, -1])
def test_compiled_bounds_match_the_fonttools_drawing(monkeypatch, len_iv: int | None) -> None:
    program = Type1FontProgram(type1_program(len_iv))
    glyph_id = program.glyph_id_for_name("square")
    assert glyph_id is not None
    compiled = program.glyph_bbox_for_gid(glyph_id)
    assert compiled is not None
    monkeypatch.setattr(fonts_program_type1, "type1_glyph_bounds", lambda *args: None)
    drawn = Type1FontProgram(type1_program(len_iv)).glyph_bbox_for_gid(glyph_id)
    assert drawn is not None
    assert [float(v).hex() for v in compiled] == [float(v).hex() for v in drawn]
