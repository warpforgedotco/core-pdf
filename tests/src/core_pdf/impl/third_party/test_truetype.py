from core_pdf.impl.third_party.truetype import TrueTypeFontProgram


def test_explicit_cid_to_gid_map_returns_notdef_outside_stream() -> None:
    font = object.__new__(TrueTypeFontProgram)
    font.cid_to_gid = b"\x00\x07"

    assert font.glyph_id_for_code(0) == 7
    assert font.glyph_id_for_code(1) == 0
