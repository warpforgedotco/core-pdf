from core_pdf.impl._impl.fonts.helpers import (
    MAC_ROMAN_ENCODING_TABLE,
    STANDARD_ENCODING_TABLE,
    WIN_ANSI_ENCODING_TABLE,
)
from core_pdf.impl.spec.s_09_fonts.data.base_encodings import STANDARD_ENCODING


def test_decode_tables_expand_ligatures_and_leave_undefined_codes_empty() -> None:
    assert STANDARD_ENCODING[0xAE] == "ﬁ"
    assert STANDARD_ENCODING_TABLE[0xAE] == "fi"
    assert MAC_ROMAN_ENCODING_TABLE[0xDE] == "fi"
    assert STANDARD_ENCODING_TABLE[0xA0] == ""


def test_decode_tables_keep_raw_values_for_the_control_range() -> None:
    for table in (STANDARD_ENCODING_TABLE, WIN_ANSI_ENCODING_TABLE, MAC_ROMAN_ENCODING_TABLE):
        assert table[0x00] == "\x00"
        assert table[0x0C] == "\x0c"
