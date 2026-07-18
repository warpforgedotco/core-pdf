import pytest

from core_jpeg import JpegParseError, JpegUnsupportedError
from core_jpeg.impl.codecs.dct.bitstream import JpegBitReader


def test_reads_and_peeks_across_byte_boundaries() -> None:
    reader = JpegBitReader(b"\xaa\xcc")

    assert reader.get_bits(4) == 0xA
    assert reader.peek_bits(8) == 0xAC
    assert reader.get_bits(8) == 0xAC
    assert reader.get_bits(4) == 0xC


def test_skips_stuffed_zero_after_ff_byte() -> None:
    reader = JpegBitReader(b"\xff\x00\x80")

    assert reader.get_bits(8) == 0xFF
    assert reader.pos == 2
    assert reader.get_bits(8) == 0x80


def test_missing_bits_are_padded_with_ones() -> None:
    reader = JpegBitReader(b"\xa0")

    assert reader.peek_bits(12) == 0xA0F
    assert reader.get_bits(12) == 0xA0F


def test_drop_bits_discards_buffered_prefix() -> None:
    reader = JpegBitReader(b"\xf0")

    assert reader.peek_bits(8) == 0xF0
    reader.drop_bits(4)
    assert reader.get_bits(4) == 0


def test_empty_reader_reports_operation_specific_errors() -> None:
    reader = JpegBitReader(b"")

    with pytest.raises(JpegUnsupportedError, match="Unexpected end"):
        reader.fill_byte()
    with pytest.raises(JpegParseError, match="unexpected end"):
        reader.get_bit()
