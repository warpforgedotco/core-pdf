import pytest

from core_jpeg import JpegParseError, JpegUnsupportedError
from core_jpeg.impl.codecs.dct.bitstream import JpegBitReader
from core_jpeg.impl.codecs.dct.huffman import (
    build_huffman_table,
    extend_sign,
    read_huffman_value,
)


def test_builds_and_decodes_single_symbol_table() -> None:
    table = build_huffman_table(bytes([1, *([0] * 15)]), b"\x2a")

    assert read_huffman_value(JpegBitReader(b"\x00"), table) == 0x2A


def test_rejects_mismatched_symbol_counts() -> None:
    with pytest.raises(JpegParseError, match="truncated"):
        build_huffman_table(bytes([1, *([0] * 15)]), b"")

    with pytest.raises(JpegParseError, match="invalid"):
        build_huffman_table(bytes(16), b"\x00")


def test_rejects_code_not_present_in_table() -> None:
    table = build_huffman_table(bytes([1, *([0] * 15)]), b"\x2a")

    with pytest.raises(JpegUnsupportedError, match="exceeded 16 bits"):
        read_huffman_value(JpegBitReader(b"\xff\xff"), table)


@pytest.mark.parametrize(
    ("value", "bits", "expected"),
    [(0, 1, -1), (1, 1, 1), (0, 3, -7), (3, 3, -4), (4, 3, 4), (7, 3, 7)],
)
def test_extend_sign(value: int, bits: int, expected: int) -> None:
    assert extend_sign(value, bits) == expected
