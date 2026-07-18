# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import TypeAlias

from core_jpeg.impl.codecs.dct.bitstream import JpegBitReader
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError

FAST_HUFFMAN_BITS = 9
HuffmanRows: TypeAlias = list[dict[int, int]]
HuffmanTable: TypeAlias = tuple[list[int], list[int], HuffmanRows]


def build_huffman_table(
    lengths: bytes, symbols: bytes
) -> tuple[list[int], list[int], list[dict[int, int]]]:
    huff: HuffmanRows = [{} for ignored in range(16)]
    fast_values = [-1] * (1 << FAST_HUFFMAN_BITS)
    fast_lengths = [0] * (1 << FAST_HUFFMAN_BITS)
    code = 0
    k = 0
    for i, num in enumerate(lengths):
        code_length = i + 1
        table = huff[i]
        for ignored in range(num):
            if k >= len(symbols):
                raise JpegParseError("truncated JPEG Huffman table")
            symbol = symbols[k]
            table[code] = symbol
            if code_length <= FAST_HUFFMAN_BITS:
                prefix = code << (FAST_HUFFMAN_BITS - code_length)
                span = 1 << (FAST_HUFFMAN_BITS - code_length)
                for fast_code in range(prefix, prefix + span):
                    fast_values[fast_code] = symbol
                    fast_lengths[fast_code] = code_length
            k += 1
            code += 1
        code <<= 1
    if k != len(symbols):
        raise JpegParseError("invalid JPEG Huffman table")
    return fast_values, fast_lengths, huff


def read_huffman_value(
    reader: JpegBitReader,
    table: HuffmanTable,
) -> int:
    fast_values, fast_lengths, huff = table
    while reader.bits_left < FAST_HUFFMAN_BITS and reader.pos < reader.len:
        reader.fill_byte()
    bits_left = reader.bits_left
    if bits_left >= FAST_HUFFMAN_BITS:
        fast_code = (reader.buffer >> (bits_left - FAST_HUFFMAN_BITS)) & (
            (1 << FAST_HUFFMAN_BITS) - 1
        )
        fast_length = fast_lengths[fast_code]
        if fast_length:
            bits_left -= fast_length
            reader.bits_left = bits_left
            if bits_left:
                reader.buffer &= (1 << bits_left) - 1
            else:
                reader.buffer = 0
            return fast_values[fast_code]
    elif bits_left:
        missing = FAST_HUFFMAN_BITS - bits_left
        fast_code = ((reader.buffer << missing) | ((1 << missing) - 1)) & (
            (1 << FAST_HUFFMAN_BITS) - 1
        )
        fast_length = fast_lengths[fast_code]
        if fast_length and fast_length <= bits_left:
            reader.bits_left = bits_left - fast_length
            if reader.bits_left:
                reader.buffer &= (1 << reader.bits_left) - 1
            else:
                reader.buffer = 0
            return fast_values[fast_code]
    code = 0
    for current_table in huff:
        code = (code << 1) | reader.get_bit()
        value = current_table.get(code)
        if value is not None:
            return value
    raise JpegUnsupportedError("Invalid JPEG Huffman code (exceeded 16 bits)")


def extend_sign(value: int, bits: int) -> int:
    vt = 1 << (bits - 1)
    return value if value >= vt else value + (-(1 << bits) + 1)
