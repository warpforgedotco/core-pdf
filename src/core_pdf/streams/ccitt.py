"""CCITT Fax stream decoding with full Group 3/4 support.

Implements ITU-T Recommendation T.4 and T.6 fax compression.
- K = -1: Group 4 (2D compression)
- K = 0: Group 3 1D (1D only)
- K > 0: Group 3 mixed (1D rows with max K 2D rows)
- K < -1: Group 3 2D (2D only, mixed mode)
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass

from core_pdf.syntax.errors import PdfParseError, PdfUnsupportedError

# White and black terminating run-length codes per ITU-T T.4/T.6
WHITE_RUN_CODES: dict[tuple[int, int], int] = {
    (0b00110101, 8): 0,
    (0b000111, 6): 1,
    (0b0111, 4): 2,
    (0b1000, 4): 3,
    (0b1011, 4): 4,
    (0b1100, 4): 5,
    (0b1110, 4): 6,
    (0b1111, 4): 7,
    (0b10011, 5): 8,
    (0b10100, 5): 9,
    (0b00111, 5): 10,
    (0b01000, 5): 11,
    (0b001000, 6): 12,
    (0b000011, 6): 13,
    (0b110100, 6): 14,
    (0b110101, 6): 15,
    (0b101010, 6): 16,
    (0b101011, 6): 17,
    (0b0100111, 7): 18,
    (0b0101000, 7): 19,
    (0b0101011, 7): 20,
    (0b0010011, 7): 21,
    (0b0100100, 7): 22,
    (0b0101100, 7): 23,
    (0b0101101, 7): 24,
    (0b0010100, 7): 25,
    (0b0010101, 7): 26,
    (0b0010110, 7): 27,
    (0b0110010, 7): 28,
    (0b0110011, 7): 29,
    (0b1011010, 7): 30,
    (0b1011011, 7): 31,
    (0b10010010, 8): 32,
    (0b10010011, 8): 33,
    (0b10011010, 8): 34,
    (0b00100100, 8): 35,
    (0b00110011, 8): 36,
    (0b00110100, 8): 37,
    (0b01101000, 8): 38,
    (0b01100111, 8): 39,
    (0b00111010, 8): 40,
    (0b00111011, 8): 41,
    (0b01011000, 8): 42,
    (0b01011001, 8): 43,
    (0b01011010, 8): 44,
    (0b01100110, 8): 45,
    (0b01100010, 8): 46,
    (0b01010010, 8): 47,
    (0b01010011, 8): 48,
    (0b00100111, 8): 49,
    (0b0011000, 7): 50,
    (0b11011, 5): 51,
    (0b100, 3): 53,
    (0b11010, 5): 192,
    (0b010111, 6): 1664,
    (0b0110111, 7): 256,
    (0b00110110, 8): 320,
    (0b00110111, 8): 384,
    (0b01100100, 8): 448,
    (0b01100101, 8): 512,
    (0b01101001, 8): 576,
    (0b01101010, 8): 640,
    (0b011001100, 9): 704,
    (0b011001101, 9): 768,
    (0b011010010, 9): 832,
    (0b011010011, 9): 896,
    (0b011010100, 9): 960,
    (0b011010101, 9): 1024,
    (0b011010110, 9): 1088,
    (0b011010111, 9): 1152,
    (0b011011000, 9): 1216,
    (0b011011001, 9): 1280,
    (0b011011010, 9): 1344,
    (0b011011011, 9): 1408,
    (0b010011000, 9): 1472,
    (0b010011001, 9): 1536,
    (0b010011010, 9): 1600,
    (0b011000, 6): 1664,
    (0b010011011, 9): 1728,
    (0b00000001000, 11): 1792,
    (0b00000001100, 11): 1856,
    (0b00000001101, 11): 1920,
    (0b000000010010, 12): 1984,
    (0b000000010011, 12): 2048,
    (0b000000010100, 12): 2112,
    (0b000000010101, 12): 2176,
    (0b000000010110, 12): 2240,
    (0b000000010111, 12): 2304,
    (0b000000011100, 12): 2368,
    (0b000000011101, 12): 2432,
    (0b000000011110, 12): 2496,
    (0b000000011111, 12): 2560,
}

BLACK_RUN_CODES: dict[tuple[int, int], int] = {
    (0b0000110111, 10): 0,
    (0b010, 3): 1,
    (0b11, 2): 2,
    (0b10, 2): 3,
    (0b011, 3): 4,
    (0b0011, 4): 5,
    (0b0010, 4): 6,
    (0b00011, 5): 7,
    (0b000101, 6): 8,
    (0b000100, 6): 9,
    (0b0000101, 7): 10,
    (0b0000100, 7): 11,
    (0b0000111, 7): 12,
    (0b00000100, 8): 13,
    (0b00000101, 8): 14,
    (0b00000111, 8): 15,
    (0b000011000, 9): 16,
    (0b0000010111, 10): 17,
    (0b0000011000, 10): 18,
    (0b0000001111, 10): 19,
    (0b00001011, 8): 20,
    (0b00001010, 8): 21,
    (0b1000, 4): 23,
    (0b0101, 4): 24,
    (0b01010, 5): 25,
    (0b00101, 5): 26,
    (0b0010111, 7): 27,
    (0b00101000, 8): 28,
    (0b00101001, 8): 29,
    (0b0010110, 7): 30,
    (0b00010111, 8): 31,
    (0b00010, 5): 32,
    (0b11010, 5): 33,
    (0b11011, 5): 35,
    (0b10010, 5): 36,
    (0b10011, 5): 37,
    (0b10100, 5): 38,
    (0b00100, 5): 39,
    (0b01011, 5): 40,
    (0b0100, 4): 42,
    (0b1010, 4): 43,
    (0b1011, 4): 45,
    (0b1100, 4): 46,
    (0b1101, 4): 47,
    (0b1110, 4): 48,
    (0b1111, 4): 49,
    (0b10000, 5): 50,
    (0b10001, 5): 51,
    (0b010111, 6): 52,
    (0b011000, 6): 53,
    (0b011001, 6): 54,
    (0b011010, 6): 55,
    (0b011011, 6): 56,
    (0b010100, 6): 57,
    (0b010101, 6): 58,
    (0b010110, 6): 59,
    (0b011100, 6): 60,
    (0b011101, 6): 61,
    (0b011110, 6): 62,
    (0b011111, 6): 63,
    (0b001000, 6): 64,
    (0b0010011, 7): 128,
    (0b0010100, 7): 192,
    (0b0101011, 7): 256,
    (0b0101100, 7): 320,
    (0b000011001000, 12): 384,
    (0b000011001001, 12): 448,
    (0b000001011011, 12): 512,
    (0b000000110011, 12): 576,
    (0b000000110100, 12): 640,
    (0b000000110101, 12): 704,
    (0b0000001101100, 13): 768,
    (0b0000001101101, 13): 832,
    (0b0000001001010, 13): 896,
    (0b0000001001011, 13): 960,
    (0b0000001001100, 13): 1024,
    (0b0000001001101, 13): 1088,
    (0b0000001110010, 13): 1152,
    (0b0000001110011, 13): 1216,
    (0b0000001110100, 13): 1280,
    (0b0000001110101, 13): 1344,
    (0b0000001110110, 13): 1408,
    (0b0000001110111, 13): 1472,
    (0b0000001010010, 13): 1536,
    (0b0000001010011, 13): 1600,
    (0b0000001010100, 13): 1664,
    (0b0000001010101, 13): 1728,
    (0b0000001011010, 13): 1792,
    (0b0000001011011, 13): 1856,
    (0b0000001100100, 13): 1920,
    (0b00000001000, 11): 1984,
    (0b00000001100, 11): 2048,
    (0b00000001101, 11): 2112,
    (0b000000010010, 12): 2176,
    (0b000000010011, 12): 2240,
    (0b000000010100, 12): 2304,
    (0b000000010101, 12): 2368,
    (0b000000010110, 12): 2432,
    (0b000000010111, 12): 2496,
    (0b000000011100, 12): 2560,
    (0b000000011101, 12): 2624,
    (0b000000011110, 12): 2688,
    (0b000000011111, 12): 2752,
}

# 2D mode codes
MODE_CODES = {
    (0b1, 1): "V0",
    (0b011, 3): "VR1",
    (0b010, 3): "VL1",
    (0b0011, 4): "VR2",
    (0b0010, 4): "VL2",
    (0b00011, 5): "VR3",
    (0b00010, 5): "VL3",
    (0b001, 3): "H",
    (0b0001, 4): "P",
}

UNCOMPRESSED_CODES = {
    (0b1, 1): "1",
    (0b01, 2): "01",
    (0b001, 3): "001",
    (0b0001, 4): "0001",
    (0b00001, 5): "00001",
    (0b000001, 6): "00000",
    (0b00000011, 8): "T00",
    (0b00000010, 8): "T10",
    (0b000000011, 9): "T000",
    (0b000000010, 9): "T100",
    (0b0000000011, 10): "T0000",
    (0b0000000010, 10): "T1000",
    (0b00000000011, 11): "T00000",
    (0b00000000010, 11): "T10000",
}


class BitReader:
    """Read bits from byte stream in MSB-first order."""

    __slots__ = ("data", "bit_pos")

    data: bytes
    bit_pos: int

    def __init__(self, data: bytes, bit_pos: int = 0) -> None:
        self.data = data
        self.bit_pos = bit_pos

    @property
    def remaining_bits(self) -> int:
        return max(0, len(self.data) * 8 - self.bit_pos)

    def peek(self, width: int) -> int:
        if width <= 0:
            raise PdfParseError("invalid bit read width")
        if self.bit_pos + width > len(self.data) * 8:
            raise PdfParseError("insufficient CCITT data")
        byte_pos = self.bit_pos >> 3
        bit_offset = self.bit_pos & 7
        needed = (width + bit_offset + 7) >> 3
        word = 0
        for i in range(needed):
            word = (word << 8) | self.data[byte_pos + i]
        shift = needed * 8 - width - bit_offset
        return (word >> shift) & ((1 << width) - 1)

    def advance(self, width: int) -> None:
        if width < 0 or self.bit_pos + width > len(self.data) * 8:
            raise PdfParseError("insufficient CCITT data")
        self.bit_pos += width


class BitWriter:
    """Write bits to byte buffer in MSB-first order."""

    __slots__ = ("data", "current", "bit_count")

    data: bytearray
    current: int
    bit_count: int

    def __init__(self, data: bytearray, current: int = 0, bit_count: int = 0) -> None:
        self.data = data
        self.current = current
        self.bit_count = bit_count

    def write(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or (width and value >= (1 << width)):
            raise PdfUnsupportedError("invalid CCITT bit write")
        while width > 0:
            available = 8 - self.bit_count
            take = min(width, available)
            shift = width - take
            chunk = (value >> shift) & ((1 << take) - 1)
            self.current = (self.current << take) | chunk
            self.bit_count += take
            width -= take
            value &= (1 << shift) - 1 if shift else 0
            if self.bit_count == 8:
                self.data.append(self.current)
                self.current = 0
                self.bit_count = 0

    def finish(self) -> bytes:
        if self.bit_count:
            self.data.append(self.current << (8 - self.bit_count))
            self.current = 0
            self.bit_count = 0
        return bytes(self.data)


def find_run_length(reader: BitReader, is_white: bool) -> int | None:
    """Find next run-length code for white or black run."""
    codes = WHITE_RUN_CODES if is_white else BLACK_RUN_CODES
    for width in range(2, 14):
        if reader.remaining_bits < width:
            return None
        bits = reader.peek(width)
        if (bits, width) in codes:
            run_length = codes[(bits, width)]
            reader.advance(width)
            return run_length
    return 0


def decode_1d_row(reader: BitReader, writer: BitWriter, columns: int, byte_align: bool) -> bool:
    """Decode one row of 1D Group 3 data. Return True if row completed, False if EOF."""
    bits_written = 0
    is_white = True

    while bits_written < columns:
        if reader.remaining_bits < 1:
            raise PdfParseError("insufficient CCITT data")
        run_length_result = find_run_length(reader, is_white)
        if run_length_result is None:
            raise PdfParseError("invalid CCITT run length")
        run_length = run_length_result
        to_write = min(run_length, columns - bits_written)
        bit_value = 0 if is_white else 1
        for _ in range(to_write):
            writer.write(bit_value, 1)
        bits_written += to_write
        if run_length < 64:
            is_white = not is_white

    while bits_written < columns:
        writer.write(0, 1)
        bits_written += 1

    if byte_align and bits_written % 8 != 0:
        padding = 8 - (bits_written % 8)
        if reader.remaining_bits >= padding:
            reader.advance(padding)

    return bits_written > 0


def find_vertical_mode(reader: BitReader) -> int | None:
    """Decode 2D vertical mode codes: V(0), VR(1..3), VL(1..3)."""
    for width in range(1, 8):
        if reader.remaining_bits < width:
            return None
        bits = reader.peek(width)
        if width == 1 and bits == 0b1:
            reader.advance(1)
            return 0
        elif width == 3:
            if bits == 0b011:
                reader.advance(3)
                return 1
            elif bits == 0b010:
                reader.advance(3)
                return -1
        elif width == 4:
            if bits == 0b0011:
                reader.advance(4)
                return 2
            elif bits == 0b0010:
                reader.advance(4)
                return -2
        elif width == 5:
            if bits == 0b00011:
                reader.advance(5)
                return 3
            elif bits == 0b00010:
                reader.advance(5)
                return -3
    return None


def find_reference_pixel(
    ref_line: list[int], current_pos: int, target_color: int, start_index: int | None = None
) -> int:
    """Find next reference line pixel position with color change relative to current."""
    if start_index is None:
        start_index = current_pos + 1
    pos = start_index
    while pos < len(ref_line):
        if pos == 0:
            if target_color == 1 and ref_line[pos] != target_color:
                return pos
        elif ref_line[pos - 1] == target_color and ref_line[pos] != target_color:
            return pos
        pos += 1
    return len(ref_line)


def decode_2d_row(
    reader: BitReader,
    writer: BitWriter,
    ref_line: list[int],
    cur_line: list[int],
    columns: int,
    byte_align: bool,
) -> bool:
    """Decode one row of 2D Group 4/mixed data. Return True if row completed."""
    cur_pos = -1
    is_white = True

    while cur_pos < columns - 1:
        if reader.remaining_bits < 2:
            raise PdfParseError("insufficient CCITT data")

        bits = reader.peek(3)
        mode = None

        if bits & 0b100 == 0 and bits & 0b110 == 0b010:
            mode = "P"
        elif bits & 0b110 == 0b010:
            mode = "H"
        else:
            mode = find_vertical_mode(reader)

        if mode == "P":
            reader.advance(3)
            x1 = find_reference_pixel(ref_line, cur_pos, is_white)
            x2 = find_reference_pixel(ref_line, x1, 1 - is_white)
            for i in range(cur_pos + 1, x2):
                if 0 <= i < columns:
                    cur_line[i] = 0 if is_white else 1
            cur_pos = x2
        elif mode == "H":
            reader.advance(3)
            n1 = 0
            n2 = 0
            while True:
                if reader.remaining_bits < 8:
                    raise PdfParseError("insufficient CCITT data")
                run = find_run_length(reader, is_white)
                if run is None:
                    raise PdfParseError("invalid CCITT run length")
                n1 += run
                if run < 64:
                    break
            is_white = not is_white
            while True:
                if reader.remaining_bits < 8:
                    raise PdfParseError("insufficient CCITT data")
                run = find_run_length(reader, is_white)
                if run is None:
                    raise PdfParseError("invalid CCITT run length")
                n2 += run
                if run < 64:
                    break
            end = min(columns, cur_pos + 1 + n1 + n2)
            for i in range(max(0, cur_pos + 1), min(columns, cur_pos + 1 + n1)):
                cur_line[i] = 0 if (not is_white) else 1
            for i in range(min(columns, cur_pos + 1 + n1), end):
                cur_line[i] = 0 if is_white else 1
            cur_pos = end - 1
            is_white = not is_white
        elif isinstance(mode, int):
            x1 = find_reference_pixel(ref_line, cur_pos, is_white)
            x1 += mode
            x1 = max(0, min(columns - 1, x1))
            for i in range(max(0, cur_pos + 1), x1 + 1):
                cur_line[i] = 0 if is_white else 1
            cur_pos = x1
            is_white = not is_white
        else:
            break

    for i in range(cur_pos + 1, columns):
        cur_line[i] = 0

    if byte_align and (reader.bit_pos % 8) != 0:
        padding = 8 - (reader.bit_pos % 8)
        if reader.remaining_bits >= padding:
            reader.advance(padding)

    return True


def decode_ccitt_fax(
    data: bytes, columns: int = 1728, rows: int = 0, byte_align: bool = False, k: int = -1
) -> bytes:
    """Decode CCITT fax data with full Group 3/4 support.

    Args:
        data: Compressed CCITT data
        columns: Page width in pixels (default 1728)
        rows: Page height in pixels (0 = all)
        byte_align: Align rows to byte boundaries
        k: Group type selector:
            -1: Group 4 (2D only)
             0: Group 3 1D only
            >0: Group 3 mixed (max K 2D rows)
            <-1: Group 3 2D (2D only, mixed mode)
    """
    if columns <= 0:
        raise PdfParseError("invalid CCITT columns")
    if rows < 0:
        raise PdfParseError("invalid CCITT rows")
    if k < -1:
        raise PdfParseError("invalid CCITT K value")

    reader = BitReader(data)
    writer = BitWriter(bytearray())
    ref_line: list[int] = [1] * columns
    max_row_count = rows if rows > 0 else 0xFFFFFFFF

    for row_idx in range(max_row_count):
        if reader.remaining_bits < 1:
            break

        if k == 0:
            if not decode_1d_row(reader, writer, columns, byte_align):
                raise PdfParseError("truncated CCITT data")
        elif k > 0:
            if (row_idx % (k + 1)) == 0:
                if not decode_1d_row(reader, writer, columns, byte_align):
                    raise PdfParseError("truncated CCITT data")
            else:
                cur_line = [1] * columns
                if not decode_2d_row(reader, writer, ref_line, cur_line, columns, byte_align):
                    raise PdfParseError("truncated CCITT data")
                for bit_val in cur_line:
                    writer.write(bit_val, 1)
                ref_line = cur_line
        else:
            cur_line = [1] * columns
            if not decode_2d_row(reader, writer, ref_line, cur_line, columns, byte_align):
                raise PdfParseError("truncated CCITT data")
            for bit_val in cur_line:
                writer.write(bit_val, 1)
            ref_line = cur_line

    return writer.finish()
