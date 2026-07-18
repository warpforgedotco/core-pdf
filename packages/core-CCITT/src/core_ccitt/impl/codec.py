"""CCITT Group 3/4 fax codec."""

from __future__ import annotations

import typing
from bisect import bisect_left
from collections.abc import Mapping

if typing.TYPE_CHECKING:
    pass


class CcittError(Exception):
    """Base error for CCITT codec failures."""


class CcittParseError(CcittError):
    """Raised when CCITT bytes or parameters are malformed."""


class CcittUnsupportedError(CcittError):
    """Raised when valid CCITT data uses unsupported features."""


def make_run_codes(entries: tuple[tuple[int, str], ...]) -> dict[tuple[int, int], int]:
    return {(int(bits, 2), len(bits)): value for value, bits in entries}


WHITE_RUN_CODES = make_run_codes(
    (
        (0, "00110101"),
        (1, "000111"),
        (2, "0111"),
        (3, "1000"),
        (4, "1011"),
        (5, "1100"),
        (6, "1110"),
        (7, "1111"),
        (8, "10011"),
        (9, "10100"),
        (10, "00111"),
        (11, "01000"),
        (12, "001000"),
        (13, "000011"),
        (14, "110100"),
        (15, "110101"),
        (16, "101010"),
        (17, "101011"),
        (18, "0100111"),
        (19, "0001100"),
        (20, "0001000"),
        (21, "0010111"),
        (22, "0000011"),
        (23, "0000100"),
        (24, "0101000"),
        (25, "0101011"),
        (26, "0010011"),
        (27, "0100100"),
        (28, "0011000"),
        (29, "00000010"),
        (30, "00000011"),
        (31, "00011010"),
        (32, "00011011"),
        (33, "00010010"),
        (34, "00010011"),
        (35, "00010100"),
        (36, "00010101"),
        (37, "00010110"),
        (38, "00010111"),
        (39, "00101000"),
        (40, "00101001"),
        (41, "00101010"),
        (42, "00101011"),
        (43, "00101100"),
        (44, "00101101"),
        (45, "00000100"),
        (46, "00000101"),
        (47, "00001010"),
        (48, "00001011"),
        (49, "01010010"),
        (50, "01010011"),
        (51, "01010100"),
        (52, "01010101"),
        (53, "00100100"),
        (54, "00100101"),
        (55, "01011000"),
        (56, "01011001"),
        (57, "01011010"),
        (58, "01011011"),
        (59, "01001010"),
        (60, "01001011"),
        (61, "00110010"),
        (62, "00110011"),
        (63, "00110100"),
        (64, "11011"),
        (128, "10010"),
        (192, "010111"),
        (256, "0110111"),
        (320, "00110110"),
        (384, "00110111"),
        (448, "01100100"),
        (512, "01100101"),
        (576, "01101000"),
        (640, "01100111"),
        (704, "011001100"),
        (768, "011001101"),
        (832, "011010010"),
        (896, "011010011"),
        (960, "011010100"),
        (1024, "011010101"),
        (1088, "011010110"),
        (1152, "011010111"),
        (1216, "011011000"),
        (1280, "011011001"),
        (1344, "011011010"),
        (1408, "011011011"),
        (1472, "010011000"),
        (1536, "010011001"),
        (1600, "010011010"),
        (1664, "011000"),
        (1728, "010011011"),
        (1792, "00000001000"),
        (1856, "00000001100"),
        (1920, "00000001101"),
        (1984, "000000010010"),
        (2048, "000000010011"),
        (2112, "000000010100"),
        (2176, "000000010101"),
        (2240, "000000010110"),
        (2304, "000000010111"),
        (2368, "000000011100"),
        (2432, "000000011101"),
        (2496, "000000011110"),
        (2560, "000000011111"),
    )
)

BLACK_RUN_CODES = make_run_codes(
    (
        (0, "0000110111"),
        (1, "010"),
        (2, "11"),
        (3, "10"),
        (4, "011"),
        (5, "0011"),
        (6, "0010"),
        (7, "00011"),
        (8, "000101"),
        (9, "000100"),
        (10, "0000100"),
        (11, "0000101"),
        (12, "0000111"),
        (13, "00000100"),
        (14, "00000111"),
        (15, "000011000"),
        (16, "0000010111"),
        (17, "0000011000"),
        (18, "0000001000"),
        (19, "00001100111"),
        (20, "00001101000"),
        (21, "00001101100"),
        (22, "00000110111"),
        (23, "00000101000"),
        (24, "00000010111"),
        (25, "00000011000"),
        (26, "000011001010"),
        (27, "000011001011"),
        (28, "000011001100"),
        (29, "000011001101"),
        (30, "000001101000"),
        (31, "000001101001"),
        (32, "000001101010"),
        (33, "000001101011"),
        (34, "000011010010"),
        (35, "000011010011"),
        (36, "000011010100"),
        (37, "000011010101"),
        (38, "000011010110"),
        (39, "000011010111"),
        (40, "000001101100"),
        (41, "000001101101"),
        (42, "000011011010"),
        (43, "000011011011"),
        (44, "000001010100"),
        (45, "000001010101"),
        (46, "000001010110"),
        (47, "000001010111"),
        (48, "000001100100"),
        (49, "000001100101"),
        (50, "000001010010"),
        (51, "000001010011"),
        (52, "000000100100"),
        (53, "000000110111"),
        (54, "000000111000"),
        (55, "000000100111"),
        (56, "000000101000"),
        (57, "000001011000"),
        (58, "000001011001"),
        (59, "000000101011"),
        (60, "000000101100"),
        (61, "000001011010"),
        (62, "000001100110"),
        (63, "000001100111"),
        (64, "0000001111"),
        (128, "000011001000"),
        (192, "000011001001"),
        (256, "000001011011"),
        (320, "000000110011"),
        (384, "000000110100"),
        (448, "000000110101"),
        (512, "0000001101100"),
        (576, "0000001101101"),
        (640, "0000001001010"),
        (704, "0000001001011"),
        (768, "0000001001100"),
        (832, "0000001001101"),
        (896, "0000001110010"),
        (960, "0000001110011"),
        (1024, "0000001110100"),
        (1088, "0000001110101"),
        (1152, "0000001110110"),
        (1216, "0000001110111"),
        (1280, "0000001010010"),
        (1344, "0000001010011"),
        (1408, "0000001010100"),
        (1472, "0000001010101"),
        (1536, "0000001011010"),
        (1600, "0000001011011"),
        (1664, "0000001100100"),
        (1728, "0000001100101"),
        (1792, "00000001000"),
        (1856, "00000001100"),
        (1920, "00000001101"),
        (1984, "000000010010"),
        (2048, "000000010011"),
        (2112, "000000010100"),
        (2176, "000000010101"),
        (2240, "000000010110"),
        (2304, "000000010111"),
        (2368, "000000011100"),
        (2432, "000000011101"),
        (2496, "000000011110"),
        (2560, "000000011111"),
    )
)


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


def make_prefix_table(
    codes: Mapping[tuple[int, int], int | str], max_width: int
) -> list[tuple[int | str, int] | None]:
    table: list[tuple[int | str, int] | None] = [None] * (1 << max_width)
    for (bits, width), value in codes.items():
        if width > max_width:
            continue
        prefix = bits << (max_width - width)
        span = 1 << (max_width - width)
        entry = (value, width)
        for index in range(prefix, prefix + span):
            table[index] = entry
    return table


RUN_CODE_FAST_WIDTH = 13
WHITE_RUN_PREFIX_TABLE = make_prefix_table(WHITE_RUN_CODES, RUN_CODE_FAST_WIDTH)
BLACK_RUN_PREFIX_TABLE = make_prefix_table(BLACK_RUN_CODES, RUN_CODE_FAST_WIDTH)
MODE_FAST_WIDTH = 7
MODE_FAST_CODES: dict[tuple[int, int], int | str] = {
    (0b1, 1): 0,
    (0b001, 3): "H",
    (0b011, 3): 1,
    (0b010, 3): -1,
    (0b0001, 4): "P",
    (0b000011, 6): 2,
    (0b000010, 6): -2,
    (0b0000011, 7): 3,
    (0b0000010, 7): -3,
}
MODE_PREFIX_TABLE = make_prefix_table(MODE_FAST_CODES, MODE_FAST_WIDTH)


class BitReader:
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
            raise CcittParseError("invalid bit read width")
        if self.bit_pos + width > len(self.data) * 8:
            raise CcittParseError("insufficient CCITT data")
        byte_pos = self.bit_pos >> 3
        bit_offset = self.bit_pos & 7
        needed = (width + bit_offset + 7) >> 3
        data = self.data
        if needed == 1:
            word = data[byte_pos]
        elif needed == 2:
            word = (data[byte_pos] << 8) | data[byte_pos + 1]
        elif needed == 3:
            word = (data[byte_pos] << 16) | (data[byte_pos + 1] << 8) | data[byte_pos + 2]
        else:
            word = 0
            for i in range(needed):
                word = (word << 8) | data[byte_pos + i]
        shift = needed * 8 - width - bit_offset
        return (word >> shift) & ((1 << width) - 1)

    def advance(self, width: int) -> None:
        if width < 0 or self.bit_pos + width > len(self.data) * 8:
            raise CcittParseError("insufficient CCITT data")
        self.bit_pos += width


class BitWriter:
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
            raise CcittUnsupportedError("invalid CCITT bit write")
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

    def write_repeated(self, value: int, count: int) -> None:
        if value not in (0, 1) or count < 0:
            raise CcittUnsupportedError("invalid CCITT repeated bit write")
        if count == 0:
            return

        data = self.data
        current = self.current
        bit_count = self.bit_count

        if bit_count:
            take = min(count, 8 - bit_count)
            if value:
                current = (current << take) | ((1 << take) - 1)
            else:
                current <<= take
            bit_count += take
            count -= take
            if bit_count == 8:
                data.append(current)
                current = 0
                bit_count = 0

        whole_bytes = count >> 3
        if whole_bytes:
            data.extend((b"\xff" if value else b"\x00") * whole_bytes)
            count &= 7

        if count:
            current = (1 << count) - 1 if value else 0
            bit_count = count

        self.current = current
        self.bit_count = bit_count

    def write_line(self, bits: list[int]) -> None:
        if self.bit_count:
            for bit in bits:
                self.write(bit, 1)
            self.flush_byte()
            return
        if 0 not in bits:
            self.write_repeated(1, len(bits))
            self.flush_byte()
            return
        if 1 not in bits:
            self.write_repeated(0, len(bits))
            self.flush_byte()
            return

        data = self.data
        index = 0
        bit_count = len(bits)
        while index + 8 <= bit_count:
            data.append(
                (bits[index] << 7)
                | (bits[index + 1] << 6)
                | (bits[index + 2] << 5)
                | (bits[index + 3] << 4)
                | (bits[index + 4] << 3)
                | (bits[index + 5] << 2)
                | (bits[index + 6] << 1)
                | bits[index + 7]
            )
            index += 8

        if index < bit_count:
            byte = 0
            shift = 7
            while index < bit_count:
                byte |= bits[index] << shift
                shift -= 1
                index += 1
            data.append(byte)

    def flush_byte(self) -> None:
        if self.bit_count:
            self.data.append(self.current << (8 - self.bit_count))
            self.current = 0
            self.bit_count = 0

    def finish(self) -> bytes:
        self.flush_byte()
        return bytes(self.data)


def find_run_length(reader: BitReader, is_white: bool) -> int | None:
    codes = WHITE_RUN_CODES if is_white else BLACK_RUN_CODES
    if reader.remaining_bits >= RUN_CODE_FAST_WIDTH:
        table = WHITE_RUN_PREFIX_TABLE if is_white else BLACK_RUN_PREFIX_TABLE
        entry = table[reader.peek(RUN_CODE_FAST_WIDTH)]
        if entry is None:
            return None
        run_length, width = entry
        reader.advance(width)
        return run_length if type(run_length) is int else None

    for width in range(2, 14):
        if reader.remaining_bits < width:
            return None
        bits = reader.peek(width)
        if (bits, width) in codes:
            run_length = codes[(bits, width)]
            reader.advance(width)
            return run_length
    return None


def decode_1d_row(reader: BitReader, writer: BitWriter, columns: int, byte_align: bool) -> bool:
    bits_written = 0
    is_white = True

    while bits_written < columns:
        if reader.remaining_bits < 1:
            raise CcittParseError("insufficient CCITT data")
        run_length_result = find_run_length(reader, is_white)
        if run_length_result is None:
            raise CcittParseError("invalid CCITT run length")
        run_length = run_length_result
        to_write = min(run_length, columns - bits_written)
        bit_value = 1 if is_white else 0
        writer.write_repeated(bit_value, to_write)
        bits_written += to_write
        if run_length < 64:
            is_white = not is_white

    while bits_written < columns:
        writer.write_repeated(1, columns - bits_written)
        bits_written = columns
    writer.flush_byte()

    if byte_align and bits_written % 8 != 0:
        padding = 8 - (bits_written % 8)
        if reader.remaining_bits >= padding:
            reader.advance(padding)

    return bits_written > 0


def find_2d_mode(reader: BitReader) -> int | str | None:
    if reader.remaining_bits < 1:
        return None
    if reader.remaining_bits >= MODE_FAST_WIDTH:
        entry = MODE_PREFIX_TABLE[reader.peek(MODE_FAST_WIDTH)]
        if entry is None:
            return None
        mode, width = entry
        reader.advance(width)
        return mode

    if reader.peek(1) == 0b1:
        reader.advance(1)
        return 0

    if reader.remaining_bits >= 3:
        bits = reader.peek(3)
        if bits == 0b001:
            reader.advance(3)
            return "H"
        if bits == 0b011:
            reader.advance(3)
            return 1
        if bits == 0b010:
            reader.advance(3)
            return -1

    if reader.remaining_bits >= 4 and reader.peek(4) == 0b0001:
        reader.advance(4)
        return "P"

    if reader.remaining_bits >= 6:
        bits = reader.peek(6)
        if bits == 0b000011:
            reader.advance(6)
            return 2
        if bits == 0b000010:
            reader.advance(6)
            return -2

    if reader.remaining_bits >= 7:
        bits = reader.peek(7)
        if bits == 0b0000011:
            reader.advance(7)
            return 3
        if bits == 0b0000010:
            reader.advance(7)
            return -3
    return None


def find_reference_pixel(
    ref_line: list[int],
    current_pos: int,
    target_color: int,
    start_index: int | None = None,
    transition_maps: tuple[list[int], list[int]] | None = None,
) -> int:
    if start_index is None:
        start_index = current_pos + 1
    if transition_maps is not None:
        transitions = transition_maps[target_color]
        index = bisect_left(transitions, start_index)
        return transitions[index] if index < len(transitions) else len(ref_line)
    pos = start_index
    while pos < len(ref_line):
        if pos == 0:
            if target_color == 1 and ref_line[pos] != target_color:
                return pos
        elif ref_line[pos - 1] == target_color and ref_line[pos] != target_color:
            return pos
        pos += 1
    return len(ref_line)


def reference_transition_maps(ref_line: list[int]) -> tuple[list[int], list[int]]:
    transitions: tuple[list[int], list[int]] = ([], [])
    if not ref_line:
        return transitions
    if ref_line[0] != 1:
        transitions[1].append(0)
    for pos in range(1, len(ref_line)):
        previous = ref_line[pos - 1]
        current = ref_line[pos]
        if previous != current:
            transitions[previous].append(pos)
    return transitions


def set_line_run(line: list[int], start: int, end: int, value: int) -> bool:
    if value == 1 or end <= start:
        return False
    line[start:end] = [value] * (end - start)
    return True


def decode_2d_row(
    reader: BitReader,
    writer: BitWriter,
    ref_line: list[int],
    cur_line: list[int],
    columns: int,
    byte_align: bool,
    ref_transition_maps: tuple[list[int], list[int]] | None = None,
) -> tuple[bool, bool]:
    cur_pos = -1
    color = 1
    has_black = False
    transition_maps = (
        reference_transition_maps(ref_line) if ref_transition_maps is None else ref_transition_maps
    )

    while cur_pos < columns - 1:
        if reader.remaining_bits < 2:
            raise CcittParseError("insufficient CCITT data")

        mode = find_2d_mode(reader)

        if mode == "P":
            x1 = find_reference_pixel(ref_line, cur_pos, color, transition_maps=transition_maps)
            x2 = find_reference_pixel(
                ref_line,
                x1,
                1 - color,
                start_index=x1,
                transition_maps=transition_maps,
            )
            has_black = (
                set_line_run(cur_line, max(0, cur_pos), min(columns, x2), color) or has_black
            )
            cur_pos = x2
        elif mode == "H":
            n1 = 0
            n2 = 0
            while True:
                if reader.remaining_bits < 8:
                    raise CcittParseError("insufficient CCITT data")
                run = find_run_length(reader, color == 1)
                if run is None:
                    raise CcittParseError("invalid CCITT run length")
                n1 += run
                if run < 64:
                    break
            color = 1 - color
            while True:
                if reader.remaining_bits < 8:
                    raise CcittParseError("insufficient CCITT data")
                run = find_run_length(reader, color == 1)
                if run is None:
                    raise CcittParseError("invalid CCITT run length")
                n2 += run
                if run < 64:
                    break
            start = max(0, cur_pos)
            mid = min(columns, start + n1)
            end = min(columns, start + n1 + n2)
            has_black = set_line_run(cur_line, start, mid, 1 - color) or has_black
            has_black = set_line_run(cur_line, mid, end, color) or has_black
            cur_pos = end
            color = 1 - color
        elif type(mode) is int:
            x1 = find_reference_pixel(ref_line, cur_pos, color, transition_maps=transition_maps)
            x1 += mode
            x1 = max(0, min(columns, x1))
            has_black = set_line_run(cur_line, max(0, cur_pos), x1, color) or has_black
            cur_pos = x1
            color = 1 - color
        else:
            raise CcittParseError("invalid CCITT mode")

    set_line_run(cur_line, max(0, cur_pos), columns, 1)

    if byte_align and (reader.bit_pos % 8) != 0:
        padding = 8 - (reader.bit_pos % 8)
        if reader.remaining_bits >= padding:
            reader.advance(padding)

    return True, has_black


def decode_ccitt_fax(
    data: bytes,
    columns: int = 1728,
    rows: int = 0,
    byte_align: bool = False,
    k: int = -1,
) -> bytes:
    if type(columns) is not int or columns <= 0:
        raise CcittParseError("invalid CCITT columns")
    if columns > 50000:
        raise CcittParseError("invalid CCITT columns")
    if type(rows) is not int or rows < 0:
        raise CcittParseError("invalid CCITT rows")
    if rows > 0 and columns * rows > 50000000:
        raise CcittParseError("CCITT image is too large")
    if type(byte_align) is not bool:
        raise CcittParseError("invalid CCITT byte align")
    if type(k) is not int:
        raise CcittParseError("invalid CCITT K value")
    if k < -1:
        raise CcittParseError("invalid CCITT K value")

    reader = BitReader(data)
    writer = BitWriter(bytearray())
    ref_line: list[int] = [1] * columns
    ref_transition_maps: tuple[list[int], list[int]] = ([], [])
    max_row_count = rows if rows > 0 else max(1, len(data) * 8)

    for row_idx in range(max_row_count):
        if reader.remaining_bits < 1:
            break

        if k == 0:
            if not decode_1d_row(reader, writer, columns, byte_align):
                raise CcittParseError("truncated CCITT data")
        elif k > 0:
            if (row_idx % (k + 1)) == 0:
                if not decode_1d_row(reader, writer, columns, byte_align):
                    raise CcittParseError("truncated CCITT data")
            else:
                cur_line = [1] * columns
                row_ok, has_black = decode_2d_row(
                    reader,
                    writer,
                    ref_line,
                    cur_line,
                    columns,
                    byte_align,
                    ref_transition_maps,
                )
                if not row_ok:
                    raise CcittParseError("truncated CCITT data")
                if has_black:
                    writer.write_line(cur_line)
                else:
                    writer.write_repeated(1, columns)
                    writer.flush_byte()
                ref_line = cur_line
                ref_transition_maps = reference_transition_maps(ref_line) if has_black else ([], [])
        else:
            cur_line = [1] * columns
            row_ok, has_black = decode_2d_row(
                reader,
                writer,
                ref_line,
                cur_line,
                columns,
                byte_align,
                ref_transition_maps,
            )
            if not row_ok:
                raise CcittParseError("truncated CCITT data")
            if has_black:
                writer.write_line(cur_line)
            else:
                writer.write_repeated(1, columns)
                writer.flush_byte()
            ref_line = cur_line
            ref_transition_maps = reference_transition_maps(ref_line) if has_black else ([], [])

    return writer.finish()


__all__ = (
    "BLACK_RUN_CODES",
    "CcittError",
    "CcittParseError",
    "CcittUnsupportedError",
    "WHITE_RUN_CODES",
    "decode_ccitt_fax",
)
