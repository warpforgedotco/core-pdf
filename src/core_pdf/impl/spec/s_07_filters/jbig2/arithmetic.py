# SPDX-License-Identifier: AGPL-3.0-only
"""T.88 Annex E MQ coding and arithmetic bitmap/integer procedures."""

from __future__ import annotations

from core_pdf.impl.spec.s_07_filters.jbig2.structures import (
    JBIG2Image,
    Jbig2ParseError,
    Jbig2UnsupportedError,
)

GENERIC_TEMPLATE_0_DEFAULT_AT = ((3, -1), (-3, -1), (2, -2), (-2, -2))

# The MQ-coder probability estimation table, ITU-T T.88 Table E.1. One row per
# state index: the Qe probability, the next index after a more-probable-symbol
# renormalization, the next index after a less-probable one, and whether that
# LPS path swaps the sense of the MPS.
#
internal_MQ_STATES: tuple[tuple[int, int, int, int], ...] = (
    (0x5601, 1, 1, 1),
    (0x3401, 2, 6, 0),
    (0x1801, 3, 9, 0),
    (0x0AC1, 4, 12, 0),
    (0x0521, 5, 29, 0),
    (0x0221, 38, 33, 0),
    (0x5601, 7, 6, 1),
    (0x5401, 8, 14, 0),
    (0x4801, 9, 14, 0),
    (0x3801, 10, 14, 0),
    (0x3001, 11, 17, 0),
    (0x2401, 12, 18, 0),
    (0x1C01, 13, 20, 0),
    (0x1601, 29, 21, 0),
    (0x5601, 15, 14, 1),
    (0x5401, 16, 14, 0),
    (0x5101, 17, 15, 0),
    (0x4801, 18, 16, 0),
    (0x3801, 19, 17, 0),
    (0x3401, 20, 18, 0),
    (0x3001, 21, 19, 0),
    (0x2801, 22, 19, 0),
    (0x2401, 23, 20, 0),
    (0x2201, 24, 21, 0),
    (0x1C01, 25, 22, 0),
    (0x1801, 26, 23, 0),
    (0x1601, 27, 24, 0),
    (0x1401, 28, 25, 0),
    (0x1201, 29, 26, 0),
    (0x1101, 30, 27, 0),
    (0x0AC1, 31, 28, 0),
    (0x09C1, 32, 29, 0),
    (0x08A1, 33, 30, 0),
    (0x0521, 34, 31, 0),
    (0x0441, 35, 32, 0),
    (0x02A1, 36, 33, 0),
    (0x0221, 37, 34, 0),
    (0x0141, 38, 35, 0),
    (0x0111, 39, 36, 0),
    (0x0085, 40, 37, 0),
    (0x0049, 41, 38, 0),
    (0x0025, 42, 39, 0),
    (0x0015, 43, 40, 0),
    (0x0009, 44, 41, 0),
    (0x0005, 45, 42, 0),
    (0x0001, 45, 43, 0),
    (0x5601, 46, 46, 0),
)

# The decoder indexes these per pixel, so the columns stay flat tuples of ints.
MQ_QE = tuple(state[0] for state in internal_MQ_STATES)
MQ_NMPS = tuple(state[1] for state in internal_MQ_STATES)
MQ_NLPS = tuple(state[2] for state in internal_MQ_STATES)
MQ_SWITCH = tuple(state[3] for state in internal_MQ_STATES)


class JBIG2MQDecoder:
    __slots__ = ("data", "bp", "data_end", "a", "chigh", "clow", "ct", "ctx")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bp = 0
        self.data_end = len(data)
        self.chigh = data[0] if data else 0xFF
        self.clow = 0
        self.ct = 0
        self.ctx = [0] * 65536
        self.byte_in()
        self.chigh = ((self.chigh << 7) & 0xFFFF) | ((self.clow >> 9) & 0x7F)
        self.clow = (self.clow << 7) & 0xFFFF
        self.ct -= 7
        self.a = 0x8000

    @classmethod
    def from_segment(cls, data: bytes) -> JBIG2MQDecoder:
        # Annex E's FLUSH procedure terminates arithmetic segment data with
        # FF AC. Virtual one bits are valid after that marker, not a substitute
        # for bytes missing from a truncated symbol or bitmap stream.
        if not data.endswith(b"\xff\xac"):
            raise Jbig2ParseError("missing JBIG2 arithmetic termination marker")
        return cls(data)

    def byte_in(self) -> None:
        data = self.data
        bp = self.bp
        current = data[bp] if bp < self.data_end else 0xFF
        following = data[bp + 1] if bp + 1 < self.data_end else 0xFF
        if current == 0xFF:
            if following > 0x8F:
                self.clow += 0xFF00
                self.ct = 8
            else:
                bp += 1
                value = data[bp] if bp < self.data_end else 0xFF
                self.clow += value << 9
                self.ct = 7
                self.bp = bp
        else:
            bp += 1
            value = data[bp] if bp < self.data_end else 0xFF
            self.clow += value << 8
            self.ct = 8
            self.bp = bp
        if self.clow > 0xFFFF:
            self.chigh += self.clow >> 16
            self.clow &= 0xFFFF

    def read_bit(self, context: int, contexts: list[int] | None = None) -> int:
        """Decode one context-adaptive symbol using the T.88 Annex E MQ coder."""
        if contexts is None:
            contexts = self.ctx
        packed = contexts[context]
        index = packed >> 1
        mps = packed & 1
        qe = MQ_QE[index]
        interval = self.a - qe
        if self.chigh < qe:
            if interval < qe:
                pixel = mps
                index = MQ_NMPS[index]
            else:
                pixel = 1 ^ mps
                if MQ_SWITCH[index]:
                    mps = pixel
                index = MQ_NLPS[index]
            interval = qe
        else:
            self.chigh -= qe
            if interval & 0x8000:
                self.a = interval
                return mps
            if interval < qe:
                pixel = 1 ^ mps
                if MQ_SWITCH[index]:
                    mps = pixel
                index = MQ_NLPS[index]
            else:
                pixel = mps
                index = MQ_NMPS[index]
        while not (interval & 0x8000):
            if self.ct == 0:
                self.byte_in()
            interval <<= 1
            self.chigh = ((self.chigh << 1) & 0xFFFF) | ((self.clow >> 15) & 1)
            self.clow = (self.clow << 1) & 0xFFFF
            self.ct -= 1
        self.a = interval
        contexts[context] = (index << 1) | mps
        return pixel


class ArithmeticIntegers:
    """Independent Annex A context banks sharing one arithmetic code stream."""

    def __init__(self, decoder: JBIG2MQDecoder, symbol_count: int = 0) -> None:
        self.decoder = decoder
        self.contexts: dict[str, list[int]] = {}
        self.symbol_bits = max(0, symbol_count - 1).bit_length()
        self.symbol_contexts = [0] * (1 << self.symbol_bits)

    def decode(self, name: str) -> int | None:
        contexts = self.contexts.get(name)
        if contexts is None:
            contexts = self.contexts[name] = [0] * 512
        previous = 1

        def bit() -> int:
            nonlocal previous
            value = self.decoder.read_bit(previous, contexts)
            previous = (previous << 1) | value
            if previous >= 512:
                previous = (previous & 511) | 256
            return value

        negative = bit()
        offset = 4436
        length = 32
        for candidate_length, candidate_offset in ((2, 0), (4, 4), (6, 20), (8, 84), (12, 340)):
            if not bit():
                length, offset = candidate_length, candidate_offset
                break
        value = 0
        for _ in range(length):
            value = (value << 1) | bit()
        value += offset
        if negative:
            return -value if value else None
        return value

    def integer(self, name: str) -> int:
        value = self.decode(name)
        if value is None:
            raise Jbig2ParseError(f"unexpected JBIG2 out-of-band {name} value")
        return value

    def symbol_id(self) -> int:
        previous = 1
        for _ in range(self.symbol_bits):
            previous = (previous << 1) | self.decoder.read_bit(previous, self.symbol_contexts)
        return previous - (1 << self.symbol_bits)


def decode_refinement_bitmap(
    decoder: JBIG2MQDecoder,
    contexts: list[int],
    reference: JBIG2Image,
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> JBIG2Image:
    """T.88 6.3 template 0 with nominal AT pixels and no typical prediction.

    Context bits are gathered in reading order, first the four target pixels,
    then the reference's nine pixels. Bit assignment is implementation-defined
    by 6.3.5.3; each assignment retains its own probability state.
    """
    image = JBIG2Image.create(width, height, allow_empty=True)
    if not width or not height:
        return image

    def pixel(bitmap: JBIG2Image, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= bitmap.width or y >= bitmap.height:
            return 0
        return (bitmap.data[y * bitmap.stride + (x >> 3)] >> (7 - (x & 7))) & 1

    for y in range(height):
        for x in range(width):
            context = 0
            for ox, oy in ((-1, -1), (0, -1), (1, -1), (-1, 0)):
                context = (context << 1) | pixel(image, x + ox, y + oy)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    context = (context << 1) | pixel(reference, x - dx + ox, y - dy + oy)
            if decoder.read_bit(context, contexts):
                image.data[y * image.stride + (x >> 3)] |= 0x80 >> (x & 7)
    return image


def decode_arithmetic_generic_bitmap(
    data: bytes,
    width: int,
    height: int,
    template: int,
    prediction: bool,
    at: tuple[tuple[int, int], ...],
) -> bytes | bytearray:
    if template != 0 or at != GENERIC_TEMPLATE_0_DEFAULT_AT or width <= 0 or height <= 0:
        raise Jbig2UnsupportedError("unsupported JBIG2 generic bitmap template")
    return decode_generic_bitmap(
        JBIG2MQDecoder.from_segment(data), width, height, prediction=prediction
    ).data


def decode_arithmetic_generic_template0(
    data: bytes, width: int, height: int, *, prediction: bool = False
) -> bytearray:
    return decode_generic_bitmap(JBIG2MQDecoder(data), width, height, prediction=prediction).data


def decode_generic_bitmap(
    decoder: JBIG2MQDecoder, width: int, height: int, *, prediction: bool = False
) -> JBIG2Image:
    """Decode template 0, retaining the coder and GB contexts between symbols."""
    read_bit = decoder.read_bit
    image = JBIG2Image.create(width, height, allow_empty=True)
    if not width or not height:
        return image
    row_byte_length = image.stride
    bitmap = image.data
    previous_row = bytearray(width + 4)
    previous_previous_row = bytearray(width + 4)
    old_pixel_mask = 0x7BF7
    typical_row = 0
    for row_index in range(height):
        if prediction:
            # T.88 6.2.5.7: SLTP toggles the typical-prediction state; a
            # predicted first row copies the implicit all-zero row above it.
            typical_row ^= read_bit(0x9B25)
        if typical_row:
            if row_index:
                offset = row_index * row_byte_length
                bitmap[offset : offset + row_byte_length] = bitmap[
                    offset - row_byte_length : offset
                ]
            previous_previous_row = previous_row
            continue
        # Four sentinel bytes eliminate bounds checks for the look-ahead
        # samples at col + 3 and col + 4 in the template-0 context.
        row = bytearray(width + 4)
        row1 = row if row_index < 1 else previous_row
        row2 = row if row_index < 2 else previous_previous_row
        context = (
            (row2[0] << 13)
            | (row2[1] << 12)
            | (row2[2] << 11)
            | (row1[0] << 7)
            | (row1[1] << 6)
            | (row1[2] << 5)
            | (row1[3] << 4)
        )
        for col in range(width):
            pixel = read_bit(context)
            row[col] = pixel
            if pixel:
                bitmap[row_index * row_byte_length + (col >> 3)] |= 0x80 >> (col & 7)
            context = (
                ((context & old_pixel_mask) << 1)
                | (row2[col + 3] << 11)
                | (row1[col + 4] << 4)
                | pixel
            )
        previous_previous_row = previous_row
        previous_row = row
    return image
