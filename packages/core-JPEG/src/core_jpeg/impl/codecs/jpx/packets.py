# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_jpeg.impl.codecs.jpx.params import (
    JPX_CODEBLOCK_STYLE_LAZY,
    JPX_CODEBLOCK_STYLE_TERMALL,
)
from core_jpeg.impl.errors import JpegParseError


@dataclass(frozen=True)
class JpxPacketSegment:
    block_index: int
    segment_index: int
    num_passes: int
    length: int
    zero_bit_planes: int


@dataclass(frozen=True)
class JpxCodeBlockChunk:
    block_index: int
    segment_index: int
    num_passes: int
    zero_bit_planes: int
    payload: bytes


@dataclass
class JpxCodeBlockSegment:
    num_passes: int
    zero_bit_planes: int
    payload: bytes


@dataclass
class JpxCodewordSegmentState:
    max_passes: int
    passes: int = 0


@dataclass(frozen=True)
class JpxDecodedPacket:
    chunks: list[JpxCodeBlockChunk]
    consumed: int
    header_consumed: int = 0


@dataclass(frozen=True)
class JpxPacketStreamConsumed:
    body: int
    header: int = 0
    positions: int = 0


@dataclass(frozen=True)
class JpxPacketPosition:
    layer: int
    resolution: int
    component: int
    precinct: int
    codeblock_style: int
    bands: tuple[Any, ...]
    precinct_states: tuple[JpxPrecinct, ...]


class PacketBitReader:
    __slots__ = ("data", "byte", "bitpos")

    def __init__(self, data: bytes, offset: int = 0) -> None:
        if offset < 0:
            raise JpegParseError("invalid JPX packet header offset")
        self.data = data
        self.byte = offset
        self.bitpos = 0

    def read_bit(self) -> int:
        if self.byte >= len(self.data):
            raise JpegParseError("unexpected end of JPX packet header")
        bit = (self.data[self.byte] >> (7 - self.bitpos)) & 1
        self.bitpos += 1
        if self.bitpos == 8:
            prev = self.data[self.byte]
            self.byte += 1
            self.bitpos = 1 if prev == 0xFF and self.byte < len(self.data) else 0
        return bit

    def read_bits(self, count: int) -> int:
        value = 0
        for ignored in range(count):
            value = (value << 1) | self.read_bit()
        return value

    def align(self) -> int:
        if self.bitpos:
            extra_byte = self.data[self.byte] == 0xFF and self.byte + 1 < len(self.data)
            self.byte += 1
            if extra_byte:
                self.byte += 1
            self.bitpos = 0
        return self.byte


class JpxTagTree:
    __slots__ = ("width", "height", "levels", "values", "lows")

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("invalid JPX tag-tree dimensions")
        self.width = width
        self.height = height
        self.levels: list[tuple[int, int]] = []
        w, h = width, height
        while True:
            self.levels.append((w, h))
            if w == 1 and h == 1:
                break
            w = (w + 1) // 2
            h = (h + 1) // 2
        self.values = [[999] * (w * h) for w, h in self.levels]
        self.lows = [[0] * (w * h) for w, h in self.levels]

    def decode_less_than(self, reader: PacketBitReader, x: int, y: int, threshold: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError("JPX tag-tree coordinate out of range")
        low = 0
        path: list[tuple[int, int, int]] = []
        for level, (w, ignored_h) in enumerate(self.levels):
            path.append((level, x, y))
            x //= 2
            y //= 2
        leaf_level, leaf_x, leaf_y = path[0]
        leaf_index = leaf_y * self.levels[leaf_level][0] + leaf_x
        for level, x, y in reversed(path):
            w, ignored_h = self.levels[level]
            index = y * w + x
            if low > self.lows[level][index]:
                self.lows[level][index] = low
            else:
                low = self.lows[level][index]
            while low < threshold and low < self.values[level][index]:
                if reader.read_bit():
                    self.values[level][index] = low
                else:
                    low += 1
            self.lows[level][index] = low
        return self.values[leaf_level][leaf_index] < threshold

    def decode_value(self, reader: PacketBitReader, x: int, y: int) -> int:
        threshold = 1
        while not self.decode_less_than(reader, x, y, threshold):
            threshold += 1
        return threshold - 1


class JpxPacketCodeBlockState:
    __slots__ = ("included", "lblock", "passes", "segments", "zero_bit_planes")

    def __init__(self) -> None:
        self.included = False
        self.lblock = 3
        self.passes = 0
        self.segments: list[JpxCodewordSegmentState] = []
        self.zero_bit_planes = -1


class JpxPrecinct:
    __slots__ = (
        "block_x0",
        "block_y0",
        "blocks_w",
        "blocks_h",
        "packet_states",
        "inclusion",
        "zero_bit_planes",
    )

    def __init__(
        self,
        block_x0: int,
        block_y0: int,
        blocks_w: int,
        blocks_h: int,
    ) -> None:
        if blocks_w < 0 or blocks_h < 0:
            raise ValueError("invalid JPX precinct code-block dimensions")
        self.block_x0 = block_x0
        self.block_y0 = block_y0
        self.blocks_w = blocks_w
        self.blocks_h = blocks_h
        self.packet_states = [JpxPacketCodeBlockState() for ignored in range(blocks_w * blocks_h)]
        self.inclusion: JpxTagTree | None
        self.zero_bit_planes: JpxTagTree | None
        if blocks_w > 0 and blocks_h > 0:
            self.inclusion = JpxTagTree(blocks_w, blocks_h)
            self.zero_bit_planes = JpxTagTree(blocks_w, blocks_h)
        else:
            self.inclusion = None
            self.zero_bit_planes = None

    def global_block_index(self, local_index: int, subband_blocks_w: int) -> int:
        local_x = local_index % self.blocks_w
        local_y = local_index // self.blocks_w
        return (self.block_y0 + local_y) * subband_blocks_w + self.block_x0 + local_x


def decode_packet_pass_count(reader: PacketBitReader) -> int:
    if reader.read_bit() == 0:
        return 1
    if reader.read_bit() == 0:
        return 2
    value = reader.read_bits(2)
    if value != 3:
        return value + 3
    value = reader.read_bits(5)
    if value != 31:
        return value + 6
    return reader.read_bits(7) + 37


def codeblock_segment_max_passes(
    codeblock_style: int,
    previous_max_passes: int | None,
) -> int:
    if codeblock_style & JPX_CODEBLOCK_STYLE_TERMALL:
        return 1
    if codeblock_style & JPX_CODEBLOCK_STYLE_LAZY:
        if previous_max_passes is None:
            return 10
        return 2 if previous_max_passes in (1, 10) else 1
    return 109


def append_packet_codeword_segment(
    state: JpxPacketCodeBlockState,
    codeblock_style: int,
) -> JpxCodewordSegmentState:
    previous_max_passes = state.segments[-1].max_passes if state.segments else None
    segment = JpxCodewordSegmentState(
        max_passes=codeblock_segment_max_passes(
            codeblock_style,
            previous_max_passes,
        )
    )
    state.segments.append(segment)
    return segment


def decode_packet_codeblock_segments(
    reader: PacketBitReader,
    block_index: int,
    block_x: int,
    block_y: int,
    layer: int,
    state: JpxPacketCodeBlockState,
    inclusion: JpxTagTree,
    zero_bit_planes: JpxTagTree,
    codeblock_style: int,
) -> list[JpxPacketSegment]:
    if state.included:
        included = bool(reader.read_bit())
    else:
        included = inclusion.decode_less_than(reader, block_x, block_y, layer + 1)
    if not included:
        return []

    if state.zero_bit_planes < 0:
        state.zero_bit_planes = zero_bit_planes.decode_value(reader, block_x, block_y)

    num_new_passes = decode_packet_pass_count(reader)
    while reader.read_bit():
        state.lblock += 1
    state.included = True

    remaining_passes = num_new_passes
    packet_segments: list[JpxPacketSegment] = []
    while remaining_passes > 0:
        if not state.segments or state.segments[-1].passes == state.segments[-1].max_passes:
            segment = append_packet_codeword_segment(state, codeblock_style)
            segment_index = len(state.segments) - 1
        else:
            segment = state.segments[-1]
            segment_index = len(state.segments) - 1
        segment_passes = min(
            segment.max_passes - segment.passes,
            remaining_passes,
        )
        if segment_passes <= 0:
            raise JpegParseError("invalid JPX code-block segment pass count")
        length_bits = _ilog2(segment_passes) + state.lblock
        length = reader.read_bits(length_bits)
        segment.passes += segment_passes
        state.passes += segment_passes
        remaining_passes -= segment_passes
        packet_segments.append(
            JpxPacketSegment(
                block_index=block_index,
                segment_index=segment_index,
                num_passes=segment_passes,
                length=length,
                zero_bit_planes=state.zero_bit_planes,
            )
        )
    return packet_segments


def decode_packet_codeblock_chunks(
    packet: bytes,
    blocks_w: int,
    blocks_h: int,
    layer: int,
    states: list[JpxPacketCodeBlockState],
    inclusion: JpxTagTree,
    zero_bit_planes: JpxTagTree,
    codeblock_style: int = 0,
) -> tuple[list[JpxCodeBlockChunk], int]:
    if blocks_w <= 0 or blocks_h <= 0:
        return [], 0
    if len(states) != blocks_w * blocks_h:
        raise ValueError("JPX packet state count does not match code-block grid")
    reader = PacketBitReader(packet)
    if reader.read_bit() == 0:
        return [], reader.align()
    segments: list[JpxPacketSegment] = []
    for block_y in range(blocks_h):
        for block_x in range(blocks_w):
            block_index = block_y * blocks_w + block_x
            block_segments = decode_packet_codeblock_segments(
                reader,
                block_index=block_index,
                block_x=block_x,
                block_y=block_y,
                layer=layer,
                state=states[block_index],
                inclusion=inclusion,
                zero_bit_planes=zero_bit_planes,
                codeblock_style=codeblock_style,
            )
            segments.extend(block_segments)

    offset = reader.align()
    chunks: list[JpxCodeBlockChunk] = []
    for segment in segments:
        end = offset + segment.length
        if end > len(packet):
            raise JpegParseError("unexpected end of JPX packet body")
        chunks.append(
            JpxCodeBlockChunk(
                block_index=segment.block_index,
                segment_index=segment.segment_index,
                num_passes=segment.num_passes,
                zero_bit_planes=segment.zero_bit_planes,
                payload=packet[offset:end],
            )
        )
        offset = end
    return chunks, offset


def skip_sop_marker(packet: bytes, offset: int, packet_uses_sop: bool) -> int:
    if not packet_uses_sop or packet[offset : offset + 2] != b"\xff\x91":
        return offset
    if offset + 4 > len(packet):
        raise JpegParseError("truncated JPX SOP marker")
    length = int.from_bytes(packet[offset + 2 : offset + 4], "big")
    if length < 2 or offset + 2 + length > len(packet):
        raise JpegParseError("invalid JPX SOP marker length")
    return offset + 2 + length


def skip_eph_marker(packet: bytes, offset: int, packet_uses_eph: bool) -> int:
    if packet_uses_eph and packet[offset : offset + 2] == b"\xff\x92":
        return offset + 2
    return offset


def append_codeblock_chunk(
    block: Any,
    segment_index: int,
    num_passes: int,
    zero_bit_planes: int,
    payload: bytes,
) -> None:
    if segment_index < 0:
        raise JpegParseError("invalid JPX code-block segment index")
    while len(block.segments) <= segment_index:
        block.segments.append(
            JpxCodeBlockSegment(
                num_passes=0,
                zero_bit_planes=zero_bit_planes,
                payload=b"",
            )
        )
    segment = block.segments[segment_index]
    if segment.zero_bit_planes != zero_bit_planes:
        raise JpegParseError("JPX code-block zero bit-plane count changed")
    segment.num_passes += num_passes
    segment.payload += payload


def _ilog2(n: int) -> int:
    if n <= 0:
        return 0
    result = 0
    while n > 1:
        result += 1
        n >>= 1
    return result
