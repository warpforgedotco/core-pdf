# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass

from core_jpeg.impl.codecs.jpx.packets import JpxCodeBlockSegment, JpxPrecinct
from core_jpeg.impl.codecs.jpx.params import JpxCodingParams
from core_jpeg.impl.codecs.jpx.tier1_state import (
    T1_ORIENT_HH,
    T1_ORIENT_HL,
    T1_ORIENT_LH,
    T1_ORIENT_LL,
)
from core_jpeg.impl.errors import JpegParseError


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("invalid divisor")
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class JpxTilePart:
    tile_index: int
    tile_part_index: int
    tile_part_count: int
    coding_params: JpxCodingParams
    payload: bytes
    packet_headers: bytes | None = None


@dataclass(frozen=True)
class JpxTilePartHeader:
    tile_index: int
    tile_part_index: int
    tile_part_count: int
    coding_params: JpxCodingParams
    payload_start: int
    payload_end: int
    packet_headers: bytes | None = None


class SubBand:
    __slots__ = (
        "x0",
        "y0",
        "x1",
        "y1",
        "width",
        "height",
        "samples",
        "level",
        "is_ll",
        "orientation",
        "codeblock_w",
        "codeblock_h",
        "block_origin_x",
        "block_origin_y",
        "num_blocks_h",
        "num_blocks_v",
        "blocks",
        "precincts",
    )

    def __init__(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        level: int,
        is_ll: bool,
        cb_w: int,
        cb_h: int,
        orientation: int = T1_ORIENT_LL,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = max(x0, x1)
        self.y1 = max(y0, y1)
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.samples: list[int | float] = [0] * (self.width * self.height)
        self.level = level
        self.is_ll = is_ll
        self.orientation = orientation
        self.codeblock_w = cb_w
        self.codeblock_h = cb_h
        self.block_origin_x = self.x0 // cb_w
        self.block_origin_y = self.y0 // cb_h
        self.num_blocks_h = 0 if self.width <= 0 else ceil_div(self.x1, cb_w) - self.block_origin_x
        self.num_blocks_v = 0 if self.height <= 0 else ceil_div(self.y1, cb_h) - self.block_origin_y
        self.blocks = []
        for ignored in range(self.num_blocks_v * self.num_blocks_h):
            self.blocks.append(CodeBlock(cb_w * cb_h))
        self.precincts = []
        if self.num_blocks_h > 0 and self.num_blocks_v > 0:
            self.precincts.append(
                JpxPrecinct(
                    0,
                    0,
                    self.num_blocks_h,
                    self.num_blocks_v,
                )
            )


class ResSubBand:
    __slots__ = ("x0", "y0", "x1", "y1", "width", "height", "level", "subbands")

    def __init__(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        level: int,
        cb_w: int,
        cb_h: int,
        low_bounds: tuple[int, int, int, int] | None = None,
        high_band_bounds: tuple[
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ]
        | None = None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = max(x0, x1)
        self.y1 = max(y0, y1)
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.level = level
        self.subbands: list[SubBand] = []

        if level == 0:
            self.subbands.append(SubBand(x0, y0, x1, y1, level, True, cb_w, cb_h, T1_ORIENT_LL))
        else:
            if high_band_bounds is None:
                low_x0, low_y0, low_x1, low_y1 = (
                    low_bounds
                    if low_bounds is not None
                    else (
                        ceil_div(x0, 2),
                        ceil_div(y0, 2),
                        ceil_div(x1, 2),
                        ceil_div(y1, 2),
                    )
                )
                high_band_bounds = (
                    (low_x1, low_y0, x1, low_y1),
                    (low_x0, low_y1, low_x1, y1),
                    (low_x1, low_y1, x1, y1),
                )
            hl_bounds, lh_bounds, hh_bounds = high_band_bounds
            self.subbands.append(
                SubBand(
                    *hl_bounds,
                    level,
                    False,
                    cb_w,
                    cb_h,
                    T1_ORIENT_HL,
                )
            )
            self.subbands.append(
                SubBand(
                    *lh_bounds,
                    level,
                    False,
                    cb_w,
                    cb_h,
                    T1_ORIENT_LH,
                )
            )
            self.subbands.append(
                SubBand(
                    *hh_bounds,
                    level,
                    False,
                    cb_w,
                    cb_h,
                    T1_ORIENT_HH,
                )
            )


class TileComponent:
    __slots__ = (
        "width",
        "height",
        "x0",
        "y0",
        "h_sep",
        "v_sep",
        "codeblock_style",
        "resolutions",
    )

    def __init__(
        self,
        width: int,
        height: int,
        levels: int,
        cb_w: int,
        cb_h: int,
        *,
        x0: int = 0,
        y0: int = 0,
        h_sep: int = 1,
        v_sep: int = 1,
        codeblock_style: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.h_sep = max(1, h_sep)
        self.v_sep = max(1, v_sep)
        self.codeblock_style = codeblock_style
        self.resolutions: list[ResSubBand] = []
        x1 = x0 + width
        y1 = y0 + height
        bounds = [
            (
                ceil_div(x0, 1 << (levels - res)),
                ceil_div(y0, 1 << (levels - res)),
                ceil_div(x1, 1 << (levels - res)),
                ceil_div(y1, 1 << (levels - res)),
            )
            for res in range(levels + 1)
        ]
        self.resolutions.append(ResSubBand(*bounds[0], 0, cb_w, cb_h))
        for res in range(1, levels + 1):
            prev_bounds = bounds[res - 1]
            current_bounds = bounds[res]
            level = levels - res + 1
            level_no = levels - res
            low_scale = 1 << (level_no + 1)
            high_offset = 1 << level_no
            high_band_bounds = (
                (
                    ceil_div(x0 - high_offset, low_scale),
                    ceil_div(y0, low_scale),
                    ceil_div(x1 - high_offset, low_scale),
                    ceil_div(y1, low_scale),
                ),
                (
                    ceil_div(x0, low_scale),
                    ceil_div(y0 - high_offset, low_scale),
                    ceil_div(x1, low_scale),
                    ceil_div(y1 - high_offset, low_scale),
                ),
                (
                    ceil_div(x0 - high_offset, low_scale),
                    ceil_div(y0 - high_offset, low_scale),
                    ceil_div(x1 - high_offset, low_scale),
                    ceil_div(y1 - high_offset, low_scale),
                ),
            )
            self.resolutions.append(
                ResSubBand(
                    *current_bounds,
                    level,
                    cb_w,
                    cb_h,
                    prev_bounds,
                    high_band_bounds,
                )
            )


class CodeBlock:
    __slots__ = ("data", "segments")

    def __init__(self, size: int) -> None:
        self.data = [0] * size
        self.segments: list[JpxCodeBlockSegment] = []


class BitStream:
    __slots__ = ("data", "byte")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte = 0

    def read_u16(self) -> int:
        return (self.read_byte() << 8) | self.read_byte()

    def read_u32(self) -> int:
        return (
            (self.read_byte() << 24)
            | (self.read_byte() << 16)
            | (self.read_byte() << 8)
            | self.read_byte()
        )

    def read_byte(self) -> int:
        if self.byte >= len(self.data):
            raise JpegParseError("unexpected end of JPX codestream")
        b = self.data[self.byte]
        self.byte += 1
        return b

    def read_bytes(self, n: int) -> bytes:
        if n < 0:
            raise JpegParseError("invalid JPX read length")
        if self.byte + n > len(self.data):
            raise JpegParseError("unexpected end of JPX codestream")
        data = self.data[self.byte : self.byte + n]
        self.byte += n
        return data

    def skip_bytes(self, n: int) -> None:
        if n < 0:
            raise JpegParseError("invalid JPX skip length")
        if self.byte + n > len(self.data):
            raise JpegParseError("unexpected end of JPX codestream")
        self.byte += n
