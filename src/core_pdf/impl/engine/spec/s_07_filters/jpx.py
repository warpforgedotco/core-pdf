# SPDX-License-Identifier: AGPL-3.0-only
"""JPEG 2000 (JPXDecode) stream decoder.

Minimal pure-Python JPEG 2000 decoder for the subset commonly used
in PDF documents: greyscale images with 5/3 lossless wavelet transform.

References:
  - ITU-T T.800 (JPEG 2000 core coding)
  - OpenJPEG reference implementation
"""

from __future__ import annotations

from typing import TypedDict

from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError, PdfUnsupportedError


class ComponentData(TypedDict):
    precision: int
    is_signed: bool
    h_sep: int
    v_sep: int


class TileData(TypedDict):
    x: int
    y: int
    components: list["TileComponent"]


# ---------------------------------------------------------------------------
# MQ Arithmetic Decoder
# ---------------------------------------------------------------------------

_QE = (
    0x5601,
    0x3401,
    0x1801,
    0x0AC1,
    0x0521,
    0x0221,
    0x5601,
    0x5401,
    0x4801,
    0x3801,
    0x3001,
    0x2401,
    0x1C01,
    0x1601,
    0x5601,
    0x5401,
    0x5101,
    0x4801,
    0x3801,
    0x3401,
    0x3001,
    0x2801,
    0x2401,
    0x2201,
    0x1C01,
    0x1801,
    0x1601,
    0x1401,
    0x1201,
    0x1101,
    0x0AC1,
    0x09C1,
    0x08A1,
    0x0521,
    0x0441,
    0x02A1,
    0x0221,
    0x0141,
    0x0111,
    0x0085,
    0x0049,
    0x0025,
    0x0015,
    0x0009,
    0x0005,
    0x0001,
    0x5601,
)

_NMPS = (
    1,
    2,
    3,
    4,
    5,
    38,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    29,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    45,
    46,
)

_NLPS = (
    1,
    6,
    9,
    12,
    29,
    33,
    6,
    14,
    14,
    14,
    17,
    18,
    20,
    21,
    14,
    14,
    15,
    16,
    17,
    18,
    19,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
)

_SWITCH = (
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
)

# Context labels
CX_UNIFORM = 17
CX_SIG = 12
CX_SIGN = 13
CX_REFINE = 16
CX_MAG0 = 5
CX_MAGBITS = (8, 9, 10, 11, 12, 13)


class MQDecoder:
    """MQ arithmetic decoder following ITU-T T.800 Annex C."""

    __slots__ = ("data", "pos", "_a", "_c", "ct", "ctx")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.ctx = [(0, 0)] * 19  # (index, mps) per context
        self._a = 0x8000
        self._c = 0
        self.ct = 0
        self.init_bytes()

    def read_byte(self) -> int:
        b = self.data[self.pos]
        self.pos += 1
        return b

    def init_bytes(self) -> None:
        b = self.read_byte()
        if b == 0xFF:
            b = self.read_byte()
        self._c = b << 8
        b = self.read_byte()
        if b == 0xFF:
            b = self.read_byte()
        self._c |= b
        self.ct = 0

    def byte_in(self) -> None:
        if self.pos >= len(self.data):
            self._c |= 0xFF
            self.ct = 8
        else:
            b = self.read_byte()
            if b == 0xFF:
                self.pos += 1
            self._c |= b << self.ct
            self.ct = 8

    def renormalize(self) -> None:
        while self._a < 0x8000:
            if self.ct == 0:
                self.byte_in()
            self._c <<= 1
            self._a <<= 1
            self.ct -= 1

    def decode(self, cx: int) -> int:
        """Decode one bit using context *cx* (0..18). Returns 0 or 1."""
        idx, mps = self.ctx[cx]
        qe = _QE[idx]
        self._a -= qe
        if (self._c >> 16) < qe:
            if self._a < qe:
                self._a = qe
            else:
                mps ^= 1
            self.ctx[cx] = (_NLPS[idx], mps)
            if _SWITCH[idx]:
                mps ^= 1
            self.renormalize()
            return mps ^ 1
        self._c -= qe << 16
        if self._a < 0x8000:
            self.ctx[cx] = (_NMPS[idx], mps)
            self.renormalize()
        return mps


# ---------------------------------------------------------------------------
# Decoder State
# ---------------------------------------------------------------------------

# Progression orders
LRCP = 0  # layer-resolution-component-position
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

DEFAULT_CONTEXTS = (4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)


def ilog2(n: int) -> int:
    if n <= 0:
        return 0
    r = 0
    while n > 1:
        r += 1
        n >>= 1
    return r


class JpxImage:
    """JPEG 2000 image decoder."""

    __slots__ = (
        "width",
        "height",
        "components",
        "capabilities",
        "tiles_cols",
        "tiles_rows",
        "levels",
        "codeblock_w",
        "codeblock_h",
        "prog_order",
        "num_layers",
        "multiple_component_transform",
        "precincts",
        "quant_steps",
        "components_data",
        "tiles",
        "reversible",
        "negate",
        "swap_bytes",
    )

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.components = 0
        self.capabilities = 0
        self.tiles_cols = 0
        self.tiles_rows = 0
        self.levels = 0
        self.codeblock_w = 0
        self.codeblock_h = 0
        self.prog_order = 0
        self.num_layers = 0
        self.multiple_component_transform = 0
        self.precincts: list[object] = []
        self.quant_steps: list[list[tuple[int, int]]] = []
        self.components_data: list[ComponentData] = []
        self.tiles: list[TileData] = []
        self.reversible = False
        self.negate = False
        self.swap_bytes = False

    def parse(self, data: bytes) -> bool:
        """Parse a JPEG 2000 codestream and return True on success."""
        br = BitStream(data)
        marker = br.read_u16()
        if marker != 0xFF4F:
            return False
        if not self.parse_header(br):
            return False
        return self.parse_tiles(br)

    def parse_header(self, br: BitStream) -> bool:
        while True:
            marker = br.read_u16()
            if marker == 0xFF51:
                self.parse_siz(br)
            elif marker == 0xFF52:
                self.parse_cod(br)
            elif marker == 0xFF5C:
                self.parse_qcd(br)
            else:
                return marker == 0xFF90 or marker == 0xFF93 or marker == 0xFFD9

    def parse_siz(self, br: BitStream) -> None:
        lsiz = br.read_u16()
        if lsiz < 41:
            raise ValueError("SIZ too short")
        self.capabilities = br.read_u16()
        self.width = br.read_u32()
        self.height = br.read_u32()
        br.read_u32()
        br.read_u32()
        tile_w = br.read_u32()
        tile_h = br.read_u32()
        br.read_u32()
        br.read_u32()
        self.components = br.read_u16()
        if self.components == 0:
            raise ValueError("zero components")
        self.components_data = []
        for _ in range(self.components):
            self.components_data.append(
                {
                    "precision": br.read_byte() & 0x7F,
                    "is_signed": bool(br.read_byte() & 0x80),
                    "h_sep": br.read_byte(),
                    "v_sep": br.read_byte(),
                }
            )
        self.tiles_cols = (self.width + tile_w - 1) // tile_w if tile_w else 0
        self.tiles_rows = (self.height + tile_h - 1) // tile_h if tile_h else 0

    def parse_cod(self, br: BitStream) -> None:
        lcod = br.read_u16()
        if lcod < 8:
            raise ValueError("COD too short")
        br.read_byte()
        self.prog_order = br.read_byte()
        self.num_layers = br.read_u16()
        self.multiple_component_transform = br.read_byte()
        levels = br.read_byte()
        self.levels = levels
        cb_w = br.read_byte()
        cb_h = br.read_byte()
        self.codeblock_w = 1 << (cb_w + 2)
        self.codeblock_h = 1 << (cb_h + 2)
        br.read_byte()
        wavelet = br.read_byte()
        self.reversible = wavelet == 1

    def parse_qcd(self, br: BitStream) -> None:
        br.read_u16()
        guard = br.read_byte()
        style = (guard >> 5) & 7
        self.quant_steps = []
        steps: list[tuple[int, int]] = []
        if style == 0:
            for _ in range(3 * self.levels + 1):
                mantissa = br.read_u16()
                if mantissa == 0:
                    raise ValueError("bad quant step")
                exp = mantissa >> 11
                steps.append((mantissa & 0x7FF, exp))
        self.quant_steps.append(steps)

    def parse_tiles(self, br: BitStream) -> bool:
        self.tiles = []
        for ty in range(self.tiles_rows):
            for tx in range(self.tiles_cols):
                tile: TileData = {
                    "x": tx,
                    "y": ty,
                    "components": [],
                }
                tile_w = self.width // self.tiles_cols
                tile_h = self.height // self.tiles_rows
                for c in range(self.components):
                    tile["components"].append(
                        TileComponent(
                            tile_w,
                            tile_h,
                            self.levels,
                            self.codeblock_w,
                            self.codeblock_h,
                        )
                    )
                self.tiles.append(tile)
                if not self.parse_tile(br, tile, tile_w, tile_h):
                    return False
                if not self.decode_tile(tile):
                    return False
        return True

    def parse_tile(self, br: BitStream, tile: TileData, w: int, h: int) -> bool:
        while True:
            marker = br.read_u16()
            if marker == 0xFF90:
                self.parse_sot(br)
            elif marker == 0xFF93:
                br.bit_pos()
                for comp in tile["components"]:
                    if not self.decode_packets(
                        br,
                        w,
                        h,
                        self.levels,
                        self.codeblock_w,
                        self.codeblock_h,
                        comp,
                    ):
                        return False
                br.bit_pos()
                # skip to EOC
                # For now, just break
                return True
            elif marker == 0xFFD9:
                return True
            else:
                length = br.read_u16()
                if length >= 2:
                    br.skip_bytes(length - 2)
        return True

    def parse_sot(self, br: BitStream) -> None:
        br.read_u16()
        br.read_u16()
        br.read_u32()
        br.read_byte()
        br.read_byte()

    def decode_packets(
        self,
        br: BitStream,
        w: int,
        h: int,
        levels: int,
        cb_w: int,
        cb_h: int,
        comp: TileComponent,
    ) -> bool:
        # For a minimal decoder, decode all packets in LRCP order
        for r in range(levels + 1):
            if not self.decode_packet(br, r, comp, w, h, levels, cb_w, cb_h):
                return False
        return True

    def decode_packet(
        self,
        br: BitStream,
        res: int,
        comp: TileComponent,
        w: int,
        h: int,
        levels: int,
        cb_w: int,
        cb_h: int,
    ) -> bool:
        subband = comp.resolutions[res]
        for sub in subband.subbands:
            cw = (cb_w) >> sub.level if sub.level > 0 else cb_w
            ch = (cb_h) >> sub.level if sub.level > 0 else cb_h
            for by in range(sub.num_blocks_v):
                for bx in range(sub.num_blocks_h):
                    block = sub.blocks[by * sub.num_blocks_h + bx]
                    w2 = cw if bx < sub.num_blocks_h - 1 else sub.width - bx * cw
                    h2 = ch if by < sub.num_blocks_v - 1 else sub.height - by * ch
                    if not self.decode_block(br, block, w2, h2, sub.is_ll):
                        return False
        return True

    def decode_block(
        self,
        br: BitStream,
        block: CodeBlock,
        block_w: int,
        block_h: int,
        is_ll: bool,
    ) -> bool:
        data = block.data
        # Find max bit-plane
        max_val = 0
        for i in range(block_h * block_w):
            v = data[i]
            if v < 0:
                v = -v
            if v > max_val:
                max_val = v
        if max_val == 0:
            return True

        num_bps = ilog2(max_val) + 1
        num_passes = 3 * num_bps - 2

        # Parse passes
        first_sig_pass = 0
        while first_sig_pass < num_passes:
            pass_type = first_sig_pass % 3
            # Parse packet header to get pass lengths
            # For now, we assume all data is in one segment
            if not self.decode_pass(
                br, block, pass_type, num_bps, first_sig_pass, block_w, block_h, is_ll
            ):
                return False
            first_sig_pass += 1

        return True

    def decode_pass(
        self,
        br: BitStream,
        block: CodeBlock,
        pass_type: int,
        num_bps: int,
        pass_num: int,
        block_w: int,
        block_h: int,
        is_ll: bool,
    ) -> bool:
        bp = num_bps - 1 - (pass_num // 3)
        num = block_h * block_w
        data = block.data
        significance = block.significance

        if pass_type == 0:
            # Cleanup pass
            for i in range(num):
                if data[i] == 0 and significance[i] == 0 and br.read_bit():
                    data[i] = 1 << bp
                    if br.read_bit():
                        data[i] = -data[i]
        elif pass_type == 1:
            # Significance propagation pass
            for i in range(num):
                if data[i] == 0 and significance[i] == 0:
                    # Check if neighbor is significant
                    col = i % block_w
                    row = i // block_w
                    neighbor_sig = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < block_h and 0 <= nc < block_w:
                                ni = nr * block_w + nc
                                if significance[ni]:
                                    neighbor_sig = True
                                    break
                        if neighbor_sig:
                            break
                    if not neighbor_sig:
                        continue
                    if br.read_bit():
                        data[i] = 1 << bp
                        if br.read_bit():
                            data[i] = -data[i]
        elif pass_type == 2:
            # Magnitude refinement pass
            for i in range(num):
                if significance[i] and data[i] >= 0:
                    bp_val = 1 << bp
                    if (abs(data[i]) & bp_val) == 0:
                        pass
                    elif br.read_bit():
                        data[i] += bp_val

        # Update significance
        for i in range(num):
            if data[i] != 0 and significance[i] == 0:
                significance[i] = 1

        return True

    def decode_tile(self, tile: TileData) -> bool:
        for comp in tile["components"]:
            self.build_image(comp)
        return True

    def build_image(self, comp: TileComponent) -> None:
        # Inverse DWT
        for r in range(len(comp.resolutions)):
            resolution = comp.resolutions[r]
            if r == 0:
                continue
            for subband in resolution.subbands:
                self.inverse_dwt_53(subband)

    def inverse_dwt_53(self, subband: SubBand) -> None:
        width = subband.width
        height = subband.height
        samples = subband.samples
        # Horizontal IDWT
        for y in range(height):
            row = y * width
            idwt_53(samples, row, width, 1)
        # Vertical IDWT
        for x in range(width):
            idwt_53(samples, x, height, width)

    def to_raw(self) -> bytes:
        if not self.tiles:
            raise PdfUnsupportedError("JPXDecode produced no image tiles")
        comp = self.tiles[0]["components"][0]
        r0 = comp.resolutions[0].subbands[0]
        w = r0.width
        h = r0.height
        out = bytearray(w * h)
        for i in range(w * h):
            v = r0.samples[i]
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            out[i] = v
        return bytes(out)


def idwt_53(samples: list[int], offset: int, length: int, stride: int) -> None:
    if length <= 1:
        return
    step = 2
    while step <= length:
        half = step >> 1
        temp = [0] * step
        for i in range(step):
            temp[i] = samples[offset + i * stride]
        # Inverse lifting
        for i in range(half):
            samples[offset + (i * 2) * stride] = temp[i]
            samples[offset + (i * 2 + 1) * stride] = temp[half + i]
        for i in range(half):
            idx = offset + (i * 2 + 1) * stride
            left = samples[idx - stride] if i > 0 else 0
            right = samples[idx + stride] if i < half - 1 else 0
            samples[idx] += (left + right) >> 1
        for i in range(half):
            idx = offset + (i * 2) * stride
            left = samples[idx - stride] if i > 0 else 0
            right = samples[idx + stride] if i < half - 1 else 0
            samples[idx] -= (left + right + 2) >> 2
        step *= 2


class SubBand:
    __slots__ = (
        "width",
        "height",
        "samples",
        "level",
        "is_ll",
        "num_blocks_h",
        "num_blocks_v",
        "blocks",
    )

    def __init__(
        self, width: int, height: int, level: int, is_ll: bool, cb_w: int, cb_h: int
    ) -> None:
        self.width = width
        self.height = height
        self.samples = [0] * (width * height)
        self.level = level
        self.is_ll = is_ll
        self.num_blocks_h = (width + cb_w - 1) // cb_w
        self.num_blocks_v = (height + cb_h - 1) // cb_h
        self.blocks = []
        for _ in range(self.num_blocks_v * self.num_blocks_h):
            self.blocks.append(CodeBlock(cb_w * cb_h))


class ResSubBand:
    __slots__ = ("width", "height", "level", "subbands")

    def __init__(self, width: int, height: int, level: int, cb_w: int, cb_h: int) -> None:
        self.width = width
        self.height = height
        self.level = level
        self.subbands: list[SubBand] = []
        # LL only at highest level
        if level == 0:
            self.subbands.append(SubBand(width, height, level, True, cb_w, cb_h))
        else:
            self.subbands.append(SubBand(width, height, level, True, cb_w, cb_h))
            self.subbands.append(SubBand(width, height, level, False, cb_w, cb_h))
            self.subbands.append(SubBand(width, height, level, False, cb_w, cb_h))
            self.subbands.append(SubBand(width, height, level, False, cb_w, cb_h))


class TileComponent:
    __slots__ = ("width", "height", "resolutions")

    def __init__(self, width: int, height: int, levels: int, cb_w: int, cb_h: int) -> None:
        self.width = width
        self.height = height
        self.resolutions: list[ResSubBand] = []
        w, h = width, height
        for r in range(levels, -1, -1):
            self.resolutions.append(ResSubBand(w, h, r, cb_w, cb_h))
            w = (w + 1) // 2
            h = (h + 1) // 2


class CodeBlock:
    __slots__ = ("data", "significance")

    def __init__(self, size: int) -> None:
        self.data = [0] * size
        self.significance = [0] * size


# ---------------------------------------------------------------------------
# Bit stream helpers
# ---------------------------------------------------------------------------


class BitStream:
    __slots__ = ("data", "byte", "bitpos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte = 0
        self.bitpos = 0

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
        if self.bitpos >= len(self.data):
            raise PdfParseError("unexpected end of JPX codestream")
        b = self.data[self.bitpos]
        self.bitpos += 1
        return b

    def read_bit(self) -> int:
        if self.byte >= len(self.data):
            raise PdfParseError("unexpected end of JPX codestream")
        bit = (self.data[self.byte] >> (7 - self.bitpos)) & 1
        self.bitpos += 1
        if self.bitpos >= 8:
            self.bitpos = 0
            self.byte += 1
        return bit

    def bit_pos(self) -> int:
        return self.byte * 8 + self.bitpos

    def skip_bytes(self, n: int) -> None:
        if n < 0:
            raise PdfParseError("invalid JPX skip length")
        if self.bitpos:
            self.byte += 1
            self.bitpos = 0
        if self.byte + n > len(self.data):
            raise PdfParseError("unexpected end of JPX codestream")
        self.byte += n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decode_jpx(data: bytes) -> bytes:
    """Decode a JPEG 2000 codestream. Returns raw 8-bit greyscale pixel data."""
    img = JpxImage()
    if img.parse(data):
        return img.to_raw()
    raise PdfUnsupportedError("JPXDecode failed to parse codestream")
