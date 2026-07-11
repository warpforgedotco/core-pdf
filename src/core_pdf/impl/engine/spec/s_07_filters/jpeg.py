"""JPEG stream decoding helpers."""

from __future__ import annotations

import math
import struct

from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError, PdfUnsupportedError

PASS1_BITS = 2
FAST_HUFFMAN_BITS = 9

ZIGZAG = [
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
    63,
]

SCALEBITS = 16
ONE_HALF = 1 << (SCALEBITS - 1)
FIX_1_40200 = 91881
FIX_1_77200 = 116130
FIX_0_34414 = 22554
FIX_0_71414 = 46802

CB_TO_B = tuple(((FIX_1_77200 * (i - 128) + ONE_HALF) >> SCALEBITS) for i in range(256))
CB_TO_G = tuple((-FIX_0_34414 * (i - 128)) + ONE_HALF for i in range(256))
CR_TO_R = tuple(((FIX_1_40200 * (i - 128) + ONE_HALF) >> SCALEBITS) for i in range(256))
CR_TO_G = tuple((-FIX_0_71414 * (i - 128)) for i in range(256))

ZERO_BLOCK_64 = [0] * 64

IDCT_SCALE = tuple(1 / math.sqrt(2) if i == 0 else 1.0 for i in range(8))
IDCT_COS = tuple(
    tuple(math.cos(((2 * x + 1) * u * math.pi) / 16.0) for u in range(8)) for x in range(8)
)


class JpegBitReader:
    __slots__ = ("data", "pos", "buffer", "bits_left", "len")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.buffer = 0
        self.bits_left = 0
        self.len = len(data)

    def fill_byte(self) -> None:
        data = self.data
        pos = self.pos
        if pos >= self.len:
            raise PdfUnsupportedError("Unexpected end of JPEG scan data")
        byte = data[pos]
        pos += 1
        if byte == 0xFF and pos < self.len and data[pos] == 0x00:
            pos += 1
        self.pos = pos
        self.buffer = (self.buffer << 8) | byte
        self.bits_left += 8

    def get_bits(self, n: int) -> int:
        if n == 0:
            return 0
        while self.bits_left < n and self.pos < self.len:
            self.fill_byte()
        bits_left = self.bits_left
        if bits_left < n:
            missing = n - bits_left
            value = ((self.buffer << missing) | ((1 << missing) - 1)) & ((1 << n) - 1)
            self.bits_left = 0
            self.buffer = 0
            return value
        bits_left -= n
        value = (self.buffer >> bits_left) & ((1 << n) - 1)
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0
        return value

    def peek_bits(self, n: int) -> int:
        while self.bits_left < n and self.pos < self.len:
            self.fill_byte()
        bits_left = self.bits_left
        if bits_left < n:
            missing = n - bits_left
            return ((self.buffer << missing) | ((1 << missing) - 1)) & ((1 << n) - 1)
        return (self.buffer >> (bits_left - n)) & ((1 << n) - 1)

    def drop_bits(self, n: int) -> None:
        if n >= self.bits_left:
            self.bits_left = 0
            self.buffer = 0
            return
        bits_left = self.bits_left - n
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0

    def get_bit(self) -> int:
        bits_left = self.bits_left
        if bits_left == 0:
            if self.pos >= self.len:
                raise PdfParseError("unexpected end of JPEG scan data")
            self.fill_byte()
            bits_left = self.bits_left
        bits_left -= 1
        bit = (self.buffer >> bits_left) & 1
        self.bits_left = bits_left
        if bits_left:
            self.buffer &= (1 << bits_left) - 1
        else:
            self.buffer = 0
        return bit


def build_huffman_table(
    lengths: bytes, symbols: bytes
) -> tuple[list[int], list[int], list[dict[int, int]]]:
    huff = [{} for _ in range(16)]
    code = 0
    k = 0
    for i, num in enumerate(lengths):
        table = huff[i]
        for _ in range(num):
            if k >= len(symbols):
                raise PdfParseError("truncated JPEG Huffman table")
            symbol = symbols[k]
            table[code] = symbol
            k += 1
            code += 1
        code <<= 1
    if k != len(symbols):
        raise PdfParseError("invalid JPEG Huffman table")
    return [], [], huff


def read_huffman_value(
    reader: JpegBitReader,
    table: list[dict[int, int]] | tuple[list[int], list[int], list[dict[int, int]]],
) -> int:
    huff = table[2] if isinstance(table, tuple) else table
    code = 0
    for current_table in huff:
        code = (code << 1) | reader.get_bit()
        value = current_table.get(code)
        if value is not None:
            return value
    raise PdfUnsupportedError("Invalid JPEG Huffman code (exceeded 16 bits)")


def extend_sign(value: int, bits: int) -> int:
    vt = 1 << (bits - 1)
    return value if value >= vt else value + (-(1 << bits) + 1)


class JPEGDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.quant_tables: dict[int, list[int]] = {}
        self.huffman_tables: dict[
            tuple[int, int], tuple[list[int], list[int], list[dict[int, int]]]
        ] = {}
        self.components: list[dict] = []
        self.scans: list[dict] = []
        self.width = 0
        self.height = 0
        self.max_h = 0
        self.max_v = 0
        self.mcu_width = 0
        self.mcu_height = 0
        self.scan_data: bytes = b""
        self.restart_interval = 0
        self.mcu_counter = 0
        self.progressive = False
        self.idct_temp = [0] * 64
        self.block = [0] * 64

    def idct_2d(self, block: list[int]) -> list[int]:
        cos_table = IDCT_COS
        scale = IDCT_SCALE
        out = [0] * 64
        for y in range(8):
            cy = cos_table[y]
            for x in range(8):
                cx = cos_table[x]
                total = 0.0
                for v in range(8):
                    sv = scale[v]
                    row = v * 8
                    cyv = cy[v]
                    for u in range(8):
                        total += scale[u] * sv * block[row + u] * cx[u] * cyv
                value = int(total * 0.25 + (0.5 if total >= 0.0 else -0.5))
                if value < -128:
                    out[y * 8 + x] = -128
                elif value > 127:
                    out[y * 8 + x] = 127
                else:
                    out[y * 8 + x] = value
        block[:] = out
        return block

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise PdfUnsupportedError("Unexpected end of JPEG data")
        val = self.data[self.pos : self.pos + n]
        self.pos += n
        return val

    def read_marker(self) -> int:
        while True:
            byte = self.read(1)[0]
            if byte == 0xFF:
                while True:
                    byte = self.read(1)[0]
                    if byte != 0xFF:
                        break
                return 0xFF00 | byte

    def parse(self) -> None:
        if self.read(2) != b"\xff\xd8":
            raise PdfUnsupportedError("Missing SOI marker")
        while True:
            marker = self.read_marker()
            if marker == 0xFFC0:
                self.parse_sof0()
            elif marker == 0xFFC2:
                self.progressive = True
                self.parse_sof0()
            elif marker == 0xFFC4:
                self.parse_dht()
            elif marker == 0xFFDB:
                self.parse_dqt()
            elif marker == 0xFFDD:
                self.parse_dri()
            elif marker == 0xFFDA:
                self.parse_sos()
                if not self.progressive:
                    break
            elif marker == 0xFFD9:
                if not self.progressive:
                    raise PdfUnsupportedError("Unexpected EOI before SOS")
                else:
                    break
            else:
                segment_len = struct.unpack(">H", self.read(2))[0]
                self.read(segment_len - 2)

    def parse_dqt(self) -> None:
        length = struct.unpack(">H", self.read(2))[0]
        if length < 2:
            raise PdfUnsupportedError("invalid DQT segment length")
        remaining = length - 2
        while remaining > 0:
            if remaining < 65:
                raise PdfUnsupportedError("truncated DQT segment")
            info = self.read(1)[0]
            precision = info >> 4
            tbl_id = info & 0x0F
            if precision != 0:
                raise PdfUnsupportedError("Only 8bit quant tables supported")
            qtable = [0] * 64
            for i in range(64):
                qtable[ZIGZAG[i]] = self.read(1)[0]
            self.quant_tables[tbl_id] = qtable
            remaining -= 1 + 64

    def parse_dht(self) -> None:
        length = struct.unpack(">H", self.read(2))[0]
        if length < 2:
            raise PdfUnsupportedError("invalid DHT segment length")
        remaining = length - 2
        while remaining > 0:
            if remaining < 17:
                raise PdfUnsupportedError("truncated DHT segment")
            info = self.read(1)[0]
            tbl_class = info >> 4
            tbl_id = info & 0x0F
            num_codes = self.read(16)
            total_symbols = sum(num_codes)
            if remaining < 17 + total_symbols:
                raise PdfUnsupportedError("truncated DHT segment")
            symbols = self.read(total_symbols)
            huff = build_huffman_table(num_codes, symbols)
            self.huffman_tables[(tbl_class, tbl_id)] = huff
            remaining -= 1 + 16 + total_symbols

    def parse_sof0(self) -> None:
        length = struct.unpack(">H", self.read(2))[0]
        if length < 8:
            raise PdfUnsupportedError("invalid SOF segment length")
        self.pos += 1
        self.height = struct.unpack(">H", self.read(2))[0]
        self.width = struct.unpack(">H", self.read(2))[0]
        n_components = self.read(1)[0]
        if n_components == 0:
            raise PdfUnsupportedError("invalid JPEG component count")
        self.components = []
        self.max_h = 0
        self.max_v = 0
        for _ in range(n_components):
            comp_id = self.read(1)[0]
            sampling = self.read(1)[0]
            h = sampling >> 4
            v = sampling & 0x0F
            qt_id = self.read(1)[0]
            self.components.append({"id": comp_id, "h": h, "v": v, "qt": qt_id, "prev_dc": 0})
            self.max_h = max(self.max_h, h)
            self.max_v = max(self.max_v, v)
        self.mcu_width = self.max_h * 8
        self.mcu_height = self.max_v * 8
        remaining = length - 2 - 1 - 2 - 2 - 1 - n_components * 3
        if remaining > 0:
            self.read(remaining)

    def parse_dri(self) -> None:
        length = struct.unpack(">H", self.read(2))[0]
        if length < 4:
            raise PdfUnsupportedError("invalid DRI segment length")
        self.restart_interval = struct.unpack(">H", self.read(2))[0]
        remaining = length - 2 - 2
        if remaining:
            self.read(remaining)

    def parse_sos(self) -> None:
        length = struct.unpack(">H", self.read(2))[0]
        if length < 6:
            raise PdfUnsupportedError("invalid SOS segment length")
        n_components = self.read(1)[0]
        if n_components == 0:
            raise PdfUnsupportedError("invalid JPEG scan component count")
        scan_components: list[dict] = []
        for _ in range(n_components):
            comp_id = self.read(1)[0]
            tbl_sel = self.read(1)[0]
            dc_tbl = tbl_sel >> 4
            ac_tbl = tbl_sel & 0x0F
            matched = False
            for comp in self.components:
                if comp["id"] == comp_id:
                    matched = True
                    comp["dc_tbl"] = dc_tbl
                    comp["ac_tbl"] = ac_tbl
                    scan_components.append(
                        {
                            "comp": comp,
                            "dc_tbl": dc_tbl,
                            "ac_tbl": ac_tbl,
                            "dc_tbl_ref": self.huffman_tables.get((0, dc_tbl)),
                            "ac_tbl_ref": self.huffman_tables.get((1, ac_tbl)),
                        }
                    )
                    break
            if not matched:
                raise PdfUnsupportedError("invalid JPEG scan component")
        ss = self.read(1)[0]
        se = self.read(1)[0]
        ah_al = self.read(1)[0]
        ah = ah_al >> 4
        al = ah_al & 0x0F
        remaining = length - 2 - 1 - n_components * 2 - 3
        if remaining > 0:
            self.read(remaining)
        scan_start = self.pos
        if not self.progressive:
            self.locate_eoi()
            self.scan_data = self.data[scan_start : self.pos]
            self.pos += 2
        else:
            self.skip_progressive_scan()
            self.scans.append(
                {
                    "components": scan_components,
                    "Ss": ss,
                    "Se": se,
                    "Ah": ah,
                    "Al": al,
                    "data": self.data[scan_start : self.pos],
                }
            )

    def locate_eoi(self) -> None:
        data = self.data
        pos = self.pos
        end = len(data) - 1
        while pos < end:
            ff = data.find(b"\xff", pos)
            if ff < 0 or ff >= end:
                break
            if data[ff + 1] == 0xD9:
                while ff > self.pos and data[ff - 1] == 0xFF:
                    ff -= 1
                self.pos = ff
                return
            pos = ff + 1
        raise PdfUnsupportedError("EOI not found")

    def skip_progressive_scan(self) -> None:
        data = self.data
        pos = self.pos
        length = len(data)
        while True:
            pos = data.find(b"\xff", pos)
            if pos < 0:
                raise PdfUnsupportedError(
                    "Unexpected end of JPEG data while skipping progressive scan"
                )
            if pos + 1 >= length:
                raise PdfUnsupportedError("Unexpected end after marker prefix")
            next_byte = data[pos + 1]
            if next_byte == 0x00 or 0xD0 <= next_byte <= 0xD7:
                pos += 2
                continue
            if next_byte == 0xFF:
                pos += 1
                continue
            while pos > self.pos and data[pos - 1] == 0xFF:
                pos -= 1
            self.pos = pos
            return

    def decode_mcu_block(
        self,
        reader: JpegBitReader,
        comp: dict,
        dc_tbl: tuple[list[int], list[int], list[dict[int, int]]],
        ac_tbl: tuple[list[int], list[int], list[dict[int, int]]],
        block: list[int],
    ) -> bool:
        zigzag = ZIGZAG
        get_bits = reader.get_bits
        extend = extend_sign
        category = read_huffman_value(reader, dc_tbl)
        diff = extend(get_bits(category), category) if category else 0
        dc = comp["prev_dc"] + diff
        comp["prev_dc"] = dc
        block[:] = ZERO_BLOCK_64
        block[0] = dc
        i = 1
        zero_ac = True
        while i < 64:
            rs = read_huffman_value(reader, ac_tbl)
            if rs == 0:
                break
            zero_ac = False
            if rs == 0xF0:
                i += 16
                continue
            i += rs >> 4
            size = rs & 0x0F
            if size:
                coeff = extend(get_bits(size), size)
                if i < 64:
                    block[zigzag[i]] = coeff
            i += 1
        return zero_ac

    def decode(self) -> bytes:
        if self.progressive:
            return self.decode_progressive()
        reader = JpegBitReader(self.scan_data)
        mcu_cols = (self.width + self.mcu_width - 1) // self.mcu_width
        mcu_rows = (self.height + self.mcu_height - 1) // self.mcu_height
        comp_buf: dict[int, list[int]] = {}
        comp_w: dict[int, int] = {}
        comp_h: dict[int, int] = {}
        quant_tables = self.quant_tables
        for comp in self.components:
            cw = (self.width * comp["h"] + self.max_h - 1) // self.max_h
            ch = (self.height * comp["v"] + self.max_v - 1) // self.max_v
            comp_w[comp["id"]] = cw
            comp_h[comp["id"]] = ch
            comp_buf[comp["id"]] = [0] * (cw * ch)
        component_meta = [
            (
                comp,
                comp["v"],
                comp["h"],
                comp.get("dc_tbl_ref") or self.huffman_tables[(0, comp["dc_tbl"])],
                comp.get("ac_tbl_ref") or self.huffman_tables[(1, comp["ac_tbl"])],
                quant_tables[comp["qt"]],
                comp_w[comp["id"]],
                comp_h[comp["id"]],
                comp_buf[comp["id"]],
            )
            for comp in self.components
        ]
        decode_mcu = self.decode_mcu_block
        idct = self.idct_2d
        handle_restart = self.handle_restart
        block = self.block
        for row in range(mcu_rows):
            for col in range(mcu_cols):
                for comp, cv, ch, dc_tbl, ac_tbl, qt, cw, ch_, cbuf in component_meta:
                    for v in range(cv):
                        for h in range(ch):
                            zero_ac = decode_mcu(reader, comp, dc_tbl, ac_tbl, block)
                            x0 = (col * ch + h) * 8
                            y0 = (row * cv + v) * 8
                            block_w = cw - x0
                            if block_w > 8:
                                block_w = 8
                            block_h = ch_ - y0
                            if block_h > 8:
                                block_h = 8
                            row_base = y0 * cw + x0
                            if zero_ac:
                                dc_val = block[0] * qt[0] >> 3
                                if dc_val < -128:
                                    dc_val = -128
                                elif dc_val > 127:
                                    dc_val = 127
                                dc_pixel = dc_val + 128
                                if block_w == 8 and block_h == 8:
                                    fill = [dc_pixel] * 8
                                    for _ in range(8):
                                        cbuf[row_base : row_base + 8] = fill
                                        row_base += cw
                                else:
                                    for _ in range(block_h):
                                        for xx in range(block_w):
                                            cbuf[row_base + xx] = dc_pixel
                                        row_base += cw
                            else:
                                block[:] = [block[i] * qt[i] for i in range(64)]
                                pixels = idct(block)
                                if block_w == 8 and block_h == 8:
                                    for yy in range(8):
                                        start = yy * 8
                                        cbuf[row_base : row_base + 8] = [
                                            pixels[start + xx] + 128 for xx in range(8)
                                        ]
                                        row_base += cw
                                else:
                                    for yy in range(block_h):
                                        start = yy * 8
                                        for xx in range(block_w):
                                            cbuf[row_base + xx] = pixels[start + xx] + 128
                                        row_base += cw
                self.mcu_counter += 1
                if self.restart_interval:
                    handle_restart(reader)
        return self.compose_rgb(comp_buf, comp_w)

    def compose_rgb(self, comp_buf: dict[int, list[int]], comp_w: dict[int, int]) -> bytes:
        w, h = self.width, self.height
        rgb = bytearray(w * h * 3)
        components = self.components
        y_id = components[0]["id"]
        if len(components) == 1:
            y_buf = comp_buf[y_id]
            y_stride = comp_w[y_id]
            off = 0
            for y in range(h):
                y_buf_row = y * y_stride
                for x in range(w):
                    Y = y_buf[y_buf_row + x]
                    rgb[off] = Y
                    rgb[off + 1] = Y
                    rgb[off + 2] = Y
                    off += 3
        else:

            def row_slice(buf: list[int], stride: int, idx: int) -> list[int]:
                if idx < 0:
                    idx = 0
                elif idx >= len(buf) // stride:
                    idx = (len(buf) // stride) - 1
                start = idx * stride
                return buf[start : start + stride]

            def upsample_full(comp: dict, buf: list[int], stride: int) -> list[int]:
                if comp["h"] == self.max_h and comp["v"] == self.max_v:
                    return buf
                h_expand = self.max_h // comp["h"]
                v_expand = self.max_v // comp["v"]
                src_h = len(buf) // stride
                if h_expand == 2 and v_expand == 2:
                    outw = stride * 2
                    outh = src_h * 2
                    out = [0] * (outw * outh)
                    for inrow in range(src_h):
                        for v in (0, 1):
                            row0 = row_slice(buf, stride, inrow)
                            row1 = row_slice(buf, stride, inrow - 1 if v == 0 else inrow + 1)
                            dst = (inrow * 2 + v) * outw
                            thiscolsum = row0[0] * 3 + row1[0]
                            if stride > 1:
                                nextcolsum = row0[1] * 3 + row1[1]
                            else:
                                nextcolsum = thiscolsum
                            out[dst] = (thiscolsum * 4 + 8) >> 4
                            out[dst + 1] = (thiscolsum * 3 + nextcolsum + 7) >> 4
                            lastcolsum = thiscolsum
                            thiscolsum = nextcolsum
                            out_idx = 2
                            for col in range(1, stride - 1):
                                nextcolsum = row0[col + 1] * 3 + row1[col + 1]
                                out[dst + out_idx] = (thiscolsum * 3 + lastcolsum + 8) >> 4
                                out[dst + out_idx + 1] = (thiscolsum * 3 + nextcolsum + 7) >> 4
                                lastcolsum = thiscolsum
                                thiscolsum = nextcolsum
                                out_idx += 2
                            if stride > 1:
                                out[dst + out_idx] = (thiscolsum * 3 + lastcolsum + 8) >> 4
                                out[dst + out_idx + 1] = (thiscolsum * 4 + 7) >> 4
                    return out
                if h_expand == 2 and v_expand == 1:
                    outw = stride * 2
                    out = [0] * (outw * src_h)
                    for inrow in range(src_h):
                        row = row_slice(buf, stride, inrow)
                        dst = inrow * outw
                        if stride == 1:
                            out[dst] = row[0]
                            out[dst + 1] = row[0]
                            continue
                        invalue = row[0]
                        out[dst] = invalue
                        out[dst + 1] = (invalue * 3 + row[1] + 2) >> 2
                        out_idx = 2
                        for col in range(1, stride - 1):
                            invalue = row[col] * 3
                            out[dst + out_idx] = (invalue + row[col - 1] + 1) >> 2
                            out[dst + out_idx + 1] = (invalue + row[col + 1] + 2) >> 2
                            out_idx += 2
                        invalue = row[stride - 1]
                        out[dst + out_idx] = (invalue * 3 + row[stride - 2] + 1) >> 2
                        out[dst + out_idx + 1] = invalue
                    return out
                if h_expand == 1 and v_expand == 2:
                    outw = stride
                    outh = src_h * 2
                    out = [0] * (outw * outh)
                    for inrow in range(src_h):
                        for v in (0, 1):
                            row0 = row_slice(buf, stride, inrow)
                            row1 = row_slice(buf, stride, inrow - 1 if v == 0 else inrow + 1)
                            bias = 1 if v == 0 else 2
                            dst = (inrow * 2 + v) * outw
                            for col in range(stride):
                                thiscolsum = row0[col] * 3 + row1[col]
                                out[dst + col] = (thiscolsum + bias) >> 2
                    return out
                outw = stride * h_expand
                outh = src_h * v_expand
                out = [0] * (outw * outh)
                for inrow in range(src_h):
                    src_row = row_slice(buf, stride, inrow)
                    for dy in range(v_expand):
                        dst = (inrow * v_expand + dy) * outw
                        for col in range(stride):
                            value = src_row[col]
                            base = col * h_expand
                            for dx in range(h_expand):
                                out[dst + base + dx] = value
                return out

            y_buf = comp_buf[y_id]
            cb_id = components[1]["id"]
            cr_id = components[2]["id"]
            y_buf = upsample_full(components[0], y_buf, comp_w[y_id])
            cb_buf = upsample_full(components[1], comp_buf[cb_id], comp_w[cb_id])
            cr_buf = upsample_full(components[2], comp_buf[cr_id], comp_w[cr_id])
            cb_to_b = CB_TO_B
            cb_to_g = CB_TO_G
            cr_to_r = CR_TO_R
            cr_to_g = CR_TO_G
            for y in range(h):
                y_row = y * w
                off = y_row * 3
                for x in range(w):
                    Y = y_buf[y_row + x]
                    Cb = cb_buf[y_row + x]
                    Cr = cr_buf[y_row + x]
                    r = Y + cr_to_r[Cr]
                    g = Y + ((cb_to_g[Cb] + cr_to_g[Cr]) >> SCALEBITS)
                    b = Y + cb_to_b[Cb]
                    if r < 0:
                        r = 0
                    elif r > 255:
                        r = 255
                    if g < 0:
                        g = 0
                    elif g > 255:
                        g = 255
                    if b < 0:
                        b = 0
                    elif b > 255:
                        b = 255
                    rgb[off] = r
                    rgb[off + 1] = g
                    rgb[off + 2] = b
                    off += 3
        return bytes(rgb)

    def decode_progressive(self) -> bytes:
        mcu_cols = (self.width + self.mcu_width - 1) // self.mcu_width
        mcu_rows = (self.height + self.mcu_height - 1) // self.mcu_height
        comp_w: dict[int, int] = {}
        comp_h: dict[int, int] = {}
        blocks_w: dict[int, int] = {}
        blocks_h: dict[int, int] = {}
        coeff_buf: dict[int, list[int]] = {}
        quant_tables = self.quant_tables
        for comp in self.components:
            cw = (self.width * comp["h"] + self.max_h - 1) // self.max_h
            ch = (self.height * comp["v"] + self.max_v - 1) // self.max_v
            comp_id = comp["id"]
            comp_w[comp_id] = cw
            comp_h[comp_id] = ch
            blocks_w[comp_id] = mcu_cols * comp["h"]
            blocks_h[comp_id] = mcu_rows * comp["v"]
            coeff_buf[comp_id] = [0] * (blocks_w[comp_id] * blocks_h[comp_id] * 64)

        for scan in self.scans:
            ss = scan["Ss"]
            se = scan["Se"]
            ah = scan["Ah"]
            al = scan["Al"]
            scan_components = scan["components"]
            if ah != 0 and al != ah - 1:
                raise PdfUnsupportedError("Invalid progressive successive approximation")
            for item in scan_components:
                item["comp"]["prev_dc"] = 0

            def get_huff_table(
                tbl: tuple[list[int], list[int], list[dict[int, int]]] | None,
            ) -> list[dict[int, int]]:
                return tbl[2] if tbl else []

            scan_meta = [
                (
                    item["comp"],
                    get_huff_table(item.get("dc_tbl_ref"))
                    or get_huff_table(self.huffman_tables[(0, item["dc_tbl"])]),
                    get_huff_table(item.get("ac_tbl_ref"))
                    or get_huff_table(self.huffman_tables[(1, item["ac_tbl"])]),
                    blocks_w[item["comp"]["id"]],
                    blocks_h[item["comp"]["id"]],
                )
                for item in scan_components
            ]
            reader = JpegBitReader(scan["data"])
            self.mcu_counter = 0
            if ss == 0:
                if se != 0:
                    raise PdfUnsupportedError("Invalid progressive DC spectral selection")
                if ah == 0:
                    self.decode_progressive_dc_first_scan(
                        reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
                    )
                else:
                    self.decode_progressive_dc_refine_scan(
                        reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
                    )
            else:
                if len(scan_meta) != 1:
                    raise PdfUnsupportedError("Progressive AC scan must contain one component")
                if se >= 64 or ss > se:
                    raise PdfUnsupportedError("Invalid progressive AC spectral selection")
                if ah == 0:
                    self.decode_progressive_ac_first_scan(
                        reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, ss, se, al
                    )
                else:
                    self.decode_progressive_ac_refine_scan(
                        reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, ss, se, al
                    )

        comp_pixels: dict[int, list[int]] = {}
        idct = self.idct_2d
        block = self.block
        for comp in self.components:
            comp_id = comp["id"]
            pixels = [0] * (comp_w[comp_id] * comp_h[comp_id])
            coeffs = coeff_buf[comp_id]
            qt = quant_tables[comp["qt"]]
            bw = blocks_w[comp_id]
            bh = blocks_h[comp_id]
            row_stride = comp_w[comp_id]
            for by in range(bh):
                row_base = by * 8 * row_stride
                for bx in range(bw):
                    base = ((by * bw) + bx) * 64
                    for i in range(64):
                        block[i] = coeffs[base + i] * qt[i]
                    pixels_block = idct(block)
                    pixel_base = row_base + bx * 8
                    block_w = row_stride - bx * 8
                    if block_w > 8:
                        block_w = 8
                    block_h = comp_h[comp_id] - by * 8
                    if block_h > 8:
                        block_h = 8
                    for yy in range(8):
                        src = yy * 8
                        dst = pixel_base + yy * row_stride
                        if yy >= block_h:
                            break
                        for xx in range(block_w):
                            pixels[dst + xx] = pixels_block[src + xx] + 128
            comp_pixels[comp_id] = pixels

        return self.compose_rgb(comp_pixels, comp_w)

    def decode_progressive_dc_first_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[tuple[dict, list[dict[int, int]], list[dict[int, int]], int, int]],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        al: int,
    ) -> None:
        get_bits = reader.get_bits
        extend = extend_sign
        for row in range(mcu_rows):
            for col in range(mcu_cols):
                for comp, dc_tbl, ac_tbl, blocks_w, blocks_h in scan_meta:
                    comp_prev_dc = comp["prev_dc"]
                    comp_id = comp["id"]
                    coeffs = coeff_buf[comp_id]
                    comp_h = comp["h"]
                    comp_v = comp["v"]
                    for v in range(comp_v):
                        block_y = row * comp_v + v
                        for h in range(comp_h):
                            category = read_huffman_value(reader, dc_tbl)
                            diff = extend(get_bits(category), category) if category else 0
                            comp_prev_dc += diff << al
                            block_x = col * comp_h + h
                            base = ((block_y * blocks_w) + block_x) * 64
                            coeffs[base] = comp_prev_dc
                    comp["prev_dc"] = comp_prev_dc
                self.mcu_counter += 1
                if self.restart_interval:
                    self.handle_restart(reader)

    def decode_progressive_dc_refine_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[tuple[dict, list[dict[int, int]], list[dict[int, int]], int, int]],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        al: int,
    ) -> None:
        p1 = 1 << al
        get_bit = reader.get_bit
        for row in range(mcu_rows):
            for col in range(mcu_cols):
                for comp, dc_tbl, ac_tbl, blocks_w, blocks_h in scan_meta:
                    comp_id = comp["id"]
                    coeffs = coeff_buf[comp_id]
                    comp_h = comp["h"]
                    comp_v = comp["v"]
                    for v in range(comp_v):
                        block_y = row * comp_v + v
                        for h in range(comp_h):
                            if get_bit():
                                block_x = col * comp_h + h
                                base = ((block_y * blocks_w) + block_x) * 64
                                coeffs[base] |= p1
                self.mcu_counter += 1
                if self.restart_interval:
                    self.handle_restart(reader)

    def decode_progressive_ac_first_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[tuple[dict, list[dict[int, int]], list[dict[int, int]], int, int]],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        ss: int,
        se: int,
        al: int,
    ) -> None:
        get_bits = reader.get_bits
        extend = extend_sign
        zigzag = ZIGZAG
        comp, dc_tbl, ac_tbl, blocks_w, blocks_h = scan_meta[0]
        comp_id = comp["id"]
        coeffs = coeff_buf[comp_id]
        eobrun = 0
        for block_y in range(blocks_h):
            row_base = block_y * blocks_w * 64
            for block_x in range(blocks_w):
                base = row_base + block_x * 64
                saved = coeffs[base : base + 64]
                try:
                    if eobrun > 0:
                        eobrun -= 1
                        continue
                    k = ss
                    while k <= se:
                        rs = read_huffman_value(reader, ac_tbl)
                        r = rs >> 4
                        s = rs & 0x0F
                        if s:
                            k += r
                            value = extend(get_bits(s), s)
                            coeffs[base + zigzag[k]] = value << al
                            k += 1
                        else:
                            if r == 15:
                                k += 16
                                continue
                            eobrun = 1 << r
                            if r:
                                eobrun += reader.get_bits(r)
                            eobrun -= 1
                            break
                    self.mcu_counter += 1
                    if self.restart_interval and self.handle_restart(reader):
                        eobrun = 0
                except PdfUnsupportedError:
                    coeffs[base : base + 64] = saved
                    return

    def decode_progressive_ac_refine_scan(
        self,
        reader: JpegBitReader,
        scan_meta: list[tuple[dict, list[dict[int, int]], list[dict[int, int]], int, int]],
        coeff_buf: dict[int, list[int]],
        mcu_cols: int,
        mcu_rows: int,
        ss: int,
        se: int,
        al: int,
    ) -> None:
        p1 = 1 << al
        m1 = -p1
        zigzag = ZIGZAG
        get_bit = reader.get_bit
        comp, dc_tbl, ac_tbl, blocks_w, blocks_h = scan_meta[0]
        comp_id = comp["id"]
        coeffs = coeff_buf[comp_id]
        eobrun = 0
        for block_y in range(blocks_h):
            row_base = block_y * blocks_w * 64
            for block_x in range(blocks_w):
                base = row_base + block_x * 64
                saved = coeffs[base : base + 64]
                try:
                    k = ss
                    if eobrun == 0:
                        while k <= se:
                            rs = read_huffman_value(reader, ac_tbl)
                            r = rs >> 4
                            s = rs & 0x0F
                            if s:
                                if s != 1:
                                    s = 1
                                newcoef = p1 if get_bit() else m1
                            else:
                                if r != 15:
                                    eobrun = 1 << r
                                    if r:
                                        eobrun += reader.get_bits(r)
                                    break
                            while True:
                                idx = base + zigzag[k]
                                coef = coeffs[idx]
                                if coef != 0:
                                    if get_bit() and (coef & p1) == 0:
                                        coeffs[idx] = coef + p1 if coef >= 0 else coef + m1
                                    k += 1
                                else:
                                    r -= 1
                                    if r < 0:
                                        break
                                    k += 1
                                if k > se:
                                    break
                            if s:
                                coeffs[base + zigzag[k]] = newcoef
                                k += 1
                        if eobrun > 0:
                            while k <= se:
                                idx = base + zigzag[k]
                                coef = coeffs[idx]
                                if coef != 0 and get_bit() and (coef & p1) == 0:
                                    coeffs[idx] = coef + p1 if coef >= 0 else coef + m1
                                k += 1
                            eobrun -= 1
                    else:
                        while k <= se:
                            idx = base + zigzag[k]
                            coef = coeffs[idx]
                            if coef != 0 and get_bit() and (coef & p1) == 0:
                                coeffs[idx] = coef + p1 if coef >= 0 else coef + m1
                            k += 1
                        eobrun -= 1
                    self.mcu_counter += 1
                    if self.restart_interval and self.handle_restart(reader):
                        eobrun = 0
                except PdfUnsupportedError:
                    coeffs[base : base + 64] = saved
                    return

    def handle_restart(self, reader: JpegBitReader) -> bool:
        if self.restart_interval == 0:
            return False
        if self.mcu_counter % self.restart_interval != 0:
            return False
        if reader.bits_left != 0:
            reader.bits_left = 0
            reader.buffer = 0
        if reader.pos + 2 > len(reader.data):
            return False
        if reader.data[reader.pos] == 0xFF and 0xD0 <= reader.data[reader.pos + 1] <= 0xD7:
            reader.pos += 2
            for comp in self.components:
                comp["prev_dc"] = 0
            return True
        return False

    @classmethod
    def from_data(cls, data: bytes) -> bytes:
        try:
            decoder = cls(data)
            decoder.parse()
            return decoder.decode()
        except PdfUnsupportedError, struct.error, OSError:
            raise PdfUnsupportedError("JPEGDecode failed")
