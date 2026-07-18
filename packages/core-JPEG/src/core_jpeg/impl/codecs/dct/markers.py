# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import struct
from typing import Any, TypeAlias

from core_jpeg.impl.codecs.dct.huffman import build_huffman_table
from core_jpeg.impl.codecs.dct.tables import ZIGZAG
from core_jpeg.impl.errors import JpegUnsupportedError

ScanComponent: TypeAlias = dict[str, Any]


def read(decoder: Any, n: int) -> bytes:
    if decoder.pos + n > len(decoder.data):
        raise JpegUnsupportedError("Unexpected end of JPEG data")
    val = decoder.data[decoder.pos : decoder.pos + n]
    decoder.pos += n
    return val


def read_marker(decoder: Any) -> int:
    while True:
        byte = read(decoder, 1)[0]
        if byte == 0xFF:
            while True:
                byte = read(decoder, 1)[0]
                if byte != 0xFF:
                    break
            return 0xFF00 | byte


def parse(decoder: Any) -> None:
    if read(decoder, 2) != b"\xff\xd8":
        raise JpegUnsupportedError("Missing SOI marker")
    while True:
        marker = read_marker(decoder)
        if marker == 0xFFC0:
            parse_sof0(decoder)
        elif marker == 0xFFC2:
            decoder.progressive = True
            parse_sof0(decoder)
        elif marker == 0xFFC4:
            parse_dht(decoder)
        elif marker == 0xFFDB:
            parse_dqt(decoder)
        elif marker == 0xFFDD:
            parse_dri(decoder)
        elif marker == 0xFFEE:
            parse_app14(decoder)
        elif marker == 0xFFDA:
            parse_sos(decoder)
            if not decoder.progressive:
                break
        elif marker == 0xFFD9:
            if not decoder.progressive:
                raise JpegUnsupportedError("Unexpected EOI before SOS")
            break
        else:
            segment_len = struct.unpack(">H", read(decoder, 2))[0]
            if segment_len < 2:
                raise JpegUnsupportedError("invalid JPEG marker segment length")
            read(decoder, segment_len - 2)


def parse_dqt(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length < 2:
        raise JpegUnsupportedError("invalid DQT segment length")
    remaining = length - 2
    while remaining > 0:
        if remaining < 65:
            raise JpegUnsupportedError("truncated DQT segment")
        info = read(decoder, 1)[0]
        precision = info >> 4
        tbl_id = info & 0x0F
        if precision != 0:
            raise JpegUnsupportedError("Only 8bit quant tables supported")
        if tbl_id > 3:
            raise JpegUnsupportedError("invalid JPEG quantization table id")
        qtable = [0] * 64
        for i in range(64):
            qtable[ZIGZAG[i]] = read(decoder, 1)[0]
        decoder.quant_tables[tbl_id] = qtable
        remaining -= 1 + 64


def parse_dht(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length < 2:
        raise JpegUnsupportedError("invalid DHT segment length")
    remaining = length - 2
    while remaining > 0:
        if remaining < 17:
            raise JpegUnsupportedError("truncated DHT segment")
        info = read(decoder, 1)[0]
        tbl_class = info >> 4
        tbl_id = info & 0x0F
        if tbl_class > 1 or tbl_id > 3:
            raise JpegUnsupportedError("invalid JPEG Huffman table id")
        num_codes = read(decoder, 16)
        total_symbols = sum(num_codes)
        if remaining < 17 + total_symbols:
            raise JpegUnsupportedError("truncated DHT segment")
        symbols = read(decoder, total_symbols)
        huff = build_huffman_table(num_codes, symbols)
        decoder.huffman_tables[(tbl_class, tbl_id)] = huff
        remaining -= 1 + 16 + total_symbols


def parse_sof0(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length < 8:
        raise JpegUnsupportedError("invalid SOF segment length")
    precision = read(decoder, 1)[0]
    if precision != 8:
        raise JpegUnsupportedError("unsupported JPEG sample precision")
    decoder.height = struct.unpack(">H", read(decoder, 2))[0]
    decoder.width = struct.unpack(">H", read(decoder, 2))[0]
    if decoder.width == 0 or decoder.height == 0:
        raise JpegUnsupportedError("invalid JPEG image dimensions")
    n_components = read(decoder, 1)[0]
    if n_components not in {1, 3, 4}:
        raise JpegUnsupportedError("invalid JPEG component count")
    expected_length = 2 + 1 + 2 + 2 + 1 + n_components * 3
    if length != expected_length:
        raise JpegUnsupportedError("invalid SOF segment length")
    decoder.components = []
    decoder.max_h = 0
    decoder.max_v = 0
    seen_component_ids: set[int] = set()
    for ignored in range(n_components):
        comp_id = read(decoder, 1)[0]
        if comp_id in seen_component_ids:
            raise JpegUnsupportedError("duplicate JPEG component id")
        seen_component_ids.add(comp_id)
        sampling = read(decoder, 1)[0]
        h = sampling >> 4
        v = sampling & 0x0F
        qt_id = read(decoder, 1)[0]
        if h == 0 or v == 0:
            raise JpegUnsupportedError("invalid JPEG sampling factors")
        if qt_id > 3:
            raise JpegUnsupportedError("invalid JPEG quantization table id")
        decoder.components.append({"id": comp_id, "h": h, "v": v, "qt": qt_id, "prev_dc": 0})
        decoder.max_h = max(decoder.max_h, h)
        decoder.max_v = max(decoder.max_v, v)
    decoder.mcu_width = decoder.max_h * 8
    decoder.mcu_height = decoder.max_v * 8


def parse_dri(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length != 4:
        raise JpegUnsupportedError("invalid DRI segment length")
    decoder.restart_interval = struct.unpack(">H", read(decoder, 2))[0]


def parse_app14(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length < 2:
        raise JpegUnsupportedError("invalid APP14 segment length")
    payload = read(decoder, length - 2)
    if len(payload) >= 12 and payload[:5] == b"Adobe":
        decoder.adobe_transform = payload[11]


def parse_sos(decoder: Any) -> None:
    length = struct.unpack(">H", read(decoder, 2))[0]
    if length < 6:
        raise JpegUnsupportedError("invalid SOS segment length")
    n_components = read(decoder, 1)[0]
    if n_components == 0 or n_components > len(decoder.components):
        raise JpegUnsupportedError("invalid JPEG scan component count")
    expected_length = 2 + 1 + n_components * 2 + 3
    if length != expected_length:
        raise JpegUnsupportedError("invalid SOS segment length")
    scan_components: list[ScanComponent] = []
    seen_components: set[int] = set()
    for ignored in range(n_components):
        comp_id = read(decoder, 1)[0]
        if comp_id in seen_components:
            raise JpegUnsupportedError("duplicate JPEG scan component")
        seen_components.add(comp_id)
        tbl_sel = read(decoder, 1)[0]
        dc_tbl = tbl_sel >> 4
        ac_tbl = tbl_sel & 0x0F
        if dc_tbl > 3 or ac_tbl > 3:
            raise JpegUnsupportedError("invalid JPEG scan table id")
        matched = False
        for comp in decoder.components:
            if comp["id"] == comp_id:
                matched = True
                comp["dc_tbl"] = dc_tbl
                comp["ac_tbl"] = ac_tbl
                scan_components.append(
                    {
                        "comp": comp,
                        "dc_tbl": dc_tbl,
                        "ac_tbl": ac_tbl,
                        "dc_tbl_ref": decoder.huffman_tables.get((0, dc_tbl)),
                        "ac_tbl_ref": decoder.huffman_tables.get((1, ac_tbl)),
                    }
                )
                break
        if not matched:
            raise JpegUnsupportedError("invalid JPEG scan component")
    ss = read(decoder, 1)[0]
    se = read(decoder, 1)[0]
    ah_al = read(decoder, 1)[0]
    ah = ah_al >> 4
    al = ah_al & 0x0F
    if not decoder.progressive and (ss != 0 or se != 63 or ah != 0 or al != 0):
        raise JpegUnsupportedError("invalid baseline JPEG spectral selection")
    scan_start = decoder.pos
    if not decoder.progressive:
        locate_eoi(decoder)
        decoder.scan_data = decoder.data[scan_start : decoder.pos]
        decoder.pos += 2
    else:
        skip_progressive_scan(decoder)
        decoder.scans.append(
            {
                "components": scan_components,
                "Ss": ss,
                "Se": se,
                "Ah": ah,
                "Al": al,
                "data": decoder.data[scan_start : decoder.pos],
            }
        )


def locate_eoi(decoder: Any) -> None:
    data = decoder.data
    pos = decoder.pos
    end = len(data) - 1
    while pos < end:
        ff = data.find(b"\xff", pos)
        if ff < 0 or ff >= end:
            break
        if data[ff + 1] == 0xD9:
            while ff > decoder.pos and data[ff - 1] == 0xFF:
                ff -= 1
            decoder.pos = ff
            return
        pos = ff + 1
    raise JpegUnsupportedError("EOI not found")


def skip_progressive_scan(decoder: Any) -> None:
    data = decoder.data
    pos = decoder.pos
    length = len(data)
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0:
            raise JpegUnsupportedError(
                "Unexpected end of JPEG data while skipping progressive scan"
            )
        if pos + 1 >= length:
            raise JpegUnsupportedError("Unexpected end after marker prefix")
        next_byte = data[pos + 1]
        if next_byte == 0x00 or 0xD0 <= next_byte <= 0xD7:
            pos += 2
            continue
        if next_byte == 0xFF:
            pos += 1
            continue
        while pos > decoder.pos and data[pos - 1] == 0xFF:
            pos -= 1
        decoder.pos = pos
        return
