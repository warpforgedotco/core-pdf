# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, TypeAlias

from core_jpeg.impl.codecs.dct.bitstream import JpegBitReader
from core_jpeg.impl.codecs.dct.color import (
    cmyk_to_rgb_channels,
    inverted_cmyk_to_rgb_channels,
    ycbcr_to_rgb_channels,
    ycck_to_rgb_channels,
)
from core_jpeg.impl.codecs.dct.huffman import (
    HuffmanTable,
    extend_sign,
    read_huffman_value,
)
from core_jpeg.impl.codecs.dct.tables import ZERO_BLOCK_64, ZIGZAG
from core_jpeg.impl.errors import JpegUnsupportedError

JpegComponent: TypeAlias = dict[str, Any]
ScanMeta: TypeAlias = tuple[JpegComponent, HuffmanTable, HuffmanTable, int, int]


def decode_mcu_block(
    reader: JpegBitReader,
    comp: JpegComponent,
    dc_tbl: HuffmanTable,
    ac_tbl: HuffmanTable,
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


def decode(decoder: Any) -> bytes:
    if decoder.progressive:
        return decode_progressive(decoder)
    reader = JpegBitReader(decoder.scan_data)
    mcu_cols = (decoder.width + decoder.mcu_width - 1) // decoder.mcu_width
    mcu_rows = (decoder.height + decoder.mcu_height - 1) // decoder.mcu_height
    comp_buf: dict[int, list[int]] = {}
    comp_w: dict[int, int] = {}
    comp_h: dict[int, int] = {}
    quant_tables = decoder.quant_tables
    for comp in decoder.components:
        cw = (decoder.width * comp["h"] + decoder.max_h - 1) // decoder.max_h
        ch = (decoder.height * comp["v"] + decoder.max_v - 1) // decoder.max_v
        comp_w[comp["id"]] = cw
        comp_h[comp["id"]] = ch
        comp_buf[comp["id"]] = [0] * (cw * ch)
    component_meta = [
        (
            comp,
            comp["v"],
            comp["h"],
            comp.get("dc_tbl_ref") or decoder.huffman_tables[(0, comp["dc_tbl"])],
            comp.get("ac_tbl_ref") or decoder.huffman_tables[(1, comp["ac_tbl"])],
            quant_tables[comp["qt"]],
            comp_w[comp["id"]],
            comp_h[comp["id"]],
            comp_buf[comp["id"]],
        )
        for comp in decoder.components
    ]
    idct = decoder.idct_2d
    block = decoder.block
    for row in range(mcu_rows):
        for col in range(mcu_cols):
            for comp, cv, ch, dc_tbl, ac_tbl, qt, cw, ch_, cbuf in component_meta:
                for v in range(cv):
                    for h in range(ch):
                        zero_ac = decode_mcu_block(reader, comp, dc_tbl, ac_tbl, block)
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
                                for ignored in range(8):
                                    cbuf[row_base : row_base + 8] = fill
                                    row_base += cw
                            else:
                                for ignored in range(block_h):
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
            decoder.mcu_counter += 1
            if decoder.restart_interval:
                handle_restart(decoder, reader)
    return compose_rgb(decoder, comp_buf, comp_w)


def compose_rgb(decoder: Any, comp_buf: dict[int, list[int]], comp_w: dict[int, int]) -> bytes:
    w, h = decoder.width, decoder.height
    rgb = bytearray(w * h * 3)
    components = decoder.components
    y_id = components[0]["id"]
    if len(components) == 1:
        y_buf = comp_buf[y_id]
        y_stride = comp_w[y_id]
        off = 0
        for y in range(h):
            y_buf_row = y * y_stride
            for x in range(w):
                y_sample = y_buf[y_buf_row + x]
                rgb[off] = y_sample
                rgb[off + 1] = y_sample
                rgb[off + 2] = y_sample
                off += 3
    else:

        def row_slice(buf: list[int], stride: int, idx: int) -> list[int]:
            if idx < 0:
                idx = 0
            elif idx >= len(buf) // stride:
                idx = (len(buf) // stride) - 1
            start = idx * stride
            return buf[start : start + stride]

        def upsample_full(comp: JpegComponent, buf: list[int], stride: int) -> list[int]:
            if comp["h"] == decoder.max_h and comp["v"] == decoder.max_v:
                return buf
            h_expand = decoder.max_h // comp["h"]
            v_expand = decoder.max_v // comp["v"]
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

        y_buf = upsample_full(components[0], comp_buf[y_id], comp_w[y_id])
        second_id = components[1]["id"]
        third_id = components[2]["id"]
        second_buf = upsample_full(components[1], comp_buf[second_id], comp_w[second_id])
        third_buf = upsample_full(components[2], comp_buf[third_id], comp_w[third_id])
        if len(components) == 4:
            fourth_id = components[3]["id"]
            fourth_buf = upsample_full(components[3], comp_buf[fourth_id], comp_w[fourth_id])
            use_ycck = decoder.adobe_transform == 2
            use_inverted_cmyk = decoder.adobe_transform == 0
            for y in range(h):
                y_row = y * w
                off = y_row * 3
                for x in range(w):
                    idx = y_row + x
                    if use_ycck:
                        r, g, b = ycck_to_rgb_channels(
                            y_buf[idx],
                            second_buf[idx],
                            third_buf[idx],
                            fourth_buf[idx],
                        )
                    elif use_inverted_cmyk:
                        r, g, b = inverted_cmyk_to_rgb_channels(
                            y_buf[idx],
                            second_buf[idx],
                            third_buf[idx],
                            fourth_buf[idx],
                        )
                    else:
                        r, g, b = cmyk_to_rgb_channels(
                            y_buf[idx],
                            second_buf[idx],
                            third_buf[idx],
                            fourth_buf[idx],
                        )
                    rgb[off] = r
                    rgb[off + 1] = g
                    rgb[off + 2] = b
                    off += 3
            return bytes(rgb)
        for y in range(h):
            y_row = y * w
            off = y_row * 3
            for x in range(w):
                y_sample = y_buf[y_row + x]
                cb = second_buf[y_row + x]
                cr = third_buf[y_row + x]
                r, g, b = ycbcr_to_rgb_channels(y_sample, cb, cr)
                rgb[off] = r
                rgb[off + 1] = g
                rgb[off + 2] = b
                off += 3
    return bytes(rgb)


def decode_progressive(decoder: Any) -> bytes:
    mcu_cols = (decoder.width + decoder.mcu_width - 1) // decoder.mcu_width
    mcu_rows = (decoder.height + decoder.mcu_height - 1) // decoder.mcu_height
    comp_w: dict[int, int] = {}
    comp_h: dict[int, int] = {}
    blocks_w: dict[int, int] = {}
    blocks_h: dict[int, int] = {}
    coeff_buf: dict[int, list[int]] = {}
    quant_tables = decoder.quant_tables
    for comp in decoder.components:
        cw = (decoder.width * comp["h"] + decoder.max_h - 1) // decoder.max_h
        ch = (decoder.height * comp["v"] + decoder.max_v - 1) // decoder.max_v
        comp_id = comp["id"]
        comp_w[comp_id] = cw
        comp_h[comp_id] = ch
        blocks_w[comp_id] = mcu_cols * comp["h"]
        blocks_h[comp_id] = mcu_rows * comp["v"]
        coeff_buf[comp_id] = [0] * (blocks_w[comp_id] * blocks_h[comp_id] * 64)

    for scan in decoder.scans:
        ss = scan["Ss"]
        se = scan["Se"]
        ah = scan["Ah"]
        al = scan["Al"]
        scan_components = scan["components"]
        if ah != 0 and al != ah - 1:
            raise JpegUnsupportedError("Invalid progressive successive approximation")
        for item in scan_components:
            item["comp"]["prev_dc"] = 0

        def get_huff_table(
            tbl: HuffmanTable | None,
            table_class: int,
            table_id: int,
        ) -> HuffmanTable:
            return tbl if tbl is not None else decoder.huffman_tables[(table_class, table_id)]

        scan_meta = [
            (
                item["comp"],
                get_huff_table(item.get("dc_tbl_ref"), 0, item["dc_tbl"]),
                get_huff_table(item.get("ac_tbl_ref"), 1, item["ac_tbl"]),
                blocks_w[item["comp"]["id"]],
                blocks_h[item["comp"]["id"]],
            )
            for item in scan_components
        ]
        reader = JpegBitReader(scan["data"])
        decoder.mcu_counter = 0
        if ss == 0:
            if se != 0:
                raise JpegUnsupportedError("Invalid progressive DC spectral selection")
            if ah == 0:
                decode_progressive_dc_first_scan(
                    decoder, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
                )
            else:
                decode_progressive_dc_refine_scan(
                    decoder, reader, scan_meta, coeff_buf, mcu_cols, mcu_rows, al
                )
        else:
            if len(scan_meta) != 1:
                raise JpegUnsupportedError("Progressive AC scan must contain one component")
            if se >= 64 or ss > se:
                raise JpegUnsupportedError("Invalid progressive AC spectral selection")
            if ah == 0:
                decode_progressive_ac_first_scan(
                    decoder,
                    reader,
                    scan_meta,
                    coeff_buf,
                    mcu_cols,
                    mcu_rows,
                    ss,
                    se,
                    al,
                )
            else:
                decode_progressive_ac_refine_scan(
                    decoder,
                    reader,
                    scan_meta,
                    coeff_buf,
                    mcu_cols,
                    mcu_rows,
                    ss,
                    se,
                    al,
                )

    comp_pixels: dict[int, list[int]] = {}
    idct = decoder.idct_2d
    block = decoder.block
    for comp in decoder.components:
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

    return compose_rgb(decoder, comp_pixels, comp_w)


def decode_progressive_dc_first_scan(
    decoder: Any,
    reader: JpegBitReader,
    scan_meta: list[ScanMeta],
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
            decoder.mcu_counter += 1
            if decoder.restart_interval:
                handle_restart(decoder, reader)


def decode_progressive_dc_refine_scan(
    decoder: Any,
    reader: JpegBitReader,
    scan_meta: list[ScanMeta],
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
            decoder.mcu_counter += 1
            if decoder.restart_interval:
                handle_restart(decoder, reader)


def decode_progressive_ac_first_scan(
    decoder: Any,
    reader: JpegBitReader,
    scan_meta: list[ScanMeta],
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
                decoder.mcu_counter += 1
                if decoder.restart_interval and handle_restart(decoder, reader):
                    eobrun = 0
            except JpegUnsupportedError:
                coeffs[base : base + 64] = saved
                return


def decode_progressive_ac_refine_scan(
    decoder: Any,
    reader: JpegBitReader,
    scan_meta: list[ScanMeta],
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
                        newcoef = 0
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
                decoder.mcu_counter += 1
                if decoder.restart_interval and handle_restart(decoder, reader):
                    eobrun = 0
            except JpegUnsupportedError:
                coeffs[base : base + 64] = saved
                return


def handle_restart(decoder: Any, reader: JpegBitReader) -> bool:
    if decoder.restart_interval == 0:
        return False
    if decoder.mcu_counter % decoder.restart_interval != 0:
        return False
    if reader.bits_left != 0:
        reader.bits_left = 0
        reader.buffer = 0
    if reader.pos + 2 > len(reader.data):
        return False
    if reader.data[reader.pos] == 0xFF and 0xD0 <= reader.data[reader.pos + 1] <= 0xD7:
        reader.pos += 2
        for comp in decoder.components:
            comp["prev_dc"] = 0
        return True
    return False
