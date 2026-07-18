# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import replace
from typing import Any

from core_jpeg.impl.codecs.jpx.params import (
    CPRL,
    JPX_CODING_STYLE_SUPPORTED,
    JpxCodingParams,
    JpxComponentCodingParams,
    JpxProgressionChange,
    coding_params_with_component_quantization,
    coding_params_with_component_roi_shift,
    coding_params_with_component_style,
    copy_coding_params,
    validate_jpx_precinct_size,
    validate_jpx_spcod_spcoc,
)
from core_jpeg.impl.codecs.jpx.progression import (
    merge_ppm_packet_headers,
    merge_ppt_packet_headers,
)
from core_jpeg.impl.codecs.jpx.structures import BitStream, JpxTilePartHeader, ceil_div
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError


def read_tile_part_header(
    image: Any,
    br: BitStream,
    data_len: int,
    tile_params: dict[int, JpxCodingParams] | None = None,
    global_params: JpxCodingParams | None = None,
) -> JpxTilePartHeader:
    tile_start = br.byte - 2
    lsot = br.read_u16()
    if lsot != 10:
        raise JpegParseError("invalid JPX SOT length")
    tile_index = br.read_u16()
    tile_part_length = br.read_u32()
    tile_part_index = br.read_byte()
    tile_part_count = br.read_byte()
    if tile_part_length and tile_part_length < 14:
        raise JpegParseError("invalid JPX tile-part length")
    if tile_params is not None and tile_index in tile_params:
        base_params = tile_params[tile_index]
    elif global_params is not None:
        base_params = global_params
    else:
        base_params = image.coding_params()
    coding_params = copy_coding_params(base_params)
    ppt_markers: dict[int, bytes] = {}
    while True:
        marker = br.read_u16()
        if marker == 0xFF93:
            break
        if marker == 0xFFD9:
            raise JpegParseError("JPX tile-part ended before SOD")
        if marker == 0xFF52:
            coding_params = parse_cod_params(image, br, coding_params)
        elif marker == 0xFF53:
            coding_params = parse_coc_params(image, br, coding_params)
        elif marker == 0xFF5C:
            coding_params = parse_qcd_params(image, br, coding_params)
        elif marker == 0xFF5D:
            coding_params = parse_qcc_params(image, br, coding_params)
        elif marker == 0xFF5E:
            coding_params = parse_rgn_params(image, br, coding_params)
        elif marker == 0xFF5F:
            coding_params = parse_poc_params(image, br, coding_params)
        elif marker == 0xFF61:
            if image.ppm_markers:
                raise JpegParseError("JPX codestream mixes PPM and PPT markers")
            parse_ppt_marker(br, ppt_markers)
        elif marker == 0xFF58:
            parse_plt_marker(br)
        elif marker == 0xFF64:
            skip_marker_segment(br)
        else:
            length = br.read_u16()
            if length < 2:
                raise JpegParseError("invalid JPX tile-part marker length")
            br.skip_bytes(length - 2)
    payload_start = br.byte
    if tile_part_length == 0:
        payload_end = data_len - 2
    else:
        payload_end = tile_start + tile_part_length
        if payload_end > data_len:
            raise JpegParseError("invalid JPX tile-part length")
    if payload_end < payload_start:
        raise JpegParseError("invalid JPX tile payload")
    return JpxTilePartHeader(
        tile_index=tile_index,
        tile_part_index=tile_part_index,
        tile_part_count=tile_part_count,
        coding_params=coding_params,
        payload_start=payload_start,
        payload_end=payload_end,
        packet_headers=merge_ppt_packet_headers(ppt_markers) if ppt_markers else None,
    )


def parse_header(image: Any, br: BitStream) -> bool:
    seen_siz = False
    while True:
        marker = br.read_u16()
        if not seen_siz:
            if marker != 0xFF51:
                return False
            parse_siz(image, br)
            seen_siz = True
        elif marker == 0xFF51:
            return False
        elif marker == 0xFF52:
            parse_cod(image, br)
        elif marker == 0xFF53:
            parse_coc(image, br)
        elif marker == 0xFF5C:
            parse_qcd(image, br)
        elif marker == 0xFF5D:
            parse_qcc(image, br)
        elif marker == 0xFF5E:
            parse_rgn(image, br)
        elif marker == 0xFF5F:
            parse_poc(image, br)
        elif marker == 0xFF60:
            parse_ppm(image, br)
        elif marker == 0xFF55:
            parse_tlm(image, br)
        elif marker == 0xFF57:
            parse_plm(br)
        elif marker == 0xFF63:
            parse_crg(image, br)
        elif marker in {0xFF50, 0xFF64}:
            skip_marker_segment(br)
        else:
            return marker in {0xFF90, 0xFF93, 0xFFD9}


def skip_marker_segment(br: BitStream) -> None:
    length = br.read_u16()
    if length < 2:
        raise JpegParseError("invalid JPX marker segment length")
    br.skip_bytes(length - 2)


def parse_tlm(image: Any, br: BitStream) -> None:
    length = br.read_u16()
    if length < 4:
        raise JpegParseError("TLM marker segment too short")
    payload = BitStream(br.read_bytes(length - 2))
    payload.read_byte()
    stlm = payload.read_byte()
    tile_index_size = (stlm >> 4) & 0x03
    if tile_index_size == 3:
        return
    tile_part_length_size = ((stlm >> 6) & 0x01) + 1
    entry_size = tile_index_size + tile_part_length_size * 2
    remaining = len(payload.data) - payload.byte
    if entry_size == 0 or remaining % entry_size:
        return
    base_index = len(image.tile_part_lengths)
    tile_count = image.tiles_cols * image.tiles_rows
    entries: list[tuple[int, int]] = []
    for offset in range(remaining // entry_size):
        if tile_index_size == 0:
            tile_index = base_index + offset
        else:
            tile_index = int.from_bytes(
                payload.read_bytes(tile_index_size),
                "big",
            )
        if tile_count and tile_index >= tile_count:
            return
        tile_part_length = int.from_bytes(
            payload.read_bytes(tile_part_length_size * 2),
            "big",
        )
        entries.append((tile_index, tile_part_length))
    image.tile_part_lengths.extend(entries)


def parse_plm(br: BitStream) -> None:
    length = br.read_u16()
    if length < 3:
        raise JpegParseError("PLM marker segment too short")
    br.skip_bytes(length - 2)


def parse_ppm(image: Any, br: BitStream) -> None:
    length = br.read_u16()
    if length < 4:
        raise JpegParseError("PPM marker segment too short")
    index = br.read_byte()
    if index in image.ppm_markers:
        raise JpegParseError("duplicate JPX PPM marker index")
    image.ppm_markers[index] = br.read_bytes(length - 3)
    image.packed_packet_headers = None


def parse_ppt_marker(br: BitStream, markers: dict[int, bytes]) -> None:
    length = br.read_u16()
    if length < 4:
        raise JpegParseError("PPT marker segment too short")
    index = br.read_byte()
    if index in markers:
        raise JpegParseError("duplicate JPX PPT marker index")
    markers[index] = br.read_bytes(length - 3)


def parse_plt_marker(br: BitStream) -> None:
    length = br.read_u16()
    if length < 3:
        raise JpegParseError("PLT marker segment too short")
    payload = br.read_bytes(length - 2)
    packet_length = 0
    for value in payload[1:]:
        packet_length |= value & 0x7F
        if value & 0x80:
            packet_length <<= 7
        else:
            packet_length = 0
    if packet_length != 0:
        raise JpegParseError("unterminated JPX PLT packet length")


def parse_crg(image: Any, br: BitStream) -> None:
    length = br.read_u16()
    payload_size = length - 2
    if length < 2 or payload_size != image.components * 4:
        raise JpegParseError("bad JPX CRG marker length")
    br.skip_bytes(payload_size)


def ppm_packet_headers(image: Any) -> bytes | None:
    if not image.ppm_markers:
        return None
    if image.packed_packet_headers is None:
        image.packed_packet_headers = merge_ppm_packet_headers(image.ppm_markers)
    return image.packed_packet_headers


def parse_siz(image: Any, br: BitStream) -> None:
    lsiz = br.read_u16()
    if lsiz < 41:
        raise ValueError("SIZ too short")
    if (lsiz - 38) % 3:
        raise ValueError("bad JPX SIZ marker length")
    image.capabilities = br.read_u16()
    image.x_end = br.read_u32()
    image.y_end = br.read_u32()
    image.x_origin = br.read_u32()
    image.y_origin = br.read_u32()
    image.width = image.x_end - image.x_origin
    image.height = image.y_end - image.y_origin
    image.tile_width = br.read_u32()
    image.tile_height = br.read_u32()
    image.tile_x_origin = br.read_u32()
    image.tile_y_origin = br.read_u32()
    image.components = br.read_u16()
    expected_lsiz = 38 + 3 * image.components
    if lsiz != expected_lsiz:
        raise ValueError("JPX SIZ component count does not match marker length")
    if image.components == 0:
        raise ValueError("zero components")
    if image.components > 16384:
        raise ValueError("too many JPX components")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("invalid JPX image size")
    if (
        image.expected_width is not None
        and image.expected_height is not None
        and (image.expected_width != image.width or image.expected_height != image.height)
    ):
        raise JpegParseError("JP2 IHDR dimensions do not match JPX SIZ marker")
    image.components_data = []
    for ignored in range(image.components):
        sample_spec = br.read_byte()
        h_sep = br.read_byte()
        v_sep = br.read_byte()
        if h_sep <= 0 or v_sep <= 0:
            raise ValueError("invalid JPX component separation")
        image.components_data.append(
            {
                "precision": (sample_spec & 0x7F) + 1,
                "is_signed": bool(sample_spec & 0x80),
                "h_sep": h_sep,
                "v_sep": v_sep,
            }
        )
    if image.tile_width <= 0 or image.tile_height <= 0:
        raise ValueError("invalid JPX tile size")
    if (
        image.tile_x_origin > image.x_origin
        or image.tile_y_origin > image.y_origin
        or image.tile_x_origin + image.tile_width <= image.x_origin
        or image.tile_y_origin + image.tile_height <= image.y_origin
    ):
        raise ValueError("illegal JPX tile offset")
    image.tiles_cols = (
        ceil_div(image.image_x_end() - image.tile_x_origin, image.tile_width)
        if image.image_x_end() > image.tile_x_origin
        else 0
    )
    image.tiles_rows = (
        ceil_div(image.image_y_end() - image.tile_y_origin, image.tile_height)
        if image.image_y_end() > image.tile_y_origin
        else 0
    )


def parse_cod(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_cod_params(image, br, image.coding_params()))


def parse_cod_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    lcod = br.read_u16()
    if lcod < 8:
        raise ValueError("COD too short")
    scod = br.read_byte()
    if scod & ~JPX_CODING_STYLE_SUPPORTED:
        raise JpegUnsupportedError("unsupported JPX COD coding style")
    packet_uses_sop = bool(scod & 0x02)
    packet_uses_eph = bool(scod & 0x04)
    prog_order = br.read_byte()
    if prog_order > CPRL:
        raise JpegUnsupportedError("unsupported JPX progression order")
    num_layers = br.read_u16()
    if num_layers < 1:
        raise ValueError("invalid JPX layer count")
    multiple_component_transform = br.read_byte()
    if multiple_component_transform > 1:
        raise ValueError("invalid JPX multiple component transform")
    levels = br.read_byte()
    cb_w = br.read_byte()
    cb_h = br.read_byte()
    validate_jpx_spcod_spcoc(levels, cb_w, cb_h)
    codeblock_w = 1 << (cb_w + 2)
    codeblock_h = 1 << (cb_h + 2)
    codeblock_style = br.read_byte()
    wavelet = br.read_byte()
    if wavelet > 1:
        raise ValueError("invalid JPX wavelet transform")
    reversible = wavelet == 1
    remaining = lcod - 12
    precincts: list[Any] = []
    if scod & 0x01:
        if remaining != levels + 1:
            raise ValueError("bad JPX precinct size list")
        for index in range(levels + 1):
            value = br.read_byte()
            validate_jpx_precinct_size(value, index)
            precincts.append((1 << (value & 0x0F), 1 << (value >> 4)))
    elif remaining:
        raise ValueError("bad JPX COD marker length")
    return replace(
        base_params,
        levels=levels,
        codeblock_w=codeblock_w,
        codeblock_h=codeblock_h,
        codeblock_style=codeblock_style,
        prog_order=prog_order,
        num_layers=num_layers,
        packet_uses_sop=packet_uses_sop,
        packet_uses_eph=packet_uses_eph,
        multiple_component_transform=multiple_component_transform,
        precincts=precincts,
        reversible=reversible,
    )


def parse_coc(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_coc_params(image, br, image.coding_params()))


def parse_coc_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    length = br.read_u16()
    component_bytes = 1 if image.components < 257 else 2
    if length < component_bytes + 8:
        raise ValueError("COC too short")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise ValueError("COC component out of range")
    scoc = br.read_byte()
    levels = br.read_byte()
    cb_w = br.read_byte()
    cb_h = br.read_byte()
    validate_jpx_spcod_spcoc(levels, cb_w, cb_h)
    codeblock_w = 1 << (cb_w + 2)
    codeblock_h = 1 << (cb_h + 2)
    codeblock_style = br.read_byte()
    wavelet = br.read_byte()
    if wavelet > 1:
        raise ValueError("invalid JPX wavelet transform")
    reversible = wavelet == 1
    remaining = length - 2 - component_bytes - 6
    precincts: list[Any] = []
    if scoc & 0x01:
        if remaining != levels + 1:
            raise ValueError("bad JPX component precinct size list")
        for index in range(levels + 1):
            value = br.read_byte()
            validate_jpx_precinct_size(value, index)
            precincts.append((1 << (value & 0x0F), 1 << (value >> 4)))
    elif remaining:
        raise ValueError("bad JPX COC marker length")
    return coding_params_with_component_style(
        base_params,
        component_index,
        JpxComponentCodingParams(
            levels=levels,
            codeblock_w=codeblock_w,
            codeblock_h=codeblock_h,
            codeblock_style=codeblock_style,
            precincts=precincts,
            reversible=reversible,
        ),
    )


def parse_qcd(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_qcd_params(image, br, image.coding_params()))


def parse_qcd_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    length = br.read_u16()
    if length < 3:
        raise ValueError("QCD too short")
    guard_bits, steps = parse_quantization(br, length - 2)
    return replace(
        base_params,
        quant_guard_bits=guard_bits,
        quant_steps=[steps],
        quant_guard_bits_by_component=[guard_bits],
    )


def parse_qcc(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_qcc_params(image, br, image.coding_params()))


def parse_qcc_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    length = br.read_u16()
    component_bytes = 1 if image.components < 257 else 2
    if length < component_bytes + 3:
        raise ValueError("QCC too short")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise ValueError("QCC component out of range")
    guard_bits, steps = parse_quantization(br, length - 2 - component_bytes)
    return coding_params_with_component_quantization(
        base_params,
        component_index,
        guard_bits,
        steps,
    )


def parse_rgn(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_rgn_params(image, br, image.coding_params()))


def parse_rgn_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    length = br.read_u16()
    component_bytes = 1 if image.components <= 256 else 2
    if length != 2 + component_bytes + 2:
        raise ValueError("bad JPX RGN marker length")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise ValueError("RGN component out of range")
    roi_style = br.read_byte()
    if roi_style != 0:
        raise JpegUnsupportedError("unsupported JPX ROI style")
    roi_shift = br.read_byte()
    return coding_params_with_component_roi_shift(
        base_params,
        component_index,
        roi_shift,
    )


def parse_poc(image: Any, br: BitStream) -> None:
    image.set_coding_params(parse_poc_params(image, br, image.coding_params()))


def parse_poc_params(
    image: Any,
    br: BitStream,
    base_params: JpxCodingParams,
) -> JpxCodingParams:
    length = br.read_u16()
    component_bytes = 1 if image.components <= 256 else 2
    record_size = 5 + component_bytes * 2
    payload_size = length - 2
    if length < 2 + record_size or payload_size % record_size:
        raise ValueError("bad JPX POC marker length")
    changes = list(base_params.progression_changes)
    for ignored in range(payload_size // record_size):
        resolution_start = br.read_byte()
        component_start = br.read_byte() if component_bytes == 1 else br.read_u16()
        layer_end = br.read_u16()
        resolution_end = br.read_byte()
        component_end = br.read_byte() if component_bytes == 1 else br.read_u16()
        progression_order = br.read_byte()
        if progression_order > CPRL:
            raise JpegUnsupportedError("unsupported JPX POC progression order")
        changes.append(
            JpxProgressionChange(
                resolution_start=resolution_start,
                component_start=component_start,
                layer_end=layer_end,
                resolution_end=resolution_end,
                component_end=component_end,
                progression_order=progression_order,
            )
        )
    return replace(base_params, progression_changes=changes)


def parse_quantization(
    br: BitStream,
    length: int,
) -> tuple[int, list[tuple[int, int]]]:
    if length < 1:
        raise ValueError("quantization segment too short")
    sqcx = br.read_byte()
    guard_bits = sqcx >> 5
    style = sqcx & 0x1F
    remaining = length - 1
    if style == 0:
        steps = [(0, br.read_byte() >> 3) for ignored in range(remaining)]
    elif style == 1:
        if remaining != 2:
            raise ValueError("bad derived quantization step")
        mantissa = br.read_u16()
        steps = [(mantissa & 0x7FF, mantissa >> 11)]
    else:
        if remaining % 2:
            raise ValueError("bad expounded quantization steps")
        steps = []
        for ignored in range(remaining // 2):
            mantissa = br.read_u16()
            steps.append((mantissa & 0x7FF, mantissa >> 11))
    return guard_bits, steps
