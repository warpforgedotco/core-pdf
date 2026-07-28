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
    packet_lengths: list[int] = []
    plt_next_index = 0
    saw_tile_poc = False
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
            if not saw_tile_poc:
                coding_params = replace(coding_params, progression_changes=[])
                saw_tile_poc = True
            coding_params = parse_poc_params(image, br, coding_params)
        elif marker == 0xFF61:
            if image.ppm_markers:
                raise JpegParseError("JPX codestream mixes PPM and PPT markers")
            parse_ppt_marker(br, ppt_markers)
        elif marker == 0xFF58:
            packet_lengths.extend(parse_plt_marker(br, plt_next_index))
            plt_next_index += 1
        elif marker == 0xFF64:
            skip_marker_segment(br)
        else:
            length = br.read_u16()
            if length < 2:
                raise JpegParseError("invalid JPX tile-part marker length")
            br.skip_bytes(length - 2)
    payload_start = br.byte
    if tile_part_length == 0:
        if br.data[data_len - 2 : data_len] != b"\xff\xd9":
            raise JpegParseError("zero-length JPX tile-part is not terminated by EOC")
        payload_end = data_len - 2
    else:
        payload_end = tile_start + tile_part_length
        if payload_end > data_len - 2:
            raise JpegParseError("invalid JPX tile-part length")
    if payload_end < payload_start:
        raise JpegParseError("invalid JPX tile payload")
    payload_size = payload_end - payload_start
    if packet_lengths and sum(packet_lengths) != payload_size:
        raise JpegParseError("JPX PLT packet lengths do not match tile-part payload")
    if image.plm_consume_index < len(image.plm_packet_lengths):
        plm_lengths = image.plm_packet_lengths[image.plm_consume_index]
        if sum(plm_lengths) != payload_size:
            raise JpegParseError("JPX PLM packet lengths do not match tile-part payload")
        if packet_lengths and tuple(plm_lengths) != tuple(packet_lengths):
            raise JpegParseError("JPX PLM and PLT packet lengths disagree")
        image.plm_consume_index += 1
    elif image.plm_packet_lengths:
        raise JpegParseError("JPX tile-part has no matching PLM entry")
    return JpxTilePartHeader(
        tile_index=tile_index,
        tile_part_index=tile_part_index,
        tile_part_count=tile_part_count,
        tile_part_length=tile_part_length,
        coding_params=coding_params,
        payload_start=payload_start,
        payload_end=payload_end,
        packet_headers=merge_ppt_packet_headers(ppt_markers) if ppt_markers else None,
        packet_lengths=tuple(packet_lengths),
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
            parse_plm(image, br)
        elif marker == 0xFF63:
            parse_crg(image, br)
        elif marker == 0xFF50:
            parse_cap(br)
        elif marker == 0xFF64:
            skip_marker_segment(br)
        else:
            if image.ppm_markers:
                image.packed_packet_headers = merge_ppm_packet_headers(image.ppm_markers)
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
    index = payload.read_byte()
    if index != image.tlm_next_index:
        raise JpegParseError("non-consecutive JPX TLM marker index")
    image.tlm_next_index += 1
    stlm = payload.read_byte()
    tile_index_size = (stlm >> 4) & 0x03
    if tile_index_size == 3:
        raise JpegParseError("invalid JPX TLM tile-index size")
    tile_part_length_size = ((stlm >> 6) & 0x01) + 1
    entry_size = tile_index_size + tile_part_length_size * 2
    remaining = len(payload.data) - payload.byte
    if entry_size == 0 or remaining % entry_size:
        raise JpegParseError("invalid JPX TLM entry length")
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
            raise JpegParseError("JPX TLM tile index out of range")
        tile_part_length = int.from_bytes(
            payload.read_bytes(tile_part_length_size * 2),
            "big",
        )
        entries.append((tile_index, tile_part_length))
    image.tile_part_lengths.extend(entries)


def parse_plm(image: Any, br: BitStream) -> None:
    length = br.read_u16()
    if length < 4:
        raise JpegParseError("PLM marker segment too short")
    payload = BitStream(br.read_bytes(length - 2))
    index = payload.read_byte()
    if index != image.plm_next_index:
        raise JpegParseError("non-consecutive JPX PLM marker index")
    image.plm_next_index += 1
    while payload.byte < len(payload.data):
        block_length = payload.read_byte()
        if block_length == 0 or payload.byte + block_length > len(payload.data):
            raise JpegParseError("invalid JPX PLM packet-length block")
        # Each Nplm block describes one tile-part's packet lengths in order.
        image.plm_packet_lengths.append(
            parse_packet_lengths(payload.read_bytes(block_length), "PLM")
        )


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


def parse_plt_marker(br: BitStream, expected_index: int = 0) -> list[int]:
    length = br.read_u16()
    if length < 3:
        raise JpegParseError("PLT marker segment too short")
    payload = br.read_bytes(length - 2)
    if payload[0] != expected_index:
        raise JpegParseError("non-consecutive JPX PLT marker index")
    return parse_packet_lengths(payload[1:], "PLT")


def parse_packet_lengths(payload: bytes, marker_name: str) -> list[int]:
    lengths: list[int] = []
    packet_length = 0
    continued = False
    for value in payload:
        packet_length = (packet_length << 7) | (value & 0x7F)
        continued = bool(value & 0x80)
        if not continued:
            lengths.append(packet_length)
            packet_length = 0
    if continued:
        raise JpegParseError(f"unterminated JPX {marker_name} packet length")
    return lengths


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


def parse_cap(br: BitStream) -> None:
    length = br.read_u16()
    if length < 6 or (length - 6) % 2:
        raise JpegParseError("invalid JPX CAP marker length")
    capabilities = br.read_u32()
    expected_words = capabilities.bit_count()
    if length != 6 + expected_words * 2:
        raise JpegParseError("JPX CAP capability fields do not match Pcap")
    br.read_bytes(expected_words * 2)
    if capabilities:
        raise JpegUnsupportedError("unsupported capability signaled by JPX CAP marker")


def parse_siz(image: Any, br: BitStream) -> None:
    lsiz = br.read_u16()
    if lsiz < 41:
        raise JpegParseError("SIZ too short")
    if (lsiz - 38) % 3:
        raise JpegParseError("bad JPX SIZ marker length")
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
        raise JpegParseError("JPX SIZ component count does not match marker length")
    if image.components == 0:
        raise JpegParseError("zero components")
    if image.components > 16384:
        raise JpegParseError("too many JPX components")
    if image.width <= 0 or image.height <= 0:
        raise JpegParseError("invalid JPX image size")
    if (
        image.expected_width is not None
        and image.expected_height is not None
        and (image.expected_width != image.width or image.expected_height != image.height)
    ):
        raise JpegParseError("JP2 IHDR dimensions do not match JPX SIZ marker")
    image.components_data = []
    for ignored in range(image.components):
        sample_spec = br.read_byte()
        precision = (sample_spec & 0x7F) + 1
        if precision > 38:
            raise JpegUnsupportedError("JPX component precision exceeds 38 bits")
        h_sep = br.read_byte()
        v_sep = br.read_byte()
        if h_sep <= 0 or v_sep <= 0:
            raise JpegParseError("invalid JPX component separation")
        image.components_data.append(
            {
                "precision": precision,
                "is_signed": bool(sample_spec & 0x80),
                "h_sep": h_sep,
                "v_sep": v_sep,
            }
        )
    if image.tile_width <= 0 or image.tile_height <= 0:
        raise JpegParseError("invalid JPX tile size")
    if (
        image.tile_x_origin > image.x_origin
        or image.tile_y_origin > image.y_origin
        or image.tile_x_origin + image.tile_width <= image.x_origin
        or image.tile_y_origin + image.tile_height <= image.y_origin
    ):
        raise JpegParseError("illegal JPX tile offset")
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
        raise JpegParseError("COD too short")
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
        raise JpegParseError("invalid JPX layer count")
    multiple_component_transform = br.read_byte()
    if multiple_component_transform > 1:
        raise JpegParseError("invalid JPX multiple component transform")
    levels = br.read_byte()
    cb_w = br.read_byte()
    cb_h = br.read_byte()
    validate_jpx_spcod_spcoc(levels, cb_w, cb_h)
    codeblock_w = 1 << (cb_w + 2)
    codeblock_h = 1 << (cb_h + 2)
    codeblock_style = br.read_byte()
    wavelet = br.read_byte()
    if wavelet > 1:
        raise JpegParseError("invalid JPX wavelet transform")
    reversible = wavelet == 1
    remaining = lcod - 12
    precincts: list[Any] = []
    if scod & 0x01:
        if remaining != levels + 1:
            raise JpegParseError("bad JPX precinct size list")
        for index in range(levels + 1):
            value = br.read_byte()
            validate_jpx_precinct_size(value, index)
            precincts.append((1 << (value & 0x0F), 1 << (value >> 4)))
    elif remaining:
        raise JpegParseError("bad JPX COD marker length")
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
        raise JpegParseError("COC too short")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise JpegParseError("COC component out of range")
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
        raise JpegParseError("invalid JPX wavelet transform")
    reversible = wavelet == 1
    remaining = length - 2 - component_bytes - 6
    precincts: list[Any] = []
    if scoc & 0x01:
        if remaining != levels + 1:
            raise JpegParseError("bad JPX component precinct size list")
        for index in range(levels + 1):
            value = br.read_byte()
            validate_jpx_precinct_size(value, index)
            precincts.append((1 << (value & 0x0F), 1 << (value >> 4)))
    elif remaining:
        raise JpegParseError("bad JPX COC marker length")
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
        raise JpegParseError("QCD too short")
    guard_bits, style, steps = parse_quantization(br, length - 2)
    return replace(
        base_params,
        quant_guard_bits=guard_bits,
        quant_style=style,
        quant_steps=[steps],
        quant_guard_bits_by_component=[guard_bits],
        quant_style_by_component=[style],
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
        raise JpegParseError("QCC too short")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise JpegParseError("QCC component out of range")
    guard_bits, style, steps = parse_quantization(br, length - 2 - component_bytes)
    return coding_params_with_component_quantization(
        base_params,
        component_index,
        guard_bits,
        style,
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
        raise JpegParseError("bad JPX RGN marker length")
    component_index = br.read_byte() if component_bytes == 1 else br.read_u16()
    if component_index >= image.components:
        raise JpegParseError("RGN component out of range")
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
        raise JpegParseError("bad JPX POC marker length")
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
        if component_end == 0:
            component_end = 256 if component_bytes == 1 else 16384
        if not (resolution_start < resolution_end <= 33):
            raise JpegParseError("invalid JPX POC resolution bounds")
        if layer_end < 1 or layer_end > base_params.num_layers:
            raise JpegParseError("invalid JPX POC layer bound")
        if not (component_start < component_end <= image.components):
            raise JpegParseError("invalid JPX POC component bounds")
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
) -> tuple[int, int, list[tuple[int, int]]]:
    if length < 1:
        raise JpegParseError("quantization segment too short")
    sqcx = br.read_byte()
    guard_bits = sqcx >> 5
    style = sqcx & 0x1F
    remaining = length - 1
    if style == 0:
        steps = [(0, br.read_byte() >> 3) for ignored in range(remaining)]
    elif style == 1:
        if remaining != 2:
            raise JpegParseError("bad derived quantization step")
        mantissa = br.read_u16()
        steps = [(mantissa & 0x7FF, mantissa >> 11)]
    elif style == 2:
        if remaining % 2:
            raise JpegParseError("bad expounded quantization steps")
        steps = []
        for ignored in range(remaining // 2):
            mantissa = br.read_u16()
            steps.append((mantissa & 0x7FF, mantissa >> 11))
    else:
        raise JpegUnsupportedError("unsupported JPX quantization style")
    return guard_bits, style, steps
