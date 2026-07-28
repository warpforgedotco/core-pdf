# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_jpeg.impl.codecs.jpx.packets import (
    JpxCodeBlockChunk,
    JpxDecodedPacket,
    JpxPacketPosition,
    JpxPacketSegment,
    JpxPacketStreamConsumed,
    JpxPrecinct,
    PacketBitReader,
    append_codeblock_chunk,
    decode_packet_codeblock_chunks,
    decode_packet_codeblock_segments,
    skip_eph_marker,
    skip_sop_marker,
)
from core_jpeg.impl.codecs.jpx.params import (
    CPRL,
    LRCP,
    PCRL,
    RLCP,
    RPCL,
    JpxProgressionChange,
)
from core_jpeg.impl.codecs.jpx.structures import SubBand, TileComponent, ceil_div
from core_jpeg.impl.codecs.jpx.tier1 import decode_tier1_codeblock_segments
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError


def decode_packet_into_subband(
    packet: bytes,
    subband: SubBand,
    layer: int,
    precinct: JpxPrecinct | None = None,
    codeblock_style: int = 0,
) -> JpxDecodedPacket:
    precinct_state = precinct if precinct is not None else subband.precincts[0]
    if precinct_state.inclusion is None or precinct_state.zero_bit_planes is None:
        return JpxDecodedPacket(chunks=[], consumed=0)
    chunks, consumed = decode_packet_codeblock_chunks(
        packet,
        blocks_w=precinct_state.blocks_w,
        blocks_h=precinct_state.blocks_h,
        layer=layer,
        states=precinct_state.packet_states,
        inclusion=precinct_state.inclusion,
        zero_bit_planes=precinct_state.zero_bit_planes,
        codeblock_style=codeblock_style,
    )
    for chunk in chunks:
        block_index = precinct_state.global_block_index(
            chunk.block_index,
            subband.num_blocks_h,
        )
        block = subband.blocks[block_index]
        append_codeblock_chunk(
            block,
            segment_index=chunk.segment_index,
            num_passes=chunk.num_passes,
            zero_bit_planes=chunk.zero_bit_planes,
            payload=chunk.payload,
        )
    return JpxDecodedPacket(chunks=chunks, consumed=consumed)


def decode_packet_header_segments(
    packet_header: bytes,
    position: JpxPacketPosition,
    *,
    offset: int = 0,
    packet_uses_eph: bool = False,
) -> tuple[list[tuple[SubBand, JpxPrecinct, JpxPacketSegment]], int]:
    reader = PacketBitReader(packet_header, offset)
    if reader.read_bit() == 0:
        return [], skip_eph_marker(packet_header, reader.align(), packet_uses_eph)

    entries: list[tuple[SubBand, JpxPrecinct, JpxPacketSegment]] = []
    for subband, precinct in zip(position.bands, position.precinct_states, strict=True):
        if precinct.inclusion is None or precinct.zero_bit_planes is None:
            continue
        for block_y in range(precinct.blocks_h):
            for block_x in range(precinct.blocks_w):
                local_index = block_y * precinct.blocks_w + block_x
                block_segments = decode_packet_codeblock_segments(
                    reader,
                    block_index=local_index,
                    block_x=block_x,
                    block_y=block_y,
                    layer=position.layer,
                    state=precinct.packet_states[local_index],
                    inclusion=precinct.inclusion,
                    zero_bit_planes=precinct.zero_bit_planes,
                    codeblock_style=position.codeblock_style,
                )
                for segment in block_segments:
                    entries.append((subband, precinct, segment))
    return entries, skip_eph_marker(packet_header, reader.align(), packet_uses_eph)


def decode_packet_body_segments(
    packet_body: bytes,
    entries: list[tuple[SubBand, JpxPrecinct, JpxPacketSegment]],
    *,
    offset: int = 0,
) -> tuple[list[JpxCodeBlockChunk], int]:
    chunks: list[JpxCodeBlockChunk] = []
    for subband, precinct, segment in entries:
        end = offset + segment.length
        if end > len(packet_body):
            raise JpegParseError("unexpected end of JPX packet body")
        global_index = precinct.global_block_index(
            segment.block_index,
            subband.num_blocks_h,
        )
        block = subband.blocks[global_index]
        payload = packet_body[offset:end]
        append_codeblock_chunk(
            block,
            segment_index=segment.segment_index,
            num_passes=segment.num_passes,
            zero_bit_planes=segment.zero_bit_planes,
            payload=payload,
        )
        chunks.append(
            JpxCodeBlockChunk(
                block_index=global_index,
                segment_index=segment.segment_index,
                num_passes=segment.num_passes,
                zero_bit_planes=segment.zero_bit_planes,
                payload=payload,
            )
        )
        offset = end
    return chunks, offset


def decode_packet_position(
    packet: bytes, position: JpxPacketPosition, packet_uses_eph: bool = False
) -> JpxDecodedPacket:
    entries, body_offset = decode_packet_header_segments(
        packet,
        position,
        packet_uses_eph=packet_uses_eph,
    )
    chunks, consumed = decode_packet_body_segments(
        packet,
        entries,
        offset=body_offset,
    )
    return JpxDecodedPacket(
        chunks=chunks,
        consumed=consumed,
        header_consumed=body_offset,
    )


def decode_packet_position_with_packed_headers(
    packet_headers: bytes,
    packet_body: bytes,
    position: JpxPacketPosition,
    *,
    header_offset: int = 0,
    body_offset: int = 0,
    packet_uses_eph: bool = False,
) -> JpxDecodedPacket:
    entries, header_end = decode_packet_header_segments(
        packet_headers,
        position,
        offset=header_offset,
        packet_uses_eph=packet_uses_eph,
    )
    chunks, body_end = decode_packet_body_segments(
        packet_body,
        entries,
        offset=body_offset,
    )
    return JpxDecodedPacket(
        chunks=chunks,
        consumed=body_end - body_offset,
        header_consumed=header_end - header_offset,
    )


def codeblock_dimensions(subband: SubBand, block_x: int, block_y: int) -> tuple[int, int]:
    if block_x < 0 or block_y < 0:
        raise ValueError("invalid JPX code-block coordinate")
    if block_x >= subband.num_blocks_h or block_y >= subband.num_blocks_v:
        raise ValueError("JPX code-block coordinate out of range")
    grid_x0 = (subband.block_origin_x + block_x) * subband.codeblock_w
    grid_y0 = (subband.block_origin_y + block_y) * subband.codeblock_h
    x0 = max(grid_x0, subband.x0)
    y0 = max(grid_y0, subband.y0)
    x1 = min(grid_x0 + subband.codeblock_w, subband.x1)
    y1 = min(grid_y0 + subband.codeblock_h, subband.y1)
    return max(0, x1 - x0), max(0, y1 - y0)


def place_codeblock_samples(subband: SubBand, block_index: int, samples: list[int]) -> None:
    if subband.num_blocks_h <= 0:
        raise ValueError("JPX subband has no code-block columns")
    block_x = block_index % subband.num_blocks_h
    block_y = block_index // subband.num_blocks_h
    width, height = codeblock_dimensions(subband, block_x, block_y)
    if len(samples) < width * height:
        raise JpegParseError("JPX code-block decoded too few samples")
    grid_x0 = (subband.block_origin_x + block_x) * subband.codeblock_w
    grid_y0 = (subband.block_origin_y + block_y) * subband.codeblock_h
    x0 = max(grid_x0, subband.x0) - subband.x0
    y0 = max(grid_y0, subband.y0) - subband.y0
    for row in range(height):
        src = row * width
        dst = (y0 + row) * subband.width + x0
        subband.samples[dst : dst + width] = samples[src : src + width]


def decode_subband_codeblocks(
    subband: SubBand,
    num_bitplanes: int,
    codeblock_style: int = 0,
) -> None:
    subband.samples = [0] * (subband.width * subband.height)
    for block_index, block in enumerate(subband.blocks):
        if not block.segments:
            continue
        block_x = block_index % subband.num_blocks_h
        block_y = block_index // subband.num_blocks_h
        block_w, block_h = codeblock_dimensions(subband, block_x, block_y)
        samples = decode_tier1_codeblock_segments(
            block,
            width=block_w,
            height=block_h,
            num_bitplanes=num_bitplanes,
            orientation=subband.orientation,
            codeblock_style=codeblock_style,
        )
        place_codeblock_samples(subband, block_index, samples)


def iter_progression_packet_positions(
    components: list[TileComponent],
    prog_order: int,
    num_layers: int,
    codeblock_style: int = 0,
    progression_changes: list[JpxProgressionChange] | None = None,
) -> list[JpxPacketPosition]:
    if num_layers <= 0:
        return []
    max_resolutions = max((len(comp.resolutions) for comp in components), default=0)

    def packet_templates_for(
        layer: int, resolution: int, component: int
    ) -> list[tuple[tuple[int, int], JpxPacketPosition]]:
        if component >= len(components):
            return []
        comp = components[component]
        if resolution >= len(comp.resolutions):
            return []
        positions: list[tuple[tuple[int, int], JpxPacketPosition]] = []
        subbands = [
            subband for subband in comp.resolutions[resolution].subbands if subband.precincts
        ]
        precinct_count = max((len(subband.precincts) for subband in subbands), default=0)
        for precinct_index in range(precinct_count):
            bands: list[SubBand] = []
            precincts: list[JpxPrecinct] = []
            for subband in subbands:
                if precinct_index < len(subband.precincts):
                    bands.append(subband)
                    precincts.append(subband.precincts[precinct_index])
            if bands:
                packet = JpxPacketPosition(
                    layer=layer,
                    resolution=resolution,
                    component=component,
                    precinct=precinct_index,
                    codeblock_style=(
                        comp.codeblock_style
                        if comp.codeblock_style is not None
                        else codeblock_style
                    ),
                    bands=tuple(bands),
                    precinct_states=tuple(precincts),
                )
                positions.append(
                    (
                        precinct_reference_position_key(comp, resolution, packet),
                        packet,
                    )
                )
        return positions

    def positions_for(layer: int, resolution: int, component: int) -> list[JpxPacketPosition]:
        return [
            packet for ignored_key, packet in packet_templates_for(layer, resolution, component)
        ]

    def add_positions_for_order(
        positions: list[JpxPacketPosition],
        order: int,
        layer_start: int,
        layer_end: int,
        resolution_start: int,
        resolution_end: int,
        component_start: int,
        component_end: int,
    ) -> None:
        if order == LRCP:
            for layer in range(layer_start, layer_end):
                for resolution in range(resolution_start, resolution_end):
                    for component in range(component_start, component_end):
                        positions.extend(positions_for(layer, resolution, component))
        elif order == RLCP:
            for resolution in range(resolution_start, resolution_end):
                for layer in range(layer_start, layer_end):
                    for component in range(component_start, component_end):
                        positions.extend(positions_for(layer, resolution, component))
        elif order == RPCL:
            for resolution in range(resolution_start, resolution_end):
                packet_templates = packet_templates_for_range(
                    resolution,
                    component_start,
                    component_end,
                    layer_start,
                    layer_end,
                )
                for key in sorted(packet_templates):
                    component_packets = packet_templates[key]
                    for component in range(component_start, component_end):
                        for layer in range(layer_start, layer_end):
                            packet = component_packets.get((component, layer))
                            if packet is not None:
                                positions.append(packet)
        elif order == PCRL:
            component_resolution_templates = packet_templates_for_component_resolution_range(
                component_start,
                component_end,
                resolution_start,
                resolution_end,
                layer_start,
                layer_end,
            )
            for key in sorted(component_resolution_templates):
                component_resolution_packets = component_resolution_templates[key]
                for component in range(component_start, component_end):
                    for resolution in range(resolution_start, resolution_end):
                        for layer in range(layer_start, layer_end):
                            packet = component_resolution_packets.get(
                                (component, resolution, layer)
                            )
                            if packet is not None:
                                positions.append(packet)
        elif order == CPRL:
            for component in range(component_start, component_end):
                packet_templates = packet_templates_for_component(
                    component,
                    resolution_start,
                    resolution_end,
                    layer_start,
                    layer_end,
                )
                for key in sorted(packet_templates):
                    resolution_packets = packet_templates[key]
                    for resolution in range(resolution_start, resolution_end):
                        for layer in range(layer_start, layer_end):
                            packet = resolution_packets.get((resolution, layer))
                            if packet is not None:
                                positions.append(packet)
        else:
            raise JpegUnsupportedError("unsupported JPX progression order")

    def packet_templates_for_range(
        resolution: int,
        component_start: int,
        component_end: int,
        layer_start: int,
        layer_end: int,
    ) -> dict[tuple[int, int], dict[tuple[int, int], JpxPacketPosition]]:
        grouped: dict[tuple[int, int], dict[tuple[int, int], JpxPacketPosition]] = {}
        for component in range(component_start, component_end):
            for layer in range(layer_start, layer_end):
                for key, packet in packet_templates_for(layer, resolution, component):
                    grouped.setdefault(key, {})[(component, layer)] = packet
        return grouped

    def packet_templates_for_component_resolution_range(
        component_start: int,
        component_end: int,
        resolution_start: int,
        resolution_end: int,
        layer_start: int,
        layer_end: int,
    ) -> dict[tuple[int, int], dict[tuple[int, int, int], JpxPacketPosition]]:
        grouped: dict[tuple[int, int], dict[tuple[int, int, int], JpxPacketPosition]] = {}
        for component in range(component_start, component_end):
            for resolution in range(resolution_start, resolution_end):
                for layer in range(layer_start, layer_end):
                    for key, packet in packet_templates_for(layer, resolution, component):
                        grouped.setdefault(key, {})[(component, resolution, layer)] = packet
        return grouped

    def packet_templates_for_component(
        component: int,
        resolution_start: int,
        resolution_end: int,
        layer_start: int,
        layer_end: int,
    ) -> dict[tuple[int, int], dict[tuple[int, int], JpxPacketPosition]]:
        grouped: dict[tuple[int, int], dict[tuple[int, int], JpxPacketPosition]] = {}
        for resolution in range(resolution_start, resolution_end):
            for layer in range(layer_start, layer_end):
                for key, packet in packet_templates_for(layer, resolution, component):
                    grouped.setdefault(key, {})[(resolution, layer)] = packet
        return grouped

    positions: list[JpxPacketPosition] = []
    if progression_changes:
        included: set[tuple[int, int, int, int]] = set()
        for change in progression_changes:
            change_positions: list[JpxPacketPosition] = []
            add_positions_for_order(
                change_positions,
                change.progression_order,
                0,
                min(change.layer_end, num_layers),
                min(change.resolution_start, max_resolutions),
                min(change.resolution_end, max_resolutions),
                min(change.component_start, len(components)),
                min(change.component_end, len(components)),
            )
            for position in change_positions:
                key = (
                    position.layer,
                    position.resolution,
                    position.component,
                    position.precinct,
                )
                if key not in included:
                    included.add(key)
                    positions.append(position)
        return positions

    add_positions_for_order(
        positions,
        prog_order,
        0,
        num_layers,
        0,
        max_resolutions,
        0,
        len(components),
    )
    return positions


def precinct_reference_position_key(
    component: TileComponent,
    resolution: int,
    packet: JpxPacketPosition,
) -> tuple[int, int]:
    levels = max(0, len(component.resolutions) - 1)
    scale = 1 << max(0, levels - resolution)
    y_positions: list[int] = []
    x_positions: list[int] = []
    for subband, precinct in zip(packet.bands, packet.precinct_states, strict=True):
        sample_x = precinct.block_x0 * subband.codeblock_w
        sample_y = precinct.block_y0 * subband.codeblock_h
        x_positions.append((component.x0 + sample_x * scale) * component.h_sep)
        y_positions.append((component.y0 + sample_y * scale) * component.v_sep)
    return (
        min(y_positions, default=0),
        min(x_positions, default=0),
    )


def decode_progression_packets(
    payload: bytes,
    components: list[TileComponent],
    prog_order: int,
    num_layers: int,
    codeblock_style: int = 0,
    *,
    packet_uses_sop: bool = False,
    packet_uses_eph: bool = False,
    progression_changes: list[JpxProgressionChange] | None = None,
) -> int:
    return decode_progression_packet_streams(
        payload,
        components,
        prog_order,
        num_layers,
        codeblock_style,
        packet_uses_sop=packet_uses_sop,
        packet_uses_eph=packet_uses_eph,
        progression_changes=progression_changes,
    ).body


def merge_ppm_packet_headers(markers: dict[int, bytes]) -> bytes:
    validate_marker_sequence(markers, "PPM")
    raw = b"".join(markers[index] for index in sorted(markers))
    offset = 0
    packet_headers = bytearray()
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise JpegParseError("truncated JPX PPM packet-header length")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        offset += 4
        end = offset + length
        if end > len(raw):
            raise JpegParseError("truncated JPX PPM packet-header data")
        packet_headers.extend(raw[offset:end])
        offset = end
    return bytes(packet_headers)


def merge_ppt_packet_headers(markers: dict[int, bytes]) -> bytes:
    validate_marker_sequence(markers, "PPT")
    return b"".join(markers[index] for index in sorted(markers))


def validate_marker_sequence(markers: dict[int, bytes], marker_name: str) -> None:
    indices = sorted(markers)
    if indices != list(range(len(indices))):
        raise JpegParseError(f"non-consecutive JPX {marker_name} marker indices")


def decode_progression_packet_streams(
    payload: bytes,
    components: list[TileComponent],
    prog_order: int,
    num_layers: int,
    codeblock_style: int = 0,
    *,
    packet_uses_sop: bool = False,
    packet_uses_eph: bool = False,
    progression_changes: list[JpxProgressionChange] | None = None,
    packet_headers: bytes | None = None,
    packet_header_offset: int = 0,
    packet_position_offset: int = 0,
    packet_sequence_offset: int = 0,
    included_packet_keys: set[tuple[int, int, int, int]] | None = None,
) -> JpxPacketStreamConsumed:
    body_offset = 0
    header_offset = packet_header_offset
    positions = iter_progression_packet_positions(
        components,
        prog_order,
        num_layers,
        codeblock_style,
        progression_changes=progression_changes,
    )
    decoded_positions = 0
    for position in positions[packet_position_offset:]:
        position_key = (
            position.layer,
            position.resolution,
            position.component,
            position.precinct,
        )
        if included_packet_keys is not None and position_key in included_packet_keys:
            continue
        packet_sequence = (packet_sequence_offset + decoded_positions) % 65536
        if packet_headers is None:
            if body_offset >= len(payload):
                break
            body_offset = skip_sop_marker(
                payload,
                body_offset,
                packet_uses_sop,
                packet_sequence,
            )
            decoded = decode_packet_position(
                payload[body_offset:],
                position,
                packet_uses_eph=packet_uses_eph,
            )
            body_offset += decoded.consumed
            if included_packet_keys is not None:
                included_packet_keys.add(position_key)
            decoded_positions += 1
            continue

        if header_offset >= len(packet_headers):
            break
        body_offset = skip_sop_marker(
            payload,
            body_offset,
            packet_uses_sop,
            packet_sequence,
        )
        decoded = decode_packet_position_with_packed_headers(
            packet_headers,
            payload,
            position,
            header_offset=header_offset,
            body_offset=body_offset,
            packet_uses_eph=packet_uses_eph,
        )
        header_offset += decoded.header_consumed
        body_offset += decoded.consumed
        if included_packet_keys is not None:
            included_packet_keys.add(position_key)
        decoded_positions += 1
    return JpxPacketStreamConsumed(
        body=body_offset,
        header=header_offset - packet_header_offset,
        positions=decoded_positions,
    )


def build_subband_precincts(
    subband: SubBand,
    precinct_width: int,
    precinct_height: int,
    *,
    precinct_grid_x0: int | None = None,
    precinct_grid_y0: int | None = None,
    precinct_cols: int | None = None,
    precinct_rows: int | None = None,
) -> list[JpxPrecinct]:
    if precinct_width <= 0 or precinct_height <= 0:
        raise ValueError("invalid JPX precinct size")
    precincts: list[JpxPrecinct] = []
    grid_x0 = subband.x0 if precinct_grid_x0 is None else precinct_grid_x0
    grid_y0 = subband.y0 if precinct_grid_y0 is None else precinct_grid_y0
    cols = (
        ceil_div(max(1, subband.x1 - grid_x0), precinct_width)
        if precinct_cols is None
        else precinct_cols
    )
    rows = (
        ceil_div(max(1, subband.y1 - grid_y0), precinct_height)
        if precinct_rows is None
        else precinct_rows
    )
    for precinct_y in range(rows):
        cell_y0 = grid_y0 + precinct_y * precinct_height
        cell_y1 = cell_y0 + precinct_height
        precinct_y0 = max(cell_y0, subband.y0)
        precinct_y1 = min(cell_y1, subband.y1)
        tlcblk_y = (precinct_y0 // subband.codeblock_h) * subband.codeblock_h
        brcblk_y = ceil_div(precinct_y1, subband.codeblock_h) * subband.codeblock_h
        block_y0 = max(0, (tlcblk_y // subband.codeblock_h) - subband.block_origin_y)
        block_y1 = min(
            subband.num_blocks_v,
            (brcblk_y // subband.codeblock_h) - subband.block_origin_y,
        )
        for precinct_x in range(cols):
            cell_x0 = grid_x0 + precinct_x * precinct_width
            cell_x1 = cell_x0 + precinct_width
            precinct_x0 = max(cell_x0, subband.x0)
            precinct_x1 = min(cell_x1, subband.x1)
            tlcblk_x = (precinct_x0 // subband.codeblock_w) * subband.codeblock_w
            brcblk_x = ceil_div(precinct_x1, subband.codeblock_w) * subband.codeblock_w
            block_x0 = max(
                0,
                (tlcblk_x // subband.codeblock_w) - subband.block_origin_x,
            )
            block_x1 = min(
                subband.num_blocks_h,
                (brcblk_x // subband.codeblock_w) - subband.block_origin_x,
            )
            blocks_w = max(0, block_x1 - block_x0)
            blocks_h = max(0, block_y1 - block_y0)
            precincts.append(JpxPrecinct(block_x0, block_y0, blocks_w, blocks_h))
    return precincts
