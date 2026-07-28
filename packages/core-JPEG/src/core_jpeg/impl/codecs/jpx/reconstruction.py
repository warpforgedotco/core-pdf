# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_jpeg.impl.codecs.jpx.output import (
    place_tile_bytes,
    tile_interleaved_samples,
)
from core_jpeg.impl.codecs.jpx.packets import JpxPacketStreamConsumed
from core_jpeg.impl.codecs.jpx.params import (
    JpxCodingParams,
    coding_component_quant_guard_bits,
    coding_component_quant_steps,
    coding_component_quant_style,
    coding_component_roi_shift,
    coding_component_style_params,
)
from core_jpeg.impl.codecs.jpx.progression import (
    decode_progression_packet_streams,
    decode_subband_codeblocks,
)
from core_jpeg.impl.codecs.jpx.structures import TileComponent
from core_jpeg.impl.codecs.jpx.wavelet import (
    apply_roi_shift_subband,
    dequantize_subband,
    quant_num_bitplanes,
    resolve_quant_step,
    synthesize_component,
)
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError


def decode_tile_payload(
    image: Any,
    tile_index: int,
    payload: bytes,
    coding_params: JpxCodingParams | None = None,
) -> int:
    return decode_tile_payload_stream(
        image,
        tile_index,
        payload,
        coding_params,
    ).body


def decode_tile_payload_stream(
    image: Any,
    tile_index: int,
    payload: bytes,
    coding_params: JpxCodingParams | None = None,
    *,
    packet_headers: bytes | None = None,
    packet_header_offset: int = 0,
) -> JpxPacketStreamConsumed:
    params = coding_params if coding_params is not None else image.coding_params()
    tile = image.ensure_tile(tile_index, params)
    if coding_params is None:
        params = tile.get("coding_params", params)
    else:
        tile["coding_params"] = params
    included_packet_keys = tile.setdefault("included_packet_keys", set())
    consumed = decode_progression_packet_streams(
        payload,
        tile["components"],
        params.prog_order,
        params.num_layers,
        params.codeblock_style,
        packet_uses_sop=params.packet_uses_sop,
        packet_uses_eph=params.packet_uses_eph,
        progression_changes=params.progression_changes,
        packet_headers=packet_headers,
        packet_header_offset=packet_header_offset,
        included_packet_keys=included_packet_keys,
        # Nsop is tile-scoped (ISO/IEC 15444-1 A.8.1), not tile-part-scoped.
        packet_sequence_offset=int(tile.get("packet_sequence", 0)),
    )
    tile["packet_sequence"] = (
        int(tile.get("packet_sequence", 0)) + int(consumed.positions)
    ) % 65536
    decode_tile_components(image, tile, params)
    return consumed


def reconstruct_component(
    image: Any,
    comp: TileComponent,
    comp_index: int,
    coding_params: JpxCodingParams | None = None,
) -> None:
    params = coding_params if coding_params is not None else image.coding_params()
    component_params = coding_component_style_params(params, comp_index)
    quant_steps = coding_component_quant_steps(params, comp_index)
    quant_guard_bits = coding_component_quant_guard_bits(params, comp_index)
    quant_style = coding_component_quant_style(params, comp_index)
    roi_shift = coding_component_roi_shift(params, comp_index)
    component_spec = image.components_data[min(comp_index, len(image.components_data) - 1)]
    precision = int(component_spec.get("precision", 8))
    for res in comp.resolutions:
        for subband in res.subbands:
            ignored_mantissa, exponent = resolve_quant_step(
                quant_style,
                quant_steps,
                component_params.levels,
                subband,
            )
            num_bitplanes = quant_num_bitplanes(exponent, quant_guard_bits)
            decode_subband_codeblocks(
                subband,
                num_bitplanes=num_bitplanes + roi_shift,
                codeblock_style=component_params.codeblock_style,
            )
            apply_roi_shift_subband(subband, roi_shift)
            dequantize_subband(
                subband,
                quant_steps,
                quant_style,
                component_params.levels,
                precision,
                component_params.reversible,
            )
    build_image(comp, component_params.reversible)


def decode_tile_components(
    image: Any,
    tile: dict[Any, Any],
    coding_params: JpxCodingParams | None = None,
) -> bool:
    params = (
        coding_params
        if coding_params is not None
        else tile.get(
            "coding_params",
            image.coding_params(),
        )
    )
    for comp_index, comp in enumerate(tile["components"]):
        reconstruct_component(image, comp, comp_index, params)
    return True


def build_image(comp: TileComponent, reversible: bool) -> None:
    samples, ignored_width, ignored_height = synthesize_component(
        comp,
        reversible=reversible,
    )
    if comp.resolutions:
        comp.resolutions[0].subbands[0].samples = samples
        comp.resolutions[0].subbands[0].width = comp.width
        comp.resolutions[0].subbands[0].height = comp.height


def to_raw(image: Any, component_mode: str = "default") -> bytes:
    if not image.tiles:
        raise JpegUnsupportedError("JPXDecode produced no image tiles")
    first_decoded = (
        image.decoded_tile_data[0]
        if component_mode == "default" and image.decoded_tile_data
        else None
    )
    if first_decoded is None:
        first_tile = image.tiles[0]
        if not first_tile:
            first_tile = image.ensure_tile(0, image.coding_params())
        first_params = first_tile.get("coding_params", image.coding_params())
        first_channels, ignored_first_data = tile_interleaved_samples(
            first_tile,
            image.components_data,
            first_params.multiple_component_transform,
            first_params.reversible,
            component_mode=component_mode,
        )
    else:
        first_channels, ignored_first_data = first_decoded
    if first_channels <= 0:
        raise JpegUnsupportedError("JPXDecode produced no image samples")
    out = bytearray(image.width * image.height * first_channels)
    for tile_index, tile in enumerate(image.tiles):
        if not tile:
            tile = image.ensure_tile(tile_index, image.coding_params())
        tile_w = int(tile.get("width") or 0)
        tile_h = int(tile.get("height") or 0)
        tile_x = int(tile.get("ref_x0", 0)) - image.x_origin
        tile_y = int(tile.get("ref_y0", 0)) - image.y_origin
        decoded = (
            image.decoded_tile_data[tile_index]
            if component_mode == "default" and tile_index < len(image.decoded_tile_data)
            else None
        )
        if decoded is None:
            params = tile.get("coding_params", image.coding_params())
            channels, tile_data = tile_interleaved_samples(
                tile,
                image.components_data,
                params.multiple_component_transform,
                params.reversible,
                component_mode=component_mode,
            )
        else:
            channels, tile_data = decoded
        if channels != first_channels:
            raise JpegParseError("JPX tile channel count changed")
        place_tile_bytes(
            out,
            image.width,
            tile_x,
            tile_y,
            tile_w,
            tile_h,
            channels,
            tile_data,
        )
    return bytes(out)
