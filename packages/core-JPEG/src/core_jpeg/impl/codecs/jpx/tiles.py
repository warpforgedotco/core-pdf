# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from core_jpeg.impl.codecs.jpx.output import (
    component_h_sep,
    component_samples,
    component_v_sep,
    tile_rgb_samples,
)
from core_jpeg.impl.codecs.jpx.params import (
    JpxCodingParams,
    JpxComponentCodingParams,
    coding_component_style_params,
    validate_jpx_coding_params,
)
from core_jpeg.impl.codecs.jpx.progression import build_subband_precincts
from core_jpeg.impl.codecs.jpx.structures import (
    CodeBlock,
    JpxTilePart,
    JpxTilePartHeader,
    SubBand,
    TileComponent,
    ceil_div,
)
from core_jpeg.impl.errors import JpegParseError

JPX_PARALLEL_TILE_MIN_BYTES = 256_000
JPX_PARALLEL_TILE_MAX_WORKERS = 8
JPX_WORKER_ENV = "CORE_JPEG_JPX_WORKERS"


def _jpx_tile_executor_class() -> Any:
    if sys.version_info < (3, 14):
        return ProcessPoolExecutor
    from concurrent.futures import InterpreterPoolExecutor

    return InterpreterPoolExecutor


InterpreterPoolExecutor = _jpx_tile_executor_class()


def validate_tile_part_header(
    image: Any,
    header: JpxTilePartHeader,
    *,
    tile_part_indices: dict[int, int],
    tile_part_counts: dict[int, int],
) -> None:
    tile_count = image.tiles_cols * image.tiles_rows
    if tile_count and header.tile_index >= tile_count:
        raise JpegParseError("JPX tile index out of range")
    seen_parts = sum(index + 1 for index in tile_part_indices.values())
    if seen_parts < len(image.tile_part_lengths):
        expected_tile, expected_length = image.tile_part_lengths[seen_parts]
        if (header.tile_index, header.tile_part_length) != (
            expected_tile,
            expected_length,
        ):
            raise JpegParseError("JPX SOT does not match TLM entry")
    expected_part = tile_part_indices.get(header.tile_index, -1) + 1
    if header.tile_part_index != expected_part:
        raise JpegParseError("JPX tile-part index out of order")
    known_count = tile_part_counts.get(header.tile_index, 0)
    if known_count and header.tile_part_index >= known_count:
        raise JpegParseError("JPX tile-part index exceeds tile-part count")
    if header.tile_part_count:
        if known_count and header.tile_part_count != known_count:
            raise JpegParseError("JPX tile-part count changed")
        if header.tile_part_index >= header.tile_part_count:
            raise JpegParseError("JPX tile-part index exceeds tile-part count")
        tile_part_counts[header.tile_index] = header.tile_part_count
    tile_part_indices[header.tile_index] = header.tile_part_index


def validate_tile_part_completion(
    image: Any,
    *,
    tile_part_indices: dict[int, int],
    tile_part_counts: dict[int, int],
) -> None:
    for tile_index, count in tile_part_counts.items():
        if tile_part_indices.get(tile_index, -1) + 1 != count:
            raise JpegParseError("JPX codestream ended before all tile-parts")
    seen_parts = sum(index + 1 for index in tile_part_indices.values())
    if image.tile_part_lengths and seen_parts != len(image.tile_part_lengths):
        raise JpegParseError("JPX codestream tile-parts do not match TLM")
    if image.plm_packet_lengths and image.plm_consume_index != len(image.plm_packet_lengths):
        raise JpegParseError("JPX codestream tile-parts do not match PLM")


def decode_tile_parts(image: Any, parts: list[JpxTilePart]) -> None:
    initialize_tile_slots(image)
    if not parts:
        return
    ppm_headers = image.ppm_packet_headers()
    if ppm_headers is not None:
        decode_tile_parts_with_ppm(image, parts, ppm_headers)
        return
    grouped: dict[int, list[JpxTilePart]] = {}
    for part in sorted(parts, key=lambda item: (item.tile_index, item.tile_part_index)):
        grouped.setdefault(part.tile_index, []).append(part)
    worker_count = parallel_tile_worker_count(grouped)
    if worker_count <= 1:
        for tile_index, tile_parts in grouped.items():
            for part in tile_parts:
                consumed = image.decode_tile_payload_stream(
                    tile_index,
                    part.payload,
                    part.coding_params,
                    packet_headers=part.packet_headers,
                )
                if consumed.body != len(part.payload):
                    raise JpegParseError("JPX tile-part payload was not consumed exactly")
        return
    config = worker_config(image)
    jobs = [(config, tile_index, tile_parts) for tile_index, tile_parts in grouped.items()]
    # Materialize all worker results before mutating the parent image so a late
    # failure cannot leave a partially filled decoded_tile_data / tiles state.
    with InterpreterPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_decode_jpx_tile_parts_interpreter_job, jobs))
    for tile_index, channels, tile_data, component_planes, coding_params in results:
        image.decoded_tile_data[tile_index] = (channels, tile_data)
        install_decoded_tile_component_planes(
            image,
            tile_index,
            component_planes,
            coding_params,
        )


def decode_tile_parts_with_ppm(
    image: Any,
    parts: list[JpxTilePart],
    packet_headers: bytes,
) -> None:
    header_offset = 0
    for part in parts:
        consumed = image.decode_tile_payload_stream(
            part.tile_index,
            part.payload,
            part.coding_params,
            packet_headers=packet_headers,
            packet_header_offset=header_offset,
        )
        header_offset += consumed.header
        if consumed.body != len(part.payload):
            raise JpegParseError("JPX tile-part payload was not consumed exactly")


def parallel_tile_worker_count(grouped: dict[int, list[JpxTilePart]]) -> int:
    if len(grouped) <= 1:
        return 1
    total_payload = sum(len(part.payload) for tile_parts in grouped.values() for part in tile_parts)
    if total_payload < JPX_PARALLEL_TILE_MIN_BYTES:
        return 1
    configured = os.environ.get(JPX_WORKER_ENV)
    if configured is not None:
        try:
            return max(1, min(len(grouped), int(configured)))
        except ValueError:
            return 1
    return max(1, min(len(grouped), os.cpu_count() or 1, JPX_PARALLEL_TILE_MAX_WORKERS))


def worker_config(image: Any) -> dict[str, Any]:
    return {
        "width": image.width,
        "height": image.height,
        "components": image.components,
        "x_origin": image.x_origin,
        "y_origin": image.y_origin,
        "x_end": image.x_end,
        "y_end": image.y_end,
        "tile_width": image.tile_width,
        "tile_height": image.tile_height,
        "tile_x_origin": image.tile_x_origin,
        "tile_y_origin": image.tile_y_origin,
        "tiles_cols": image.tiles_cols,
        "tiles_rows": image.tiles_rows,
        "levels": image.levels,
        "codeblock_w": image.codeblock_w,
        "codeblock_h": image.codeblock_h,
        "codeblock_style": image.codeblock_style,
        "prog_order": image.prog_order,
        "num_layers": image.num_layers,
        "capabilities": image.capabilities,
        "packet_uses_sop": image.packet_uses_sop,
        "packet_uses_eph": image.packet_uses_eph,
        "multiple_component_transform": image.multiple_component_transform,
        "precincts": image.precincts,
        "component_coding_params": image.component_coding_params,
        "progression_changes": image.progression_changes,
        "roi_shift_by_component": image.roi_shift_by_component,
        "quant_guard_bits": image.quant_guard_bits,
        "quant_guard_bits_by_component": image.quant_guard_bits_by_component,
        "quant_style": image.quant_style,
        "quant_style_by_component": image.quant_style_by_component,
        "quant_steps": image.quant_steps,
        "packed_packet_headers": image.packed_packet_headers,
        "components_data": image.components_data,
        "reversible": image.reversible,
    }


def new_tile(
    image: Any,
    tx: int,
    ty: int,
    coding_params: JpxCodingParams | None = None,
) -> dict[Any, Any]:
    params = coding_params if coding_params is not None else image.coding_params()
    validate_jpx_coding_params(params)
    tile_w, tile_h = tile_dimensions(image, tx, ty)
    ref_x0, ref_y0, ignored_ref_x1, ignored_ref_y1 = tile_reference_bounds(
        image,
        tx,
        ty,
    )
    tile: dict[Any, Any] = {
        "x": tx,
        "y": ty,
        "ref_x0": ref_x0,
        "ref_y0": ref_y0,
        "width": tile_w,
        "height": tile_h,
        "coding_params": params,
        "components": [],
    }
    for comp_index in range(image.components):
        component_params = coding_component_style_params(params, comp_index)
        comp_x0, comp_y0, comp_x1, comp_y1 = component_tile_bounds(
            image,
            tx,
            ty,
            comp_index,
        )
        component = TileComponent(
            max(0, comp_x1 - comp_x0),
            max(0, comp_y1 - comp_y0),
            component_params.levels,
            component_params.codeblock_w,
            component_params.codeblock_h,
            x0=comp_x0,
            y0=comp_y0,
            h_sep=component_h_sep(image.components_data, comp_index),
            v_sep=component_v_sep(image.components_data, comp_index),
            codeblock_style=component_params.codeblock_style,
        )
        configure_component_precincts(image, component, component_params)
        tile["components"].append(component)
    return tile


def configure_component_precincts(
    image: Any,
    component: TileComponent,
    coding_params: JpxComponentCodingParams | JpxCodingParams | None = None,
) -> None:
    params = coding_params if coding_params is not None else image.coding_params()
    if not params.precincts:
        return
    for res_index, resolution in enumerate(component.resolutions):
        precinct_w, precinct_h = params.precincts[min(res_index, len(params.precincts) - 1)]
        grid_x0 = (resolution.x0 // precinct_w) * precinct_w
        grid_y0 = (resolution.y0 // precinct_h) * precinct_h
        grid_x1 = ceil_div(resolution.x1, precinct_w) * precinct_w
        grid_y1 = ceil_div(resolution.y1, precinct_h) * precinct_h
        precinct_cols = 0 if resolution.x0 == resolution.x1 else (grid_x1 - grid_x0) // precinct_w
        precinct_rows = 0 if resolution.y0 == resolution.y1 else (grid_y1 - grid_y0) // precinct_h
        for subband in resolution.subbands:
            subband_precinct_w = precinct_w if subband.is_ll else max(1, precinct_w // 2)
            subband_precinct_h = precinct_h if subband.is_ll else max(1, precinct_h // 2)
            configure_subband_codeblocks(
                subband,
                min(params.codeblock_w, subband_precinct_w),
                min(params.codeblock_h, subband_precinct_h),
            )
            subband_grid_x0 = grid_x0 if subband.is_ll else ceil_div(grid_x0, 2)
            subband_grid_y0 = grid_y0 if subband.is_ll else ceil_div(grid_y0, 2)
            subband.precincts = build_subband_precincts(
                subband,
                subband_precinct_w,
                subband_precinct_h,
                precinct_grid_x0=subband_grid_x0,
                precinct_grid_y0=subband_grid_y0,
                precinct_cols=precinct_cols,
                precinct_rows=precinct_rows,
            )


def configure_subband_codeblocks(
    subband: SubBand,
    codeblock_w: int,
    codeblock_h: int,
) -> None:
    if codeblock_w <= 0 or codeblock_h <= 0:
        raise ValueError("invalid JPX code-block size")
    if subband.codeblock_w == codeblock_w and subband.codeblock_h == codeblock_h:
        return
    subband.codeblock_w = codeblock_w
    subband.codeblock_h = codeblock_h
    subband.block_origin_x = subband.x0 // codeblock_w
    subband.block_origin_y = subband.y0 // codeblock_h
    subband.num_blocks_h = (
        0 if subband.width <= 0 else ceil_div(subband.x1, codeblock_w) - subband.block_origin_x
    )
    subband.num_blocks_v = (
        0 if subband.height <= 0 else ceil_div(subband.y1, codeblock_h) - subband.block_origin_y
    )
    subband.blocks = [
        CodeBlock(codeblock_w * codeblock_h)
        for ignored in range(subband.num_blocks_h * subband.num_blocks_v)
    ]


def initialize_tile_slots(image: Any) -> None:
    image.tiles = [{} for ignored in range(image.tiles_rows * image.tiles_cols)]
    image.decoded_tile_data = [None] * len(image.tiles)


def initialize_tiles(image: Any) -> None:
    image.tiles = []
    params = image.coding_params()
    for ty in range(image.tiles_rows):
        for tx in range(image.tiles_cols):
            image.tiles.append(new_tile(image, tx, ty, params))
    image.decoded_tile_data = [None] * len(image.tiles)


def ensure_tile(
    image: Any,
    tile_index: int,
    coding_params: JpxCodingParams | None = None,
) -> dict[Any, Any]:
    if not image.tiles:
        initialize_tile_slots(image)
    if tile_index < 0 or tile_index >= len(image.tiles):
        raise JpegParseError("JPX tile index out of range")
    tile = image.tiles[tile_index]
    if tile:
        return tile
    tx = tile_index % image.tiles_cols
    ty = tile_index // image.tiles_cols
    tile = new_tile(image, tx, ty, coding_params)
    image.tiles[tile_index] = tile
    return tile


def image_x_end(image: Any) -> int:
    return image.x_end if image.x_end > image.x_origin else image.x_origin + image.width


def image_y_end(image: Any) -> int:
    return image.y_end if image.y_end > image.y_origin else image.y_origin + image.height


def tile_reference_bounds(image: Any, tx: int, ty: int) -> tuple[int, int, int, int]:
    x0 = max(image.x_origin, image.tile_x_origin + tx * image.tile_width)
    y0 = max(image.y_origin, image.tile_y_origin + ty * image.tile_height)
    x1 = min(image_x_end(image), image.tile_x_origin + (tx + 1) * image.tile_width)
    y1 = min(image_y_end(image), image.tile_y_origin + (ty + 1) * image.tile_height)
    return x0, y0, x1, y1


def tile_dimensions(image: Any, tx: int, ty: int) -> tuple[int, int]:
    x0, y0, x1, y1 = tile_reference_bounds(image, tx, ty)
    return max(0, x1 - x0), max(0, y1 - y0)


def component_tile_bounds(
    image: Any,
    tx: int,
    ty: int,
    comp_index: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = tile_reference_bounds(image, tx, ty)
    h_sep = component_h_sep(image.components_data, comp_index)
    v_sep = component_v_sep(image.components_data, comp_index)
    return (
        ceil_div(x0, h_sep),
        ceil_div(y0, v_sep),
        ceil_div(x1, h_sep),
        ceil_div(y1, v_sep),
    )


def install_decoded_tile_component_planes(
    image: Any,
    tile_index: int,
    component_planes: list[list[int | float]],
    coding_params: JpxCodingParams,
) -> None:
    """Populate parent tile component LL samples from a parallel worker result."""
    tile = ensure_tile(image, tile_index, coding_params)
    tile["coding_params"] = coding_params
    components: list[TileComponent] = tile["components"]
    if len(component_planes) != len(components):
        raise JpegParseError("JPX parallel tile component count mismatch")
    for component, samples in zip(components, component_planes, strict=True):
        if not component.resolutions:
            continue
        ll = component.resolutions[0].subbands[0]
        ll.samples = list(samples)
        ll.width = component.width
        ll.height = component.height


def _decode_jpx_tile_parts_interpreter_job(
    job: tuple[dict[str, Any], int, list[JpxTilePart]],
) -> tuple[int, int, bytes, list[list[int | float]], JpxCodingParams]:
    from core_jpeg.impl.codecs.jpx.codestream import JpxImage

    config, tile_index, tile_parts = job
    image = JpxImage()
    for name, value in config.items():
        setattr(image, name, value)
    if tile_index < 0 or tile_index >= image.tiles_cols * image.tiles_rows:
        raise JpegParseError("JPX tile index out of range")
    initialize_tile_slots(image)
    for part in tile_parts:
        consumed = image.decode_tile_payload_stream(
            tile_index,
            part.payload,
            part.coding_params,
            packet_headers=part.packet_headers,
        )
        if consumed.body != len(part.payload):
            raise JpegParseError("JPX tile-part payload was not consumed exactly")
    tile = ensure_tile(image, tile_index, tile_parts[0].coding_params)
    params = tile.get("coding_params", tile_parts[0].coding_params)
    channels, tile_data = tile_rgb_samples(
        tile,
        image.components_data,
        params.multiple_component_transform,
        params.reversible,
    )
    component_planes = [list(component_samples(component)) for component in tile["components"]]
    return tile_index, channels, tile_data, component_planes, params
