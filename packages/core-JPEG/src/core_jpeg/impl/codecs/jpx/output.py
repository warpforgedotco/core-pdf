# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any

from core_jpeg.impl.codecs.jpx.color import (
    jp2_color_space_kind,
    jp2_requires_all_components,
)
from core_jpeg.impl.codecs.jpx.structures import ceil_div
from core_jpeg.impl.codecs.jpx.wavelet import inverse_mct
from core_jpeg.impl.errors import JpegParseError
from core_jpeg.impl.models import DecodedJpxComponent, DecodedJpxImage


def clamp_sample(value: int, precision: int) -> int:
    max_value = (1 << precision) - 1
    if value < 0:
        value = 0
    elif value > max_value:
        value = max_value
    if precision == 8:
        return value
    if precision < 8:
        return value << (8 - precision)
    return value >> (precision - 8)


def output_sample_byte(
    sample: int | float,
    *,
    precision: int,
    is_signed: bool,
) -> int:
    value = int(round(sample))
    if not is_signed:
        value += 1 << (precision - 1)
    return clamp_sample(value, precision)


def component_samples(component: Any) -> list[int | float]:
    if not component.resolutions:
        return []
    subband = component.resolutions[0].subbands[0]
    return list(subband.samples)


def tile_rgb_samples(
    tile: dict[Any, Any],
    component_specs: list[dict[Any, Any]],
    multiple_component_transform: int,
    reversible: bool,
) -> tuple[int, bytes]:
    return tile_interleaved_samples(
        tile,
        component_specs,
        multiple_component_transform,
        reversible,
        component_mode="default",
    )


def tile_interleaved_samples(
    tile: dict[Any, Any],
    component_specs: list[dict[Any, Any]],
    multiple_component_transform: int,
    reversible: bool,
    *,
    component_mode: str = "default",
) -> tuple[int, bytes]:
    components: list[Any] = tile["components"]
    if not components:
        return 0, b""
    component_planes = [component_samples(component) for component in components]
    if component_mode == "all":
        if multiple_component_transform and len(components) >= 3:
            planes = list(inverse_mct(component_planes[:3], reversible)) + component_planes[3:]
            plane_components = [
                components[0],
                components[0],
                components[0],
                *components[3:],
            ]
        else:
            planes = component_planes
            plane_components = components
        channels = len(components)
    elif len(components) >= 3:
        planes = (
            list(inverse_mct(component_planes[:3], reversible))
            if multiple_component_transform
            else component_planes[:3]
        )
        plane_components = (
            [components[0], components[1], components[2]]
            if not multiple_component_transform
            else [components[0], components[0], components[0]]
        )
        channels = 3
    else:
        planes = [component_planes[0]]
        plane_components = [components[0]]
        channels = 1
    tile_width = int(tile.get("width") or components[0].width)
    tile_height = int(tile.get("height") or components[0].height)
    tile_x0 = int(tile.get("ref_x0", components[0].x0 * components[0].h_sep))
    tile_y0 = int(tile.get("ref_y0", components[0].y0 * components[0].v_sep))
    pixel_count = tile_width * tile_height
    out = bytearray(pixel_count * channels)
    for y in range(tile_height):
        ref_y = tile_y0 + y
        for x in range(tile_width):
            ref_x = tile_x0 + x
            pixel_index = y * tile_width + x
            for c in range(channels):
                precision = component_precision(component_specs, c)
                sample = component_sample_at(
                    plane_components[c],
                    planes[c],
                    ref_x,
                    ref_y,
                )
                out[pixel_index * channels + c] = output_sample_byte(
                    sample,
                    precision=precision,
                    is_signed=component_is_signed(component_specs, c),
                )
    return channels, bytes(out)


def component_sample_at(
    component: Any,
    samples: list[int | float],
    ref_x: int,
    ref_y: int,
) -> int | float:
    if component.width <= 0 or component.height <= 0 or not samples:
        return 0
    sample_x = ref_x // component.h_sep
    sample_y = ref_y // component.v_sep
    start_x = component.x0 * component.h_sep
    start_y = component.y0 * component.v_sep
    if sample_x < component.x0 and ref_x < start_x:
        return 0
    if sample_y < component.y0 and ref_y < start_y:
        return 0
    local_x = min(max(sample_x, component.x0), component.x0 + component.width - 1)
    local_y = min(max(sample_y, component.y0), component.y0 + component.height - 1)
    index = (local_y - component.y0) * component.width + (local_x - component.x0)
    return samples[index] if 0 <= index < len(samples) else 0


def component_h_sep(component_specs: list[dict[Any, Any]], index: int) -> int:
    if not component_specs:
        return 1
    spec = component_specs[min(index, len(component_specs) - 1)]
    try:
        return max(1, int(spec.get("h_sep", 1)))
    except TypeError:
        return 1


def component_v_sep(component_specs: list[dict[Any, Any]], index: int) -> int:
    if not component_specs:
        return 1
    spec = component_specs[min(index, len(component_specs) - 1)]
    try:
        return max(1, int(spec.get("v_sep", 1)))
    except TypeError:
        return 1


def component_precision(component_specs: list[dict[Any, Any]], index: int) -> int:
    if not component_specs:
        return 8
    spec = component_specs[min(index, len(component_specs) - 1)]
    try:
        precision = int(spec.get("precision", 8))
    except TypeError:
        return 8
    return max(1, precision)


def component_is_signed(component_specs: list[dict[Any, Any]], index: int) -> bool:
    if not component_specs:
        return False
    spec = component_specs[min(index, len(component_specs) - 1)]
    return bool(spec.get("is_signed"))


def place_tile_bytes(
    image: bytearray,
    image_width: int,
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
    channels: int,
    tile_data: bytes,
) -> None:
    row_bytes = tile_width * channels
    expected = row_bytes * tile_height
    if len(tile_data) < expected:
        raise JpegParseError("JPX tile produced too few samples")
    for row in range(tile_height):
        dst = ((tile_y + row) * image_width + tile_x) * channels
        src = row * row_bytes
        image[dst : dst + row_bytes] = tile_data[src : src + row_bytes]


def jp2_decode_component_mode(
    jp2: Any,
    *,
    apply_embedded_color: bool,
) -> str:
    if not apply_embedded_color:
        # OpenJPEG component output preserves codestream components when color
        # conversion is disabled; PDF callers with explicit color spaces need
        # the same component parity instead of display-oriented RGB truncation.
        return "all"
    if jp2_requires_all_components(jp2):
        return "all"
    return "default"


def jp2_preserves_native_component_output(
    img: Any,
    jp2: Any,
    *,
    apply_embedded_color: bool,
) -> bool:
    return not apply_embedded_color and not jp2.component_mapping and not jp2.channel_definitions


def decoded_jpx_image_from_interleaved(
    raw: bytes,
    img: Any,
    jp2: Any,
    *,
    component_mode: str,
    components: tuple[DecodedJpxComponent, ...] | None = None,
) -> DecodedJpxImage:
    pixel_count = img.width * img.height
    if pixel_count <= 0:
        raise JpegParseError("invalid JPX decoded image dimensions")
    if len(raw) % pixel_count:
        raise JpegParseError("invalid JPX decoded sample count")
    channel_count = len(raw) // pixel_count if pixel_count else 0
    decoded_components: list[DecodedJpxComponent] = []
    if components is not None:
        decoded_components = list(components)
    elif channel_count:
        for channel in range(channel_count):
            spec_index = (
                channel
                if component_mode == "all" and channel < len(img.components_data)
                else min(channel, max(0, len(img.components_data) - 1))
            )
            decoded_components.append(
                DecodedJpxComponent(
                    index=channel,
                    width=img.width,
                    height=img.height,
                    precision=component_precision(img.components_data, spec_index),
                    is_signed=component_is_signed(img.components_data, spec_index),
                    data=raw[channel::channel_count],
                )
            )
    image_width = img.width
    image_height = img.height
    if decoded_components:
        image_width = max(component.width for component in decoded_components)
        image_height = max(component.height for component in decoded_components)
    return DecodedJpxImage(
        width=image_width,
        height=image_height,
        color_space=jp2_color_space_kind(jp2.color_specification),
        components=tuple(decoded_components),
        interleaved=raw,
    )


def decoded_jpx_native_components(img: Any) -> tuple[DecodedJpxComponent, ...]:
    output_count = len(img.components_data)
    if output_count <= 0:
        return ()
    reference_indices = decoded_jpx_native_component_reference_indices(img)
    outputs: list[bytearray] = []
    output_bounds: list[tuple[int, int, int, int]] = []
    for component_index, reference_index in enumerate(reference_indices):
        bounds = image_component_bounds(img, reference_index)
        output_bounds.append(bounds)
        x0, y0, x1, y1 = bounds
        outputs.append(bytearray(max(0, x1 - x0) * max(0, y1 - y0)))
    for tile_index, tile in enumerate(img.tiles):
        if not tile:
            tile = img.ensure_tile(tile_index, img.coding_params())
        tile_components: list[Any] = tile["components"]
        if not tile_components:
            continue
        params = tile.get("coding_params", img.coding_params())
        planes, plane_components = tile_native_component_planes(
            tile_components,
            params.multiple_component_transform,
            params.reversible,
        )
        for component_index, output in enumerate(outputs):
            if component_index >= len(planes):
                break
            precision = component_precision(img.components_data, component_index)
            is_signed = component_is_signed(img.components_data, component_index)
            place_native_component_samples(
                output,
                output_bounds[component_index],
                plane_components[component_index],
                planes[component_index],
                precision=precision,
                is_signed=is_signed,
            )
    return tuple(
        DecodedJpxComponent(
            index=component_index,
            width=max(0, bounds[2] - bounds[0]),
            height=max(0, bounds[3] - bounds[1]),
            precision=component_precision(img.components_data, component_index),
            is_signed=component_is_signed(img.components_data, component_index),
            data=bytes(output),
        )
        for component_index, (bounds, output) in enumerate(zip(output_bounds, outputs))
    )


def decoded_jpx_native_component_reference_indices(img: Any) -> list[int]:
    component_count = len(img.components_data)
    if component_count >= 3 and decoded_jpx_native_uses_mct(img):
        return [0, 0, 0, *range(3, component_count)]
    return list(range(component_count))


def decoded_jpx_native_uses_mct(img: Any) -> bool:
    for tile in img.tiles:
        if tile:
            params = tile.get("coding_params", img.coding_params())
            if params.multiple_component_transform:
                return True
    return bool(img.coding_params().multiple_component_transform)


def image_component_bounds(
    img: Any,
    component_index: int,
) -> tuple[int, int, int, int]:
    h_sep = component_h_sep(img.components_data, component_index)
    v_sep = component_v_sep(img.components_data, component_index)
    return (
        ceil_div(img.x_origin, h_sep),
        ceil_div(img.y_origin, v_sep),
        ceil_div(img.image_x_end(), h_sep),
        ceil_div(img.image_y_end(), v_sep),
    )


def tile_native_component_planes(
    components: list[Any],
    multiple_component_transform: int,
    reversible: bool,
) -> tuple[list[list[int | float]], list[Any]]:
    component_planes = [component_samples(component) for component in components]
    if multiple_component_transform and len(components) >= 3:
        return (
            list(inverse_mct(component_planes[:3], reversible)) + component_planes[3:],
            [
                components[0],
                components[0],
                components[0],
                *components[3:],
            ],
        )
    return component_planes, components


def place_native_component_samples(
    output: bytearray,
    output_bounds: tuple[int, int, int, int],
    component: Any,
    samples: list[int | float],
    *,
    precision: int,
    is_signed: bool,
) -> None:
    x0, y0, x1, y1 = output_bounds
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if width <= 0 or height <= 0 or not samples:
        return
    for y in range(component.height):
        dst_y = component.y0 - y0 + y
        if dst_y < 0 or dst_y >= height:
            continue
        src = y * component.width
        dst_row = dst_y * width
        for x in range(component.width):
            dst_x = component.x0 - x0 + x
            src_index = src + x
            if dst_x < 0 or dst_x >= width or src_index >= len(samples):
                continue
            output[dst_row + dst_x] = output_sample_byte(
                samples[src_index],
                precision=precision,
                is_signed=is_signed,
            )
