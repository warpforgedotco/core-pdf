# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_jpeg.impl.codecs.jpx.boxes import (
    Jp2ChannelDefinition,
    Jp2ColorSpecification,
    Jp2ComponentMapping,
    Jp2ImageData,
    Jp2Palette,
    parse_jp2_cielab_parameters,
)
from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError
from core_jpeg.impl.graphics.color_math import (
    adapt_d50_to_d65,
    lab_to_xyz,
    xyz_to_srgb,
)
from core_jpeg.impl.graphics.icc_profiles import (
    convert_icc_profile_samples,
    icc_profile_alt_name,
)


def convert_jp2_cmyk_to_rgb(raw: bytes) -> bytes:
    if len(raw) % 4:
        raise JpegParseError("invalid JP2 CMYK sample data")
    out = bytearray()
    inv255 = 1.0 / 255.0
    for offset in range(0, len(raw), 4):
        c, m, y, k = raw[offset : offset + 4]
        red = int(255 * (1 - c * inv255) * (1 - k * inv255))
        green = int(255 * (1 - m * inv255) * (1 - k * inv255))
        blue = int(255 * (1 - y * inv255) * (1 - k * inv255))
        out.extend(
            [
                max(0, min(255, red)),
                max(0, min(255, green)),
                max(0, min(255, blue)),
            ]
        )
    return bytes(out)


def convert_jp2_sycc_to_rgb(raw: bytes) -> bytes:
    if len(raw) % 3:
        raise JpegParseError("invalid JP2 sYCC sample data")
    out = bytearray()
    offset = 128
    for index in range(0, len(raw), 3):
        y = raw[index]
        cb = raw[index + 1] - offset
        cr = raw[index + 2] - offset
        red = y + int(1.402 * cr)
        green = y - int(0.344 * cb + 0.714 * cr)
        blue = y + int(1.772 * cb)
        out.extend(
            [
                max(0, min(255, red)),
                max(0, min(255, green)),
                max(0, min(255, blue)),
            ]
        )
    return bytes(out)


def convert_jp2_eycc_to_rgb(raw: bytes) -> bytes:
    if len(raw) % 3:
        raise JpegParseError("invalid JP2 eYCC sample data")
    out = bytearray()
    offset = 128
    for index in range(0, len(raw), 3):
        y = raw[index]
        cb = raw[index + 1] - offset
        cr = raw[index + 2] - offset
        red = int(y - 0.0000368 * cb + 1.40199 * cr + 0.5)
        green = int(1.0003 * y - 0.344125 * cb - 0.7141128 * cr + 0.5)
        blue = int(0.999823 * y + 1.77204 * cb - 0.000008 * cr + 0.5)
        out.extend(
            [
                max(0, min(255, red)),
                max(0, min(255, green)),
                max(0, min(255, blue)),
            ]
        )
    return bytes(out)


def convert_jp2_cielab_to_rgb(
    raw: bytes,
    color_specification: Jp2ColorSpecification | None = None,
) -> bytes:
    if len(raw) % 3:
        raise JpegParseError("invalid JP2 CIELab sample data")
    cielab = (
        color_specification.cielab
        if color_specification is not None and color_specification.cielab is not None
        else parse_jp2_cielab_parameters(b"")
    )
    out = bytearray()
    white_point = (0.9642, 1.0, 0.8249)
    for offset in range(0, len(raw), 3):
        l_byte, a_byte, b_byte = raw[offset : offset + 3]
        l_star = jp2_cielab_sample_to_value(
            l_byte,
            cielab.range_l,
            cielab.offset_l,
        )
        a_star = jp2_cielab_sample_to_value(
            a_byte,
            cielab.range_a,
            cielab.offset_a,
        )
        b_star = jp2_cielab_sample_to_value(
            b_byte,
            cielab.range_b,
            cielab.offset_b,
        )
        x, y, z = lab_to_xyz(l_star, a_star, b_star, white_point)
        ax, ay, az = adapt_d50_to_d65(x, y, z)
        red, green, blue = xyz_to_srgb(ax, ay, az)
        out.extend(
            [
                max(0, min(255, int(round(red * 255.0)))),
                max(0, min(255, int(round(green * 255.0)))),
                max(0, min(255, int(round(blue * 255.0)))),
            ]
        )
    return bytes(out)


def jp2_cielab_sample_to_value(sample: int, value_range: int, offset: int) -> float:
    return (float(sample) - float(offset)) * float(value_range) / 255.0


def apply_jp2_container_transforms(
    raw: bytes,
    width: int,
    height: int,
    jp2: Jp2ImageData,
) -> bytes:
    if jp2.component_mapping:
        raw = apply_jp2_component_mapping(
            raw,
            width,
            height,
            jp2.component_mapping,
            jp2.palette,
            jp2.color_specification,
            preserve_all_channels=bool(jp2.channel_definitions),
        )
    if jp2.channel_definitions:
        raw = apply_jp2_channel_definitions(
            raw,
            width,
            height,
            jp2.channel_definitions,
            jp2.color_specification,
        )
    return raw


def apply_jp2_embedded_color_transforms(
    raw: bytes,
    width: int,
    height: int,
    jp2: Jp2ImageData,
) -> bytes:
    color_space_kind = jp2_color_space_kind(jp2.color_specification)
    if jp2.color_specification is not None and jp2.color_specification.icc_profile:
        converted = convert_icc_profile_samples(
            raw,
            jp2.color_specification.icc_profile,
        )
        if converted is not None:
            raw = converted
    if color_space_kind == "CMYK" and len(raw) == width * height * 4:
        return convert_jp2_cmyk_to_rgb(raw)
    if color_space_kind == "sYCC" and len(raw) == width * height * 3:
        return convert_jp2_sycc_to_rgb(raw)
    if color_space_kind == "eYCC" and len(raw) == width * height * 3:
        return convert_jp2_eycc_to_rgb(raw)
    if color_space_kind == "CIELab" and len(raw) == width * height * 3:
        return convert_jp2_cielab_to_rgb(raw, jp2.color_specification)
    return raw


def apply_jp2_component_mapping(
    raw: bytes,
    width: int,
    height: int,
    component_mapping: tuple[Jp2ComponentMapping, ...],
    palette: Jp2Palette | None = None,
    color_specification: Jp2ColorSpecification | None = None,
    *,
    preserve_all_channels: bool = False,
) -> bytes:
    pixel_count = width * height
    if pixel_count <= 0:
        return raw
    if len(raw) % pixel_count:
        return raw
    input_channels = len(raw) // pixel_count
    if input_channels <= 0:
        return raw
    mappings = component_mapping
    if not mappings:
        if palette is None or not palette.entries:
            return raw
        column_count = len(palette.entries[0])
        mappings = tuple(
            Jp2ComponentMapping(component=0, mapping_type=1, palette_column=index)
            for index in range(
                column_count
                if preserve_all_channels
                else min(
                    jp2_container_output_channels(
                        color_specification=color_specification,
                        palette=palette,
                    ),
                    column_count,
                )
            )
        )
    output_channels = (
        len(mappings)
        if preserve_all_channels
        else min(
            jp2_container_output_channels(
                color_specification=color_specification,
                component_mapping=mappings,
                palette=palette,
            ),
            len(mappings),
        )
    )
    result = bytearray(pixel_count * output_channels)
    for pixel in range(pixel_count):
        for out_channel, mapping in enumerate(mappings[:output_channels]):
            value = jp2_mapped_component_value(
                raw,
                pixel,
                input_channels,
                palette,
                mapping,
            )
            result[pixel * output_channels + out_channel] = value
    return bytes(result)


def apply_jp2_palette(
    raw: bytes,
    width: int,
    height: int,
    palette: Jp2Palette,
    component_mapping: tuple[Jp2ComponentMapping, ...],
    color_specification: Jp2ColorSpecification | None = None,
) -> bytes:
    return apply_jp2_component_mapping(
        raw,
        width,
        height,
        component_mapping,
        palette,
        color_specification,
    )


def jp2_mapped_component_value(
    raw: bytes,
    pixel: int,
    input_channels: int,
    palette: Jp2Palette | None,
    mapping: Jp2ComponentMapping,
) -> int:
    if mapping.component >= input_channels:
        raise JpegParseError("JP2 component mapping index out of range")
    sample = raw[pixel * input_channels + mapping.component]
    if mapping.mapping_type == 0:
        return sample
    if mapping.mapping_type != 1:
        raise JpegUnsupportedError("unsupported JP2 component mapping type")
    if palette is None or not palette.entries:
        raise JpegParseError("JP2 palette mapping requires palette entries")
    if mapping.palette_column >= len(palette.entries[0]):
        raise JpegParseError("JP2 palette column out of range")
    palette_index = min(sample, len(palette.entries) - 1)
    return palette.entries[palette_index][mapping.palette_column]


def apply_jp2_channel_definitions(
    raw: bytes,
    width: int,
    height: int,
    definitions: tuple[Jp2ChannelDefinition, ...],
    color_specification: Jp2ColorSpecification | None = None,
) -> bytes:
    pixel_count = width * height
    if pixel_count <= 0 or not definitions or len(raw) % pixel_count:
        return raw
    input_channels = len(raw) // pixel_count
    if input_channels <= 0:
        return raw
    validate_jp2_channel_definitions(definitions, input_channels)
    color_mappings = jp2_color_channel_definitions(
        definitions,
        color_specification=color_specification,
    )
    opacity_mappings = jp2_opacity_channel_definitions(definitions)
    output_mappings = color_mappings + opacity_mappings
    if not output_mappings:
        return raw
    output_channels = min(
        jp2_container_output_channels(
            color_specification=color_specification,
            channel_definitions=definitions,
        ),
        len(output_mappings),
    )
    result = bytearray(pixel_count * output_channels)
    for pixel in range(pixel_count):
        for out_channel, definition in enumerate(output_mappings[:output_channels]):
            if definition.component >= input_channels:
                raise JpegParseError("JP2 channel definition component out of range")
            result[pixel * output_channels + out_channel] = raw[
                pixel * input_channels + definition.component
            ]
    return bytes(result)


def validate_jp2_channel_definitions(
    definitions: tuple[Jp2ChannelDefinition, ...],
    channel_count: int,
) -> None:
    if channel_count <= 0:
        raise JpegParseError("invalid JP2 channel count")
    seen_channels: set[int] = set()
    seen_color_associations: set[int] = set()
    for definition in definitions:
        if definition.component >= channel_count:
            raise JpegParseError("JP2 channel definition component out of range")
        if definition.component in seen_channels:
            raise JpegParseError("duplicate JP2 channel definition component")
        seen_channels.add(definition.component)
        if definition.channel_type == 0 and 0 < definition.association < 65535:
            if definition.association in seen_color_associations:
                raise JpegParseError("duplicate JP2 channel definition association")
            seen_color_associations.add(definition.association)
        if definition.association == 65535:
            continue
        if definition.association > 0 and definition.association - 1 >= channel_count:
            raise JpegParseError("JP2 channel definition association out of range")
    if len(seen_channels) < channel_count:
        raise JpegParseError("incomplete JP2 channel definitions")


def default_jp2_output_channels(
    color_specification: Jp2ColorSpecification | None,
) -> int:
    color_space_kind = jp2_color_space_kind(color_specification)
    if color_space_kind == "GRAY":
        return 1
    if color_space_kind == "CMYK":
        return 4
    return 3


def jp2_container_output_channels(
    *,
    color_specification: Jp2ColorSpecification | None,
    component_mapping: tuple[Jp2ComponentMapping, ...] = (),
    channel_definitions: tuple[Jp2ChannelDefinition, ...] = (),
    palette: Jp2Palette | None = None,
) -> int:
    color_channels = len(
        jp2_color_channel_definitions(
            channel_definitions,
            color_specification=color_specification,
        )
    )
    opacity_channels = len(jp2_opacity_channel_definitions(channel_definitions))
    if color_channels > 0:
        return color_channels + opacity_channels
    color_space_kind = jp2_color_space_kind(color_specification)
    if color_space_kind == "GRAY":
        return 1 + opacity_channels
    if color_space_kind == "CMYK":
        return 4 + opacity_channels
    if color_space_kind is not None:
        return 3 + opacity_channels
    if component_mapping:
        return len(component_mapping)
    return 3


def jp2_color_channel_definitions(
    definitions: tuple[Jp2ChannelDefinition, ...],
    *,
    color_specification: Jp2ColorSpecification | None,
) -> tuple[Jp2ChannelDefinition, ...]:
    max_association = jp2_explicit_color_channel_count(color_specification)
    filtered = sorted(
        (
            definition
            for definition in definitions
            if definition.channel_type == 0
            and definition.association > 0
            and (max_association is None or definition.association <= max_association)
        ),
        key=lambda definition: (definition.association, definition.component),
    )
    if not filtered:
        return tuple(
            sorted(
                (
                    definition
                    for definition in definitions
                    if definition.channel_type == 0 and definition.association == 0
                ),
                key=lambda definition: definition.component,
            )
        )
    deduped: list[Jp2ChannelDefinition] = []
    seen_associations: set[int] = set()
    for definition in filtered:
        if definition.association in seen_associations:
            continue
        seen_associations.add(definition.association)
        deduped.append(definition)
    return tuple(deduped)


def jp2_opacity_channel_definitions(
    definitions: tuple[Jp2ChannelDefinition, ...],
) -> tuple[Jp2ChannelDefinition, ...]:
    return tuple(
        definition
        for definition in sorted(definitions, key=lambda item: item.component)
        if definition.channel_type in {1, 2}
    )


def jp2_explicit_color_channel_count(
    color_specification: Jp2ColorSpecification | None,
) -> int | None:
    color_space_kind = jp2_color_space_kind(color_specification)
    if color_space_kind == "GRAY":
        return 1
    if color_space_kind == "CMYK":
        return 4
    if color_space_kind in {"RGB", "sYCC", "eYCC", "CIELab"}:
        return 3
    return None


def jp2_color_space_kind(
    color_specification: Jp2ColorSpecification | None,
) -> str | None:
    if color_specification is None:
        return None
    enum_color_space = color_specification.enum_color_space
    if enum_color_space == 12:
        return "CMYK"
    if enum_color_space == 14:
        return "CIELab"
    if enum_color_space == 16:
        return "RGB"
    if enum_color_space == 17:
        return "GRAY"
    if enum_color_space == 18:
        return "sYCC"
    if enum_color_space == 24:
        return "eYCC"
    profile = color_specification.icc_profile
    if profile is None:
        return None
    kind = jp2_color_space_kind_from_name(icc_profile_alt_name(profile, 3))
    if kind is not None:
        return kind
    if len(profile) >= 20 and profile[16:20] in {b"Lab ", b"LAB "}:
        return "CIELab"
    return None


def jp2_color_space_kind_from_name(name: str | None) -> str | None:
    if name == "DeviceGray":
        return "GRAY"
    if name == "DeviceRGB":
        return "RGB"
    if name == "DeviceCMYK":
        return "CMYK"
    if name == "Lab":
        return "CIELab"
    return None


def jp2_color_specification_is_better(
    candidate: Jp2ColorSpecification | None,
    current: Jp2ColorSpecification | None,
) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return jp2_color_specification_rank(candidate) < jp2_color_specification_rank(current)


def jp2_color_specification_rank(
    specification: Jp2ColorSpecification,
) -> tuple[int, int, int, int]:
    return (
        0 if specification.method in {1, 2, 3} else 1,
        specification.precedence,
        specification.approximation,
        0 if specification.method == 1 else 1 if specification.method == 2 else 2,
    )


def jp2_requires_all_components(jp2: Jp2ImageData) -> bool:
    if (
        jp2_container_output_channels(
            color_specification=jp2.color_specification,
            component_mapping=jp2.component_mapping,
            channel_definitions=jp2.channel_definitions,
        )
        > 3
    ):
        return True
    if jp2.component_mapping:
        return True
    return bool(jp2.channel_definitions)
