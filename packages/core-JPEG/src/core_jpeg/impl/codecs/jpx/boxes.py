# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from core_jpeg.impl.errors import JpegParseError, JpegUnsupportedError
from core_jpeg.impl.models import Jp2Resolution


@dataclass(frozen=True)
class Jp2ComponentMapping:
    component: int
    mapping_type: int
    palette_column: int


@dataclass(frozen=True)
class Jp2Palette:
    entries: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Jp2ChannelDefinition:
    component: int
    channel_type: int
    association: int


@dataclass(frozen=True)
class Jp2CieLabParameters:
    range_l: int
    offset_l: int
    range_a: int
    offset_a: int
    range_b: int
    offset_b: int
    illuminant: int


@dataclass(frozen=True)
class Jp2ColorSpecification:
    method: int
    precedence: int
    approximation: int
    enum_color_space: int | None = None
    icc_profile: bytes | None = None
    cielab: Jp2CieLabParameters | None = None


@dataclass(frozen=True)
class Jp2ImageData:
    codestream: bytes
    brand: bytes | None = None
    min_version: int | None = None
    compatibility: tuple[bytes, ...] = ()
    width: int | None = None
    height: int | None = None
    components: int | None = None
    bits_per_component_default: int | None = None
    bits_per_component: tuple[int, ...] = ()
    palette: Jp2Palette | None = None
    component_mapping: tuple[Jp2ComponentMapping, ...] = ()
    channel_definitions: tuple[Jp2ChannelDefinition, ...] = ()
    color_specification: Jp2ColorSpecification | None = None
    color_specifications: tuple[Jp2ColorSpecification, ...] = ()
    capture_resolution: Jp2Resolution | None = None
    display_resolution: Jp2Resolution | None = None


class Jp2Parser:
    __slots__ = (
        "data",
        "brand",
        "min_version",
        "compatibility",
        "width",
        "height",
        "components",
        "bits_per_component_default",
        "bits_per_component",
        "palette",
        "component_mapping",
        "channel_definitions",
        "color_specification",
        "color_specifications",
        "capture_resolution",
        "display_resolution",
        "codestream",
    )

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.brand: bytes | None = None
        self.min_version: int | None = None
        self.compatibility: tuple[bytes, ...] = ()
        self.width: int | None = None
        self.height: int | None = None
        self.components: int | None = None
        self.bits_per_component_default: int | None = None
        self.bits_per_component: tuple[int, ...] = ()
        self.palette: Jp2Palette | None = None
        self.component_mapping: tuple[Jp2ComponentMapping, ...] = ()
        self.channel_definitions: tuple[Jp2ChannelDefinition, ...] = ()
        self.color_specification: Jp2ColorSpecification | None = None
        self.color_specifications: tuple[Jp2ColorSpecification, ...] = ()
        self.capture_resolution: Jp2Resolution | None = None
        self.display_resolution: Jp2Resolution | None = None
        self.codestream = data

    def parse(self) -> Jp2ImageData:
        if self.data.startswith(b"\xff\x4f"):
            return self.image_data()
        if not self.data:
            return self.image_data()
        if len(self.data) < 12:
            raise JpegParseError("truncated JP2 signature box")
        seen_signature = False
        seen_file_type = False
        seen_header = False
        seen_codestream = False
        for box_index, (box_type, payload_start, payload_end) in enumerate(
            self.iter_boxes(0, len(self.data)),
        ):
            if box_type == b"jP  ":
                if seen_signature:
                    raise JpegParseError("duplicate JP2 signature box")
                if box_index != 0:
                    raise JpegParseError("JP2 signature box must be first")
                self.parse_signature_box(self.data[payload_start:payload_end])
                seen_signature = True
            elif box_type == b"ftyp":
                if seen_file_type:
                    raise JpegParseError("duplicate JP2 file type box")
                if not seen_signature:
                    raise JpegParseError("JP2 signature box must be first")
                if box_index != 1:
                    raise JpegParseError("JP2 file type box must be second")
                self.parse_file_type_box(self.data[payload_start:payload_end])
                seen_file_type = True
            elif not seen_signature:
                raise JpegParseError("JP2 signature box must be first")
            elif not seen_file_type:
                raise JpegParseError("JP2 file type box must be second")
            elif box_type == b"jp2h":
                if seen_header:
                    raise JpegParseError("duplicate JP2 header box")
                if seen_codestream:
                    raise JpegParseError("JP2 header box after codestream box")
                if not seen_file_type:
                    raise JpegParseError("JP2 header box before file type box")
                self.parse_header_box(payload_start, payload_end)
                seen_header = True
            elif box_type == b"jp2c":
                if seen_codestream:
                    raise JpegParseError("duplicate JP2 codestream box")
                if not seen_header:
                    raise JpegParseError("JP2 codestream box before header box")
                self.codestream = self.data[payload_start:payload_end]
                seen_codestream = True
        if not seen_signature:
            raise JpegParseError("JP2 file missing signature box")
        if not seen_file_type:
            raise JpegParseError("JP2 file missing file type box")
        if not seen_header:
            raise JpegParseError("JP2 file missing header box")
        if not seen_codestream:
            raise JpegParseError("JP2 file missing codestream box")
        return self.image_data()

    def image_data(self) -> Jp2ImageData:
        return Jp2ImageData(
            codestream=self.codestream,
            brand=self.brand,
            min_version=self.min_version,
            compatibility=self.compatibility,
            width=self.width,
            height=self.height,
            components=self.components,
            bits_per_component_default=self.bits_per_component_default,
            bits_per_component=self.bits_per_component,
            palette=self.palette,
            component_mapping=self.component_mapping,
            channel_definitions=self.channel_definitions,
            color_specification=self.color_specification,
            color_specifications=self.color_specifications,
            capture_resolution=self.capture_resolution,
            display_resolution=self.display_resolution,
        )

    @staticmethod
    def parse_signature_box(payload: bytes) -> None:
        if payload != b"\r\n\x87\n":
            raise JpegParseError("bad JP2 signature box")

    def parse_file_type_box(self, payload: bytes) -> None:
        if len(payload) < 8 or len(payload) % 4:
            raise JpegParseError("bad JP2 file type box")
        self.brand = payload[:4]
        self.min_version = int.from_bytes(payload[4:8], "big")
        self.compatibility = tuple(
            payload[offset : offset + 4] for offset in range(8, len(payload), 4)
        )
        if self.brand != b"jp2 " and b"jp2 " not in self.compatibility:
            raise JpegUnsupportedError("file type is not JP2 compatible")

    def iter_boxes(self, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
        offset = start
        while offset + 8 <= end:
            box_start = offset
            length = int.from_bytes(self.data[offset : offset + 4], "big")
            box_type = self.data[offset + 4 : offset + 8]
            offset += 8
            if length == 1:
                if offset + 8 > end:
                    raise JpegParseError("truncated JP2 extended box length")
                length = int.from_bytes(self.data[offset : offset + 8], "big")
                offset += 8
                header_length = 16
            else:
                header_length = 8
            if length == 0:
                box_end = end
            else:
                if length < header_length:
                    raise JpegParseError("invalid JP2 box length")
                box_end = box_start + length
            if box_end > end:
                raise JpegParseError("truncated JP2 box")
            yield box_type, offset, box_end
            offset = box_end
        if offset != end:
            raise JpegParseError("truncated JP2 box header")

    def parse_header_box(self, start: int, end: int) -> None:
        seen: set[bytes] = set()
        color_specifications: list[Jp2ColorSpecification] = []
        for child_index, (box_type, payload_start, payload_end) in enumerate(
            self.iter_boxes(start, end),
        ):
            payload = self.data[payload_start:payload_end]
            if box_type == b"ihdr":
                if child_index != 0:
                    raise JpegParseError("JP2 image header box must be first in JP2 header")
                if box_type in seen:
                    raise JpegParseError("duplicate JP2 image header box")
                self.parse_image_header(payload)
            elif box_type == b"bpcc":
                if box_type in seen:
                    raise JpegParseError("duplicate JP2 bits-per-component box")
                self.bits_per_component = self.parse_bits_per_component(payload)
            elif box_type == b"pclr":
                if self.palette is not None:
                    raise JpegParseError("duplicate JP2 palette box")
                self.palette = parse_jp2_palette(payload)
            elif box_type == b"cmap":
                if self.palette is None:
                    raise JpegParseError("JP2 component mapping requires preceding palette box")
                if self.component_mapping:
                    raise JpegParseError("duplicate JP2 component mapping box")
                self.component_mapping = parse_jp2_component_mapping(
                    payload,
                    len(self.palette.entries[0]) if self.palette.entries else 0,
                )
            elif box_type == b"cdef":
                if self.channel_definitions:
                    raise JpegParseError("duplicate JP2 channel definition box")
                self.channel_definitions = parse_jp2_channel_definitions(payload)
            elif box_type == b"colr":
                color_specifications.append(self.parse_color_specification(payload))
            elif box_type == b"res ":
                if box_type in seen:
                    raise JpegParseError("duplicate JP2 resolution box")
                self.parse_resolution_box(payload_start, payload_end)
            seen.add(box_type)
        if b"ihdr" not in seen:
            raise JpegParseError("JP2 header box missing image header")
        if not color_specifications:
            raise JpegParseError("JP2 header box missing color specification")
        if self.bits_per_component_default == 255 and b"bpcc" not in seen:
            raise JpegParseError("JP2 variable component precision requires BPCC box")
        if self.bits_per_component_default != 255 and b"bpcc" in seen:
            raise JpegParseError("JP2 BPCC box requires variable component precision")
        if (self.palette is None) != (not self.component_mapping):
            raise JpegParseError("JP2 palette and component mapping boxes must occur together")
        if self.components is not None:
            for mapping in self.component_mapping:
                if mapping.component >= self.components:
                    raise JpegParseError("JP2 component mapping index out of range")
        self.color_specifications = tuple(color_specifications)
        self.color_specification = min(
            color_specifications,
            key=lambda specification: (
                specification.precedence,
                specification.approximation,
            ),
        )

    def parse_image_header(self, payload: bytes) -> None:
        if len(payload) != 14:
            raise JpegParseError("bad JP2 image header box size")
        height = int.from_bytes(payload[0:4], "big")
        width = int.from_bytes(payload[4:8], "big")
        components = int.from_bytes(payload[8:10], "big")
        bits_per_component = payload[10]
        if width < 1 or height < 1 or components < 1:
            raise JpegParseError("invalid JP2 image header values")
        if components > 16384:
            raise JpegParseError("too many JP2 components")
        if bits_per_component != 255 and (bits_per_component & 0x7F) >= 38:
            raise JpegParseError("invalid JP2 bits-per-component value")
        self.height = height
        self.width = width
        self.components = components
        self.bits_per_component_default = bits_per_component

    def parse_bits_per_component(self, payload: bytes) -> tuple[int, ...]:
        if self.components is None:
            raise JpegParseError("JP2 BPCC box requires image header")
        if len(payload) != self.components:
            raise JpegParseError("bad JP2 bits-per-component box size")
        if any((value & 0x7F) >= 38 for value in payload):
            raise JpegParseError("invalid JP2 bits-per-component value")
        return tuple(payload)

    @staticmethod
    def parse_color_specification(payload: bytes) -> Jp2ColorSpecification:
        if len(payload) < 3:
            raise JpegParseError("truncated JP2 color specification box")
        method = payload[0]
        precedence = payload[1]
        approximation = payload[2]
        if method == 1:
            if len(payload) < 7:
                raise JpegParseError("truncated JP2 enumerated color space")
            enum_color_space = int.from_bytes(payload[3:7], "big")
            if enum_color_space == 14:
                if len(payload) not in {7, 35}:
                    raise JpegParseError("bad JP2 CIELab color specification size")
            elif len(payload) != 7:
                raise JpegParseError("bad JP2 enumerated color space size")
            return Jp2ColorSpecification(
                method=method,
                precedence=precedence,
                approximation=approximation,
                enum_color_space=enum_color_space,
                cielab=parse_jp2_cielab_parameters(payload) if enum_color_space == 14 else None,
            )
        if method == 2:
            validate_restricted_icc_profile(payload[3:])
            return Jp2ColorSpecification(
                method=method,
                precedence=precedence,
                approximation=approximation,
                icc_profile=payload[3:],
            )
        if method == 3:
            raise JpegUnsupportedError("JP2 unrestricted ICC profiles are not supported")
        raise JpegUnsupportedError(f"unsupported JP2 color specification method {method}")

    def parse_resolution_box(self, start: int, end: int) -> None:
        seen: set[bytes] = set()
        for box_type, payload_start, payload_end in self.iter_boxes(start, end):
            if box_type not in {b"resc", b"resd"}:
                continue
            if box_type in seen:
                raise JpegParseError("duplicate JP2 resolution child box")
            resolution = parse_jp2_resolution(self.data[payload_start:payload_end])
            if box_type == b"resc":
                self.capture_resolution = resolution
            else:
                self.display_resolution = resolution
            seen.add(box_type)
        if not seen:
            raise JpegParseError("empty JP2 resolution box")


def parse_jp2_cielab_parameters(payload: bytes) -> Jp2CieLabParameters:
    if len(payload) == 35:
        return Jp2CieLabParameters(
            range_l=int.from_bytes(payload[7:11], "big"),
            offset_l=int.from_bytes(payload[11:15], "big"),
            range_a=int.from_bytes(payload[15:19], "big"),
            offset_a=int.from_bytes(payload[19:23], "big"),
            range_b=int.from_bytes(payload[23:27], "big"),
            offset_b=int.from_bytes(payload[27:31], "big"),
            illuminant=int.from_bytes(payload[31:35], "big"),
        )
    return Jp2CieLabParameters(
        range_l=100,
        offset_l=0,
        range_a=170,
        offset_a=128,
        range_b=200,
        offset_b=96,
        illuminant=0x00443530,
    )


def parse_jp2_palette(payload: bytes) -> Jp2Palette:
    if len(payload) < 3:
        raise JpegParseError("truncated JP2 palette box")
    entry_count = int.from_bytes(payload[:2], "big")
    channel_count = payload[2]
    offset = 3
    if entry_count == 0 or entry_count > 1024:
        raise JpegParseError("invalid JP2 palette entry count")
    if channel_count <= 0:
        raise JpegParseError("invalid JP2 palette channel count")
    if offset + channel_count > len(payload):
        raise JpegParseError("truncated JP2 palette bit depths")
    bit_depths: list[int] = []
    signed: list[bool] = []
    for ignored in range(channel_count):
        spec = payload[offset]
        offset += 1
        bit_depths.append((spec & 0x7F) + 1)
        signed.append(bool(spec & 0x80))
    entries: list[tuple[int, ...]] = []
    for ignored_entry in range(entry_count):
        entry: list[int] = []
        for depth, is_signed in zip(bit_depths, signed, strict=True):
            byte_count = (depth + 7) // 8
            if offset + byte_count > len(payload):
                raise JpegParseError("truncated JP2 palette entries")
            raw_value = int.from_bytes(payload[offset : offset + byte_count], "big")
            offset += byte_count
            entry.append(normalize_jp2_palette_value(raw_value, depth, is_signed))
        entries.append(tuple(entry))
    return Jp2Palette(entries=tuple(entries))


def normalize_jp2_palette_value(value: int, bit_depth: int, is_signed: bool) -> int:
    if bit_depth <= 0:
        return 0
    max_value = (1 << bit_depth) - 1
    if is_signed:
        sign_bit = 1 << (bit_depth - 1)
        if value & sign_bit:
            value -= 1 << bit_depth
        value += sign_bit
    value = max(0, min(max_value, value))
    if max_value == 255:
        return value
    return (value * 255 + max_value // 2) // max_value


def parse_jp2_component_mapping(
    payload: bytes,
    channel_count: int | None = None,
) -> tuple[Jp2ComponentMapping, ...]:
    if not payload or len(payload) % 4:
        raise JpegParseError("invalid JP2 component mapping box length")
    mappings: list[Jp2ComponentMapping] = []
    for offset in range(0, len(payload), 4):
        mapping_type = payload[offset + 2]
        palette_column = payload[offset + 3]
        if channel_count is not None:
            if mapping_type not in {0, 1}:
                raise JpegParseError("invalid JP2 component mapping type")
            if palette_column >= channel_count:
                raise JpegParseError("JP2 component mapping palette column out of range")
            if mapping_type == 0 and palette_column != 0:
                raise JpegParseError("invalid JP2 direct component mapping")
        mappings.append(
            Jp2ComponentMapping(
                component=int.from_bytes(payload[offset : offset + 2], "big"),
                mapping_type=mapping_type,
                palette_column=palette_column,
            )
        )
    return tuple(mappings)


def parse_jp2_resolution(payload: bytes) -> Jp2Resolution:
    if len(payload) != 10:
        raise JpegParseError("bad JP2 resolution box size")
    vertical_denominator = int.from_bytes(payload[2:4], "big")
    horizontal_denominator = int.from_bytes(payload[6:8], "big")
    if vertical_denominator == 0 or horizontal_denominator == 0:
        raise JpegParseError("invalid JP2 resolution denominator")
    vertical_exponent = int.from_bytes(payload[8:9], "big", signed=True)
    horizontal_exponent = int.from_bytes(payload[9:10], "big", signed=True)
    return Jp2Resolution(
        vertical=(
            int.from_bytes(payload[0:2], "big") / vertical_denominator * (10.0**vertical_exponent)
        ),
        horizontal=(
            int.from_bytes(payload[4:6], "big")
            / horizontal_denominator
            * (10.0**horizontal_exponent)
        ),
    )


def validate_restricted_icc_profile(profile: bytes) -> None:
    if len(profile) < 128:
        raise JpegParseError("truncated JP2 restricted ICC profile")
    declared_size = int.from_bytes(profile[0:4], "big")
    if declared_size < 128 or declared_size > len(profile):
        raise JpegParseError("bad JP2 restricted ICC profile size")
    if profile[36:40] != b"acsp":
        raise JpegParseError("bad JP2 restricted ICC profile signature")
    if profile[16:20] not in {b"GRAY", b"RGB ", b"YCbr"}:
        raise JpegUnsupportedError("unsupported JP2 restricted ICC input color space")


def parse_jp2_channel_definitions(payload: bytes) -> tuple[Jp2ChannelDefinition, ...]:
    if len(payload) < 2:
        raise JpegParseError("truncated JP2 channel definition box")
    count = int.from_bytes(payload[:2], "big")
    if count <= 0:
        raise JpegParseError("invalid JP2 channel definition count")
    if len(payload) < 2 + count * 6:
        raise JpegParseError("truncated JP2 channel definition entries")
    definitions: list[Jp2ChannelDefinition] = []
    seen_components: set[int] = set()
    seen_color_associations: set[int] = set()
    offset = 2
    for ignored in range(count):
        component = int.from_bytes(payload[offset : offset + 2], "big")
        channel_type = int.from_bytes(payload[offset + 2 : offset + 4], "big")
        association = int.from_bytes(payload[offset + 4 : offset + 6], "big")
        if component in seen_components:
            raise JpegParseError("duplicate JP2 channel definition component")
        seen_components.add(component)
        if channel_type == 0 and 0 < association < 65535:
            if association in seen_color_associations:
                raise JpegParseError("duplicate JP2 channel definition association")
            seen_color_associations.add(association)
        definitions.append(
            Jp2ChannelDefinition(
                component=component,
                channel_type=channel_type,
                association=association,
            )
        )
        offset += 6
    return tuple(definitions)
