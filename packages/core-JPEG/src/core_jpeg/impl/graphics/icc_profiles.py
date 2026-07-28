# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict

from core_jpeg.impl.graphics.color_math import (
    adapt_d50_to_d65,
    lab_to_xyz,
    xyz_to_srgb,
)


@dataclass(frozen=True)
class IccCurve:
    kind: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class IccMatrixProfile:
    color_space: str
    pcs: str
    white_point: tuple[float, float, float]
    matrix: tuple[tuple[float, float, float], ...]
    curves: tuple[IccCurve, ...]


@dataclass(frozen=True)
class IccLutProfile:
    color_space: str
    pcs: str
    input_channels: int
    output_channels: int
    grid_points: int
    matrix: tuple[tuple[float, float, float], ...]
    input_tables: tuple[tuple[float, ...], ...]
    clut: tuple[tuple[float, ...], ...]
    output_tables: tuple[tuple[float, ...], ...]


class IccLutTag(TypedDict):
    input_channels: int
    output_channels: int
    grid_points: int
    matrix: tuple[tuple[float, float, float], ...]
    input_tables: tuple[tuple[float, ...], ...]
    clut: tuple[tuple[float, ...], ...]
    output_tables: tuple[tuple[float, ...], ...]


def icc_profile_alt_name(profile: bytes | None, channels: int) -> str | None:
    if profile is None or len(profile) < 128:
        return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)
    pcs = profile[20:24]
    if pcs == b"XYZ ":
        return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)
    color_space = profile[16:20]
    if color_space == b"GRAY":
        return "DeviceGray"
    if color_space == b"RGB ":
        return "DeviceRGB"
    if color_space == b"CMYK":
        return "DeviceCMYK"
    return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)


def convert_icc_profile_samples(raw: bytes, profile: bytes | None) -> bytes | None:
    if profile is None:
        return None
    lut_profile = parse_icc_lut_profile(profile)
    if lut_profile is not None:
        return convert_icc_lut_samples(raw, lut_profile)
    parsed = parse_icc_matrix_profile(profile)
    if parsed is None:
        return None
    if parsed.color_space == "GRAY":
        if not raw:
            return raw
        return convert_icc_gray_samples(raw, parsed)
    if parsed.color_space == "RGB":
        if len(raw) % 3:
            return None
        return convert_icc_rgb_samples(raw, parsed)
    return None


def convert_icc_lut_samples(raw: bytes, profile: IccLutProfile) -> bytes | None:
    if profile.input_channels <= 0 or len(raw) % profile.input_channels:
        return None
    result = bytearray((len(raw) // profile.input_channels) * 3)
    for sample_index, offset in enumerate(range(0, len(raw), profile.input_channels)):
        components = [raw[offset + i] / 255.0 for i in range(profile.input_channels)]
        pcs = evaluate_icc_lut(profile, components)
        if len(pcs) < 3:
            return None
        red, green, blue = icc_pcs_to_srgb(profile.pcs, pcs)
        out = sample_index * 3
        result[out] = max(0, min(255, int(round(red * 255.0))))
        result[out + 1] = max(0, min(255, int(round(green * 255.0))))
        result[out + 2] = max(0, min(255, int(round(blue * 255.0))))
    return bytes(result)


def evaluate_icc_lut(profile: IccLutProfile, components: list[float]) -> tuple[float, ...]:
    values = [max(0.0, min(1.0, component)) for component in components]
    if profile.color_space == "XYZ" and profile.input_channels == 3 and len(profile.matrix) == 3:
        values = apply_icc_matrix(profile.matrix, values)
    table_values = [
        interpolate_1d_table(profile.input_tables[index], values[index])
        for index in range(profile.input_channels)
    ]
    clut_values = interpolate_icc_clut(
        profile.clut,
        profile.grid_points,
        profile.input_channels,
        profile.output_channels,
        table_values,
    )
    return tuple(
        interpolate_1d_table(profile.output_tables[index], clut_values[index])
        for index in range(profile.output_channels)
    )


def apply_icc_matrix(
    matrix: tuple[tuple[float, float, float], ...], values: list[float]
) -> list[float]:
    return [
        max(
            0.0,
            min(
                1.0,
                matrix[row][0] * values[0]
                + matrix[row][1] * values[1]
                + matrix[row][2] * values[2],
            ),
        )
        for row in range(3)
    ]


def interpolate_1d_table(table: tuple[float, ...], value: float) -> float:
    if not table:
        return max(0.0, min(1.0, value))
    if len(table) == 1:
        return table[0]
    scaled = max(0.0, min(1.0, value)) * (len(table) - 1)
    low = int(scaled)
    high = min(low + 1, len(table) - 1)
    frac = scaled - low
    return table[low] * (1.0 - frac) + table[high] * frac


def interpolate_icc_clut(
    clut: tuple[tuple[float, ...], ...],
    grid_points: int,
    input_channels: int,
    output_channels: int,
    values: list[float],
) -> list[float]:
    if grid_points <= 1:
        return list(clut[0]) if clut else [0.0] * output_channels
    scaled = [max(0.0, min(1.0, value)) * (grid_points - 1) for value in values]
    base = [min(int(value), grid_points - 2) for value in scaled]
    frac = [value - base[index] for index, value in enumerate(scaled)]
    result = [0.0] * output_channels
    for corner_bits in range(1 << input_channels):
        weight = 1.0
        coords: list[int] = []
        for axis in range(input_channels):
            if corner_bits & (1 << axis):
                coords.append(base[axis] + 1)
                weight *= frac[axis]
            else:
                coords.append(base[axis])
                weight *= 1.0 - frac[axis]
        entry = clut[icc_clut_index(coords, grid_points)]
        for channel in range(output_channels):
            result[channel] += entry[channel] * weight
    return result


def icc_clut_index(coords: list[int], grid_points: int) -> int:
    index = 0
    for coord in coords:
        index = index * grid_points + coord
    return index


def icc_pcs_to_srgb(pcs: str, values: tuple[float, ...]) -> tuple[float, float, float]:
    if pcs == "XYZ":
        ax, ay, az = adapt_d50_to_d65(values[0], values[1], values[2])
        return xyz_to_srgb(ax, ay, az)
    if pcs == "Lab":
        l_star = values[0] * 100.0
        a_star = values[1] * 255.0 - 128.0
        b_star = values[2] * 255.0 - 128.0
        x, y, z = lab_to_xyz(l_star, a_star, b_star, (0.9642, 1.0, 0.8249))
        ax, ay, az = adapt_d50_to_d65(x, y, z)
        return xyz_to_srgb(ax, ay, az)
    return 0.0, 0.0, 0.0


def convert_icc_gray_samples(raw: bytes, profile: IccMatrixProfile) -> bytes:
    result = bytearray(len(raw) * 3)
    for index, value in enumerate(raw):
        linear = apply_icc_curve(profile.curves[0], value / 255.0)
        x = profile.white_point[0] * linear
        y = profile.white_point[1] * linear
        z = profile.white_point[2] * linear
        ax, ay, az = adapt_d50_to_d65(x, y, z)
        red, green, blue = xyz_to_srgb(ax, ay, az)
        out = index * 3
        result[out] = max(0, min(255, int(round(red * 255.0))))
        result[out + 1] = max(0, min(255, int(round(green * 255.0))))
        result[out + 2] = max(0, min(255, int(round(blue * 255.0))))
    return bytes(result)


def convert_icc_rgb_samples(raw: bytes, profile: IccMatrixProfile) -> bytes:
    result = bytearray(len(raw))
    for index in range(0, len(raw), 3):
        red = apply_icc_curve(profile.curves[0], raw[index] / 255.0)
        green = apply_icc_curve(profile.curves[1], raw[index + 1] / 255.0)
        blue = apply_icc_curve(profile.curves[2], raw[index + 2] / 255.0)
        x = red * profile.matrix[0][0] + green * profile.matrix[1][0] + blue * profile.matrix[2][0]
        y = red * profile.matrix[0][1] + green * profile.matrix[1][1] + blue * profile.matrix[2][1]
        z = red * profile.matrix[0][2] + green * profile.matrix[1][2] + blue * profile.matrix[2][2]
        ax, ay, az = adapt_d50_to_d65(x, y, z)
        sr, sg, sb = xyz_to_srgb(ax, ay, az)
        result[index] = max(0, min(255, int(round(sr * 255.0))))
        result[index + 1] = max(0, min(255, int(round(sg * 255.0))))
        result[index + 2] = max(0, min(255, int(round(sb * 255.0))))
    return bytes(result)


def apply_icc_curve(curve: IccCurve, value: float) -> float:
    sample = max(0.0, min(1.0, value))
    if curve.kind == "identity":
        return sample
    if curve.kind == "gamma":
        return pow(sample, curve.values[0])
    if curve.kind == "table":
        table = curve.values
        if not table:
            return sample
        if len(table) == 1:
            return table[0]
        scaled = sample * (len(table) - 1)
        low = int(scaled)
        high = min(low + 1, len(table) - 1)
        frac = scaled - low
        return table[low] * (1.0 - frac) + table[high] * frac
    if curve.kind == "parametric":
        function_type = int(curve.values[0])
        params = curve.values[1:]
        gamma = params[0] if params else 1.0
        if function_type == 0:
            return pow(sample, gamma)
        if function_type == 1 and len(params) >= 3:
            g, a, b = params[:3]
            return pow(max(0.0, a * sample + b), g) if sample >= (-b / a) else 0.0
        if function_type == 2 and len(params) >= 4:
            g, a, b, c = params[:4]
            return pow(max(0.0, a * sample + b), g) + c if sample >= (-b / a) else c
        if function_type == 3 and len(params) >= 5:
            g, a, b, c, d = params[:5]
            return pow(max(0.0, a * sample + b), g) if sample >= d else c * sample
        if function_type == 4 and len(params) >= 7:
            g, a, b, c, d, e, f = params[:7]
            return pow(max(0.0, a * sample + b), g) + e if sample >= d else c * sample + f
    return sample


@lru_cache(maxsize=128)
def parse_icc_matrix_profile(profile: bytes) -> IccMatrixProfile | None:
    if len(profile) < 132:
        return None
    color_space = profile[16:20]
    pcs = profile[20:24]
    tags = parse_icc_tags(profile)
    if color_space == b"GRAY" and pcs == b"XYZ ":
        white_point = parse_icc_xyz_tag(tags.get(b"wtpt")) or (0.9642, 1.0, 0.8249)
        curve = parse_icc_curve_tag(tags.get(b"kTRC"))
        if curve is None:
            return None
        return IccMatrixProfile(
            color_space="GRAY",
            pcs="XYZ",
            white_point=white_point,
            matrix=((white_point[0], white_point[1], white_point[2]),),
            curves=(curve,),
        )
    if color_space == b"RGB " and pcs == b"XYZ ":
        red_xyz = parse_icc_xyz_tag(tags.get(b"rXYZ"))
        green_xyz = parse_icc_xyz_tag(tags.get(b"gXYZ"))
        blue_xyz = parse_icc_xyz_tag(tags.get(b"bXYZ"))
        red_trc = parse_icc_curve_tag(tags.get(b"rTRC"))
        green_trc = parse_icc_curve_tag(tags.get(b"gTRC"))
        blue_trc = parse_icc_curve_tag(tags.get(b"bTRC"))
        if (
            red_xyz is None
            or green_xyz is None
            or blue_xyz is None
            or red_trc is None
            or green_trc is None
            or blue_trc is None
        ):
            return None
        white_point = parse_icc_xyz_tag(tags.get(b"wtpt")) or (0.9642, 1.0, 0.8249)
        return IccMatrixProfile(
            color_space="RGB",
            pcs="XYZ",
            white_point=white_point,
            matrix=(red_xyz, green_xyz, blue_xyz),
            curves=(red_trc, green_trc, blue_trc),
        )
    return None


@lru_cache(maxsize=128)
def parse_icc_lut_profile(profile: bytes) -> IccLutProfile | None:
    if len(profile) < 132:
        return None
    color_space = icc_color_space_name(profile[16:20])
    pcs = icc_pcs_name(profile[20:24])
    if color_space is None or pcs is None:
        return None
    tags = parse_icc_tags(profile)
    intent = parse_icc_rendering_intent(profile)
    lut = parse_icc_lut_tag(
        select_icc_device_to_pcs_lut_tag(tags, intent),
    )
    if lut is None:
        return None
    return IccLutProfile(
        color_space=color_space,
        pcs=pcs,
        input_channels=lut["input_channels"],
        output_channels=lut["output_channels"],
        grid_points=lut["grid_points"],
        matrix=lut["matrix"],
        input_tables=lut["input_tables"],
        clut=lut["clut"],
        output_tables=lut["output_tables"],
    )


def parse_icc_tags(profile: bytes) -> dict[bytes, bytes]:
    if len(profile) < 132:
        return {}
    tag_count = int.from_bytes(profile[128:132], "big")
    tags: dict[bytes, bytes] = {}
    offset = 132
    for _ in range(tag_count):
        if offset + 12 > len(profile):
            return {}
        signature = profile[offset : offset + 4]
        tag_offset = int.from_bytes(profile[offset + 4 : offset + 8], "big")
        tag_size = int.from_bytes(profile[offset + 8 : offset + 12], "big")
        offset += 12
        end = tag_offset + tag_size
        if tag_offset < 0 or tag_size < 0 or end > len(profile):
            return {}
        tags[signature] = profile[tag_offset:end]
    return tags


def parse_icc_xyz_tag(payload: bytes | None) -> tuple[float, float, float] | None:
    if payload is None or len(payload) < 20 or payload[:4] != b"XYZ ":
        return None
    return (
        s15fixed16(payload[8:12]),
        s15fixed16(payload[12:16]),
        s15fixed16(payload[16:20]),
    )


def parse_icc_curve_tag(payload: bytes | None) -> IccCurve | None:
    if payload is None or len(payload) < 12:
        return None
    tag_type = payload[:4]
    if tag_type == b"curv":
        count = int.from_bytes(payload[8:12], "big")
        if count == 0:
            return IccCurve(kind="identity", values=())
        if count == 1:
            if len(payload) < 14:
                return None
            return IccCurve(kind="gamma", values=(u8fixed8(payload[12:14]),))
        values: list[float] = []
        offset = 12
        for _ in range(count):
            if offset + 2 > len(payload):
                return None
            values.append(int.from_bytes(payload[offset : offset + 2], "big") / 65535.0)
            offset += 2
        return IccCurve(kind="table", values=tuple(values))
    if tag_type == b"para":
        if len(payload) < 12:
            return None
        function_type = int.from_bytes(payload[8:10], "big")
        parameter_count = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function_type)
        if parameter_count is None or len(payload) < 12 + parameter_count * 4:
            return None
        values = [float(function_type)]
        offset = 12
        for _ in range(parameter_count):
            values.append(s15fixed16(payload[offset : offset + 4]))
            offset += 4
        return IccCurve(kind="parametric", values=tuple(values))
    return None


def parse_icc_rendering_intent(profile: bytes) -> int:
    if len(profile) < 68:
        return 0
    return int.from_bytes(profile[64:68], "big")


def select_icc_device_to_pcs_lut_tag(
    tags: dict[bytes, bytes],
    intent: int,
) -> bytes | None:
    preferred = {
        0: (b"A2B0", b"A2B1", b"A2B2"),
        1: (b"A2B1", b"A2B0", b"A2B2"),
        2: (b"A2B2", b"A2B1", b"A2B0"),
        3: (b"A2B1", b"A2B0", b"A2B2"),
    }.get(intent, (b"A2B0", b"A2B1", b"A2B2"))
    for signature in preferred:
        payload = tags.get(signature)
        if payload is not None:
            return payload
    return None


def parse_icc_lut_tag(payload: bytes | None) -> IccLutTag | None:
    if payload is None or len(payload) < 52:
        return None
    tag_type = payload[:4]
    if tag_type not in {b"mft1", b"mft2"}:
        return None
    input_channels = payload[8]
    output_channels = payload[9]
    grid_points = payload[10]
    if input_channels <= 0 or output_channels <= 0 or grid_points <= 0:
        return None

    def matrix_value(row: int, col: int) -> float:
        offset = 12 + (row * 3 + col) * 4
        return s15fixed16(payload[offset : offset + 4])

    matrix: tuple[tuple[float, float, float], ...] = (
        (matrix_value(0, 0), matrix_value(0, 1), matrix_value(0, 2)),
        (matrix_value(1, 0), matrix_value(1, 1), matrix_value(1, 2)),
        (matrix_value(2, 0), matrix_value(2, 1), matrix_value(2, 2)),
    )
    offset = 48
    if tag_type == b"mft1":
        input_table_len = 256
        output_table_len = 256
        input_tables, offset = parse_icc_lut_tables_u8(
            payload,
            offset,
            input_channels,
            input_table_len,
        )
        clut_entries = grid_points**input_channels
        clut, offset = parse_icc_lut_clut_u8(
            payload,
            offset,
            clut_entries,
            output_channels,
        )
        output_tables, offset = parse_icc_lut_tables_u8(
            payload,
            offset,
            output_channels,
            output_table_len,
        )
    else:
        if len(payload) < 52:
            return None
        input_table_len = int.from_bytes(payload[48:50], "big")
        output_table_len = int.from_bytes(payload[50:52], "big")
        if input_table_len <= 0 or output_table_len <= 0:
            return None
        offset = 52
        input_tables, offset = parse_icc_lut_tables_u16(
            payload,
            offset,
            input_channels,
            input_table_len,
        )
        clut_entries = grid_points**input_channels
        clut, offset = parse_icc_lut_clut_u16(
            payload,
            offset,
            clut_entries,
            output_channels,
        )
        output_tables, offset = parse_icc_lut_tables_u16(
            payload,
            offset,
            output_channels,
            output_table_len,
        )
    if input_tables is None or clut is None or output_tables is None:
        return None
    tag: IccLutTag = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "grid_points": grid_points,
        "matrix": matrix,
        "input_tables": input_tables,
        "clut": clut,
        "output_tables": output_tables,
    }
    return tag


def parse_icc_lut_tables_u8(
    payload: bytes,
    offset: int,
    channels: int,
    entries: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    tables: list[tuple[float, ...]] = []
    for _ in range(channels):
        end = offset + entries
        if end > len(payload):
            return None, offset
        tables.append(tuple(byte / 255.0 for byte in payload[offset:end]))
        offset = end
    return tuple(tables), offset


def parse_icc_lut_tables_u16(
    payload: bytes,
    offset: int,
    channels: int,
    entries: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    tables: list[tuple[float, ...]] = []
    for _ in range(channels):
        values: list[float] = []
        for _ in range(entries):
            end = offset + 2
            if end > len(payload):
                return None, offset
            values.append(int.from_bytes(payload[offset:end], "big") / 65535.0)
            offset = end
        tables.append(tuple(values))
    return tuple(tables), offset


def parse_icc_lut_clut_u8(
    payload: bytes,
    offset: int,
    entries: int,
    channels: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    clut: list[tuple[float, ...]] = []
    for _ in range(entries):
        end = offset + channels
        if end > len(payload):
            return None, offset
        clut.append(tuple(byte / 255.0 for byte in payload[offset:end]))
        offset = end
    return tuple(clut), offset


def parse_icc_lut_clut_u16(
    payload: bytes,
    offset: int,
    entries: int,
    channels: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    clut: list[tuple[float, ...]] = []
    for _ in range(entries):
        values: list[float] = []
        for _ in range(channels):
            end = offset + 2
            if end > len(payload):
                return None, offset
            values.append(int.from_bytes(payload[offset:end], "big") / 65535.0)
            offset = end
        clut.append(tuple(values))
    return tuple(clut), offset


def icc_color_space_name(signature: bytes) -> str | None:
    if signature == b"GRAY":
        return "GRAY"
    if signature == b"RGB ":
        return "RGB"
    if signature == b"CMYK":
        return "CMYK"
    if signature == b"XYZ ":
        return "XYZ"
    return None


def icc_pcs_name(signature: bytes) -> str | None:
    if signature == b"XYZ ":
        return "XYZ"
    if signature == b"Lab ":
        return "Lab"
    return None


def s15fixed16(data: bytes) -> float:
    return int.from_bytes(data, "big", signed=True) / 65536.0


def u8fixed8(data: bytes) -> float:
    return int.from_bytes(data, "big") / 256.0
