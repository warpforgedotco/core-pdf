# SPDX-License-Identifier: AGPL-3.0-only
"""Parse ICC profiles and apply their curves and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypedDict

import numpy

from core_pdf.impl.engine.spec.s_08_graphics.color_math import (
    adapt_d50_to_d65,
    lab_to_xyz,
    xyz_to_srgb,
)

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]


class IccProfileError(ValueError):
    """Raised when an ICC profile cannot be converted by core-color."""


class IccSampleError(ValueError):
    """Raised when sample arrays do not match an ICC transform."""


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


@dataclass(frozen=True, slots=True)
class IccTransform:
    """A compiled ICC transform from profile samples to normalized sRGB."""

    profile: IccMatrixProfile | IccLutProfile

    @property
    def input_channels(self) -> int:
        if isinstance(self.profile, IccMatrixProfile):
            return 1 if self.profile.color_space == "GRAY" else 3
        return self.profile.input_channels

    @property
    def output_channels(self) -> int:
        return 3

    @property
    def color_space(self) -> str:
        return self.profile.color_space

    @property
    def pcs(self) -> str:
        return self.profile.pcs

    @property
    def alternate_color_space(self) -> str:
        return {
            "GRAY": "DeviceGray",
            "RGB": "DeviceRGB",
            "CMYK": "DeviceCMYK",
        }.get(self.profile.color_space, "DeviceRGB")

    def apply(self, samples: ColorSamples) -> ColorSamples:
        validate_color_samples(samples, self.input_channels)
        if isinstance(self.profile, IccMatrixProfile):
            return apply_matrix_transform(self.profile, samples)
        return apply_lut_transform(self.profile, samples)


@lru_cache(maxsize=256)
def internal_curve_table_arrays(
    values: tuple[float, ...],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    axis = numpy.linspace(0.0, 1.0, len(values), dtype=numpy.float32)
    table = numpy.asarray(values, dtype=numpy.float32)
    axis.flags.writeable = False
    table.flags.writeable = False
    return axis, table


@lru_cache(maxsize=256)
def internal_readonly_float32(
    values: tuple[tuple[float, ...], ...],
) -> numpy.ndarray[Any, Any]:
    """Cache one read-only float32 array per distinct matrix or CLUT."""
    result = numpy.asarray(values, dtype=numpy.float32)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=128)
def parse_icc_transform(profile: bytes) -> IccTransform:
    lut_profile = parse_icc_lut_profile(profile)
    if lut_profile is not None:
        if lut_profile.output_channels < 3:
            raise IccProfileError("ICC LUT has fewer than three output channels")
        return IccTransform(lut_profile)
    matrix_profile = parse_icc_matrix_profile(profile)
    if matrix_profile is not None:
        return IccTransform(matrix_profile)
    raise IccProfileError("unsupported or malformed ICC profile")


def validate_color_samples(samples: ColorSamples, channels: int) -> None:
    if not isinstance(samples, numpy.ndarray):
        raise IccSampleError("samples must be a NumPy array")
    if samples.dtype != numpy.dtype(numpy.float32):
        raise IccSampleError("samples must have float32 dtype")
    if samples.ndim != 2 or samples.shape[1] != channels:
        raise IccSampleError(f"samples must have shape (count, {channels})")
    if not samples.flags.c_contiguous:
        raise IccSampleError("samples must be C-contiguous")


def apply_matrix_transform(profile: IccMatrixProfile, samples: ColorSamples) -> ColorSamples:
    curves = numpy.column_stack(
        [
            apply_curve_array(profile.curves[index], samples[:, index])
            for index in range(len(profile.curves))
        ]
    ).astype(numpy.float32)
    if profile.color_space == "GRAY":
        xyz = curves * numpy.asarray(profile.white_point, dtype=numpy.float32)
    else:
        xyz = curves @ internal_readonly_float32(profile.matrix)
    return numpy.clip(xyz_to_srgb(adapt_d50_to_d65(xyz)), 0.0, 1.0).astype(
        numpy.float32,
        copy=False,
    )


def apply_curve_array(curve: IccCurve, values: numpy.ndarray[Any, Any]) -> numpy.ndarray[Any, Any]:
    samples = numpy.clip(values, 0.0, 1.0).astype(numpy.float32, copy=False)
    if curve.kind == "identity":
        return samples
    if curve.kind == "gamma":
        return numpy.power(samples, curve.values[0]).astype(numpy.float32)
    if curve.kind == "table":
        axis, table = internal_curve_table_arrays(curve.values)
        return numpy.interp(
            samples,
            axis,
            table,
        ).astype(numpy.float32)
    if curve.kind == "parametric":
        return apply_parametric_curve_array(curve.values, samples)
    return samples


def apply_parametric_curve_array(
    values: tuple[float, ...], samples: numpy.ndarray[Any, Any]
) -> numpy.ndarray[Any, Any]:
    function_type = int(values[0])
    params = values[1:]
    gamma = params[0] if params else 1.0
    if function_type == 0:
        return numpy.power(samples, gamma).astype(numpy.float32)
    if function_type == 1 and len(params) >= 3:
        g, a, b = params[:3]
        threshold = -b / a if a else 0.0
        return numpy.where(
            samples >= threshold,
            numpy.power(numpy.maximum(0.0, a * samples + b), g),
            0.0,
        )
    if function_type == 2 and len(params) >= 4:
        g, a, b, c = params[:4]
        threshold = -b / a if a else 0.0
        return numpy.where(
            samples >= threshold,
            numpy.power(numpy.maximum(0.0, a * samples + b), g) + c,
            c,
        )
    if function_type == 3 and len(params) >= 5:
        g, a, b, c, d = params[:5]
        return numpy.where(
            samples >= d,
            numpy.power(numpy.maximum(0.0, a * samples + b), g),
            c * samples,
        )
    if function_type == 4 and len(params) >= 7:
        g, a, b, c, d, e, f = params[:7]
        return numpy.where(
            samples >= d,
            numpy.power(numpy.maximum(0.0, a * samples + b), g) + e,
            c * samples + f,
        )
    return samples


def apply_lut_transform(profile: IccLutProfile, samples: ColorSamples) -> ColorSamples:
    input_table_arrays = tuple(internal_curve_table_arrays(table) for table in profile.input_tables)
    values = numpy.column_stack(
        [
            numpy.interp(
                samples[:, index],
                axis,
                table,
            )
            for index, (axis, table) in enumerate(input_table_arrays)
        ]
    ).astype(numpy.float32)
    if profile.color_space == "XYZ" and profile.input_channels == 3:
        values = numpy.clip(
            values @ internal_readonly_float32(profile.matrix).T,
            0.0,
            1.0,
        )
    clut = interpolate_lut_array(
        internal_readonly_float32(profile.clut),
        profile.grid_points,
        values,
    )
    output_table_arrays = tuple(
        internal_curve_table_arrays(table) for table in profile.output_tables[:3]
    )
    output = numpy.column_stack(
        [
            numpy.interp(
                clut[:, index],
                axis,
                table,
            )
            for index, (axis, table) in enumerate(output_table_arrays)
        ]
    ).astype(numpy.float32)
    if profile.pcs == "Lab":
        xyz = lab_to_xyz(output, (0.9642, 1.0, 0.8249))
    else:
        xyz = output
    return numpy.clip(xyz_to_srgb(adapt_d50_to_d65(xyz)), 0.0, 1.0).astype(
        numpy.float32,
        copy=False,
    )


def interpolate_lut_array(
    clut: numpy.ndarray[Any, Any],
    grid_points: int,
    values: ColorSamples,
) -> ColorSamples:
    if grid_points <= 1:
        return numpy.broadcast_to(clut[0], (len(values), clut.shape[1])).copy()
    channels = values.shape[1]
    scaled = numpy.clip(values, 0.0, 1.0) * (grid_points - 1)
    base = numpy.minimum(scaled.astype(numpy.intp), grid_points - 2)
    fraction = scaled - base
    result = numpy.zeros((len(values), clut.shape[1]), dtype=numpy.float32)
    strides = numpy.empty(channels, dtype=numpy.intp)
    stride = 1
    for axis in range(channels - 1, -1, -1):
        strides[axis] = stride
        stride *= grid_points
    base_indices = numpy.sum(base * strides[None, :], axis=1, dtype=numpy.intp)
    for corner in range(1 << channels):
        weight = numpy.ones(len(values), dtype=numpy.float32)
        index_offset = 0
        for axis in range(channels):
            high = bool(corner & (1 << axis))
            if high:
                index_offset += int(strides[axis])
                weight *= fraction[:, axis]
            else:
                weight *= 1.0 - fraction[:, axis]
        indices = base_indices + index_offset
        result += clut[indices] * weight[:, None]
    return result


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
        input_tables, offset = parse_icc_lut_grid(
            payload, offset, input_channels, input_table_len, sample_bytes=1
        )
        clut_entries = grid_points**input_channels
        clut, offset = parse_icc_lut_grid(
            payload, offset, clut_entries, output_channels, sample_bytes=1
        )
        output_tables, offset = parse_icc_lut_grid(
            payload, offset, output_channels, output_table_len, sample_bytes=1
        )
    else:
        if len(payload) < 52:
            return None
        input_table_len = int.from_bytes(payload[48:50], "big")
        output_table_len = int.from_bytes(payload[50:52], "big")
        if input_table_len <= 0 or output_table_len <= 0:
            return None
        offset = 52
        input_tables, offset = parse_icc_lut_grid(
            payload, offset, input_channels, input_table_len, sample_bytes=2
        )
        clut_entries = grid_points**input_channels
        clut, offset = parse_icc_lut_grid(
            payload, offset, clut_entries, output_channels, sample_bytes=2
        )
        output_tables, offset = parse_icc_lut_grid(
            payload, offset, output_channels, output_table_len, sample_bytes=2
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


def parse_icc_lut_grid(
    payload: bytes,
    offset: int,
    rows: int,
    columns: int,
    *,
    sample_bytes: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, int]:
    if sample_bytes not in (1, 2):
        raise ValueError("ICC LUT samples must be one or two bytes")
    scale = 255.0 if sample_bytes == 1 else 65535.0
    values: list[tuple[float, ...]] = []
    for _ in range(rows):
        row: list[float] = []
        for _ in range(columns):
            end = offset + sample_bytes
            if end > len(payload):
                return None, offset
            sample = (
                payload[offset] if sample_bytes == 1 else int.from_bytes(payload[offset:end], "big")
            )
            row.append(sample / scale)
            offset = end
        values.append(tuple(row))
    return tuple(values), offset


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
