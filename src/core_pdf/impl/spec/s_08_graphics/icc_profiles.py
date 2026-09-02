# SPDX-License-Identifier: AGPL-3.0-only
"""Parse ICC profiles and apply their curves and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, TypedDict

import numpy

from core_pdf.impl.runtime.array_views import readonly
from core_pdf.impl.spec.s_08_graphics.color_math import (
    d50_xyz_to_srgb,
    lab_to_xyz,
)

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]

# The profile connection space is always D50-relative.
INTERNAL_D50_WHITE = (0.9642, 1.0, 0.8249)

# ICC v2 lut8/lut16 tags carry Lab in the "legacy" encoding, where 0xFF00 -- not
# 0xFFFF -- is L*=100 and a*=b*=+127. Parsing divides every 16-bit sample by
# 0xFFFF, so legacy Lab has to be scaled back up by this ratio afterwards. Left
# out, pure white decodes to L*=99.6 and paints (252, 254, 255) instead of white.
INTERNAL_LEGACY_LAB_SCALE = 65535.0 / 65280.0

# Rows per block when applying a LUT transform. Multilinear interpolation holds
# 2**input_channels intermediate arrays of (rows, 3) float32 at once, so a
# full-page CMYK scan applied in one shot would allocate hundreds of megabytes.
INTERNAL_TRANSFORM_BLOCK_ROWS = 1 << 18

# Deduplicating a byte batch costs one sort, and pays for itself many times over
# on real images: a CMYK page photograph runs around 10% distinct colours, so
# the LUT walk shrinks by an order of magnitude. Below the row floor the sort is
# not worth setting up, and above the ratio floor the batch is close enough to
# all-distinct that deduplicating would be pure overhead.
INTERNAL_DEDUPLICATE_MIN_ROWS = 4096
INTERNAL_DEDUPLICATE_MIN_RATIO = 2


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

    @cached_property
    def matrix_array(self) -> numpy.ndarray[Any, Any]:
        return readonly(numpy.asarray(self.matrix, dtype=numpy.float32))


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
    legacy_lab: bool = False
    black_point: tuple[float, float, float] | None = None

    # These arrays were previously rebuilt through module-level lru_caches keyed
    # on the tuples themselves. A tuple does not memoize its hash, so every
    # cache *hit* re-walked the key: 2055us for an 83521x3 CLUT, of which 1950us
    # was hash(). The profile is already cached on its own bytes, so the derived
    # arrays belong on it. cached_property writes through __dict__ and so works
    # on a frozen dataclass.

    @cached_property
    def matrix_array(self) -> numpy.ndarray[Any, Any]:
        return readonly(numpy.asarray(self.matrix, dtype=numpy.float32))

    @cached_property
    def clut_array(self) -> numpy.ndarray[Any, Any]:
        return readonly(numpy.asarray(self.clut, dtype=numpy.float32))

    @cached_property
    def input_table_arrays(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(internal_curve_table_arrays(table) for table in self.input_tables)

    @cached_property
    def output_table_arrays(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(internal_curve_table_arrays(table) for table in self.output_tables[:3])

    @cached_property
    def byte_input_curves(self) -> tuple[numpy.ndarray[Any, Any], ...]:
        return internal_byte_input_curves(self.input_tables)


class IccLutTag(TypedDict):
    sample_bytes: int
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
        profile = self.profile
        if isinstance(profile, IccMatrixProfile):
            return apply_matrix_transform(profile, samples)
        rows = len(samples)
        if rows <= INTERNAL_TRANSFORM_BLOCK_ROWS:
            return apply_lut_transform(profile, samples)
        result = numpy.empty((rows, 3), dtype=numpy.float32)
        for start in range(0, rows, INTERNAL_TRANSFORM_BLOCK_ROWS):
            stop = min(start + INTERNAL_TRANSFORM_BLOCK_ROWS, rows)
            result[start:stop] = apply_lut_transform(
                profile,
                numpy.ascontiguousarray(samples[start:stop]),
            )
        return result

    def apply_uint8(
        self,
        samples: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
    ) -> numpy.ndarray[Any, numpy.dtype[numpy.uint8]]:
        """Convert 8-bit device samples to 8-bit sRGB.

        Byte inputs are the overwhelmingly common case -- every image sample
        arrives this way -- and they buy two things a float batch cannot. The
        per-channel input curves collapse into a 256-entry gather instead of an
        interpolation over the batch, and a byte tuple is small enough to
        deduplicate: real images repeat colours heavily, so only the distinct
        ones need to walk the LUT.
        """
        channels = self.input_channels
        if samples.dtype != numpy.dtype(numpy.uint8):
            raise IccSampleError("samples must have uint8 dtype")
        if samples.ndim != 2 or samples.shape[1] != channels:
            raise IccSampleError(f"samples must have shape (count, {channels})")
        if channels <= 4 and len(samples) > INTERNAL_DEDUPLICATE_MIN_ROWS:
            distinct, inverse = internal_distinct_byte_rows(samples)
            if len(distinct) * INTERNAL_DEDUPLICATE_MIN_RATIO < len(samples):
                # `distinct` is all-distinct by construction, so the ratio test
                # fails on the way back in and this recurses exactly once.
                return self.apply_uint8(distinct)[inverse]
        profile = self.profile
        rows = len(samples)
        result = numpy.empty((rows, 3), dtype=numpy.uint8)
        for start in range(0, rows, INTERNAL_TRANSFORM_BLOCK_ROWS):
            stop = min(start + INTERNAL_TRANSFORM_BLOCK_ROWS, rows)
            block = samples[start:stop]
            if isinstance(profile, IccMatrixProfile):
                rgb = apply_matrix_transform(
                    profile,
                    numpy.ascontiguousarray(block, dtype=numpy.float32) / numpy.float32(255.0),
                )
            else:
                curves = profile.byte_input_curves
                values = numpy.column_stack(
                    [curve[block[:, index]] for index, curve in enumerate(curves)]
                ).astype(numpy.float32, copy=False)
                rgb = internal_lut_transform_from_curved(profile, values)
            numpy.rint(rgb * numpy.float32(255.0), out=rgb)
            result[start:stop] = rgb.astype(numpy.uint8)
        return result


@lru_cache(maxsize=256)
def internal_curve_table_arrays(
    values: tuple[float, ...],
) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
    axis = readonly(numpy.linspace(0.0, 1.0, len(values), dtype=numpy.float32))
    table = readonly(numpy.asarray(values, dtype=numpy.float32))
    return axis, table


def internal_readonly_float32(
    values: tuple[tuple[float, ...], ...],
) -> numpy.ndarray[Any, Any]:
    """Cache one read-only float32 array per distinct matrix or CLUT."""
    return readonly(numpy.asarray(values, dtype=numpy.float32))


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
    first_curve = profile.curves[0]
    if all(curve == first_curve for curve in profile.curves):
        # One contiguous pass over the whole block; identical TRCs on every
        # channel is the overwhelmingly common profile shape.
        curves = apply_curve_array(first_curve, samples).astype(numpy.float32, copy=False)
    else:
        curves = numpy.column_stack(
            [
                apply_curve_array(profile.curves[index], samples[:, index])
                for index in range(len(profile.curves))
            ]
        ).astype(numpy.float32)
    if profile.color_space == "GRAY":
        xyz = curves * numpy.asarray(profile.white_point, dtype=numpy.float32)
    else:
        xyz = curves @ profile.matrix_array
    return numpy.clip(d50_xyz_to_srgb(xyz), 0.0, 1.0).astype(
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
    input_table_arrays = profile.input_table_arrays
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
    return internal_lut_transform_from_curved(profile, values)


def internal_lut_transform_from_curved(
    profile: IccLutProfile,
    values: ColorSamples,
) -> ColorSamples:
    """Finish a LUT transform whose input curves have already been applied."""
    if profile.color_space == "XYZ" and profile.input_channels == 3:
        values = numpy.clip(
            values @ profile.matrix_array.T,
            0.0,
            1.0,
        )
    clut = interpolate_lut_array(
        profile.clut_array,
        profile.grid_points,
        values,
    )
    output_table_arrays = profile.output_table_arrays
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
        if profile.legacy_lab:
            output *= numpy.float32(INTERNAL_LEGACY_LAB_SCALE)
        xyz = lab_to_xyz(output, INTERNAL_D50_WHITE)
    else:
        xyz = output
    if profile.black_point is not None:
        xyz = internal_compensate_black_point(xyz, profile.black_point)
    return numpy.clip(d50_xyz_to_srgb(xyz), 0.0, 1.0).astype(
        numpy.float32,
        copy=False,
    )


def internal_compensate_black_point(
    xyz: ColorSamples,
    black_point: tuple[float, float, float],
) -> ColorSamples:
    """Scale PCS values so the profile's darkest colour lands on true black.

    Relative colorimetric alone reproduces the press black as the dark grey it
    actually is on paper, so a CMYK page renders washed out on a screen that can
    show real black. Every colour-managed renderer therefore pairs relative
    colorimetric with black point compensation by default, and so do we: the
    source black is mapped to the destination black (0, sRGB can reach it) and
    everything between is scaled linearly, keeping white fixed.
    """
    white = numpy.asarray(INTERNAL_D50_WHITE, dtype=numpy.float32)
    black = numpy.asarray(black_point, dtype=numpy.float32)
    return ((xyz - black) * (white / (white - black))).astype(numpy.float32, copy=False)


def internal_distinct_byte_rows(
    samples: numpy.ndarray[Any, numpy.dtype[numpy.uint8]],
) -> tuple[numpy.ndarray[Any, numpy.dtype[numpy.uint8]], numpy.ndarray[Any, Any]]:
    """Split a byte batch into its distinct rows plus the index that rebuilds it.

    Reduces "unique rows" to a single sort over a flat integer array by packing
    the channels into one key. Callers must hold the batch to at most four
    channels, which is what fits in the 32-bit key.
    """
    channels = samples.shape[1]
    if channels > 4:
        raise IccSampleError("cannot pack more than four channels into one key")
    keys = numpy.zeros(len(samples), dtype=numpy.uint32)
    for index in range(channels):
        keys <<= numpy.uint32(8)
        keys |= samples[:, index]
    # The key holds the whole row, so the distinct rows unpack straight out of
    # the unique keys. Asking for return_index instead forces numpy.unique onto
    # a stable sort, which is materially slower for no extra information.
    unique_keys, inverse = numpy.unique(keys, return_inverse=True)
    distinct = numpy.empty((len(unique_keys), channels), dtype=numpy.uint8)
    for index in range(channels):
        shift = numpy.uint32(8 * (channels - 1 - index))
        distinct[:, index] = ((unique_keys >> shift) & numpy.uint32(0xFF)).astype(numpy.uint8)
    return distinct, inverse


def internal_byte_input_curves(
    input_tables: tuple[tuple[float, ...], ...],
) -> tuple[numpy.ndarray[Any, Any], ...]:
    """Collapse each input curve into a 256-entry table indexed by a raw byte."""
    positions = numpy.linspace(0.0, 1.0, 256, dtype=numpy.float32)
    curves: list[numpy.ndarray[Any, Any]] = []
    for table in input_tables:
        axis, values = internal_curve_table_arrays(table)
        curves.append(readonly(numpy.interp(positions, axis, values).astype(numpy.float32)))
    return tuple(curves)


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
    fraction = numpy.asarray(scaled - base, dtype=numpy.float32)
    inverse = 1.0 - fraction
    result = numpy.zeros((len(values), clut.shape[1]), dtype=numpy.float32)
    strides = numpy.empty(channels, dtype=numpy.intp)
    stride = 1
    for axis in range(channels - 1, -1, -1):
        strides[axis] = stride
        stride *= grid_points
    base_indices = numpy.sum(base * strides[None, :], axis=1, dtype=numpy.intp)
    weight = numpy.empty(len(values), dtype=numpy.float32)
    for corner in range(1 << channels):
        index_offset = 0
        numpy.copyto(weight, fraction[:, 0] if corner & 1 else inverse[:, 0])
        if corner & 1:
            index_offset += int(strides[0])
        for axis in range(1, channels):
            if corner & (1 << axis):
                index_offset += int(strides[axis])
                weight *= fraction[:, axis]
            else:
                weight *= inverse[:, axis]
        gathered = clut.take(base_indices + index_offset, axis=0)
        gathered *= weight[:, None]
        result += gathered
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
    lut = parse_icc_lut_tag(select_icc_lut_tag(tags, b"A2B"))
    if lut is None:
        return None
    legacy_lab = pcs == "Lab" and profile[8] < 4
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
        legacy_lab=legacy_lab,
        black_point=internal_detect_black_point(tags, lut, pcs, legacy_lab),
    )


def internal_detect_black_point(
    tags: dict[bytes, bytes],
    device_to_pcs: IccLutTag,
    pcs: str,
    legacy_lab: bool,
) -> tuple[float, float, float] | None:
    """Find the darkest colour the profile can actually reproduce.

    This is the black point detection from the ICC's own white paper, and it is
    not the same as "100% K": on a CMYK press the darkest reachable colour is a
    rich black mixing all four inks. Ask the profile which device values it
    would use for L*=0, clamp them to what the device can hold, then ask what
    that ink combination really measures. Returns None when the profile has no
    PCS-to-device table to ask, or when the answer is already black.
    """
    pcs_to_device = parse_icc_lut_tag(select_icc_lut_tag(tags, b"B2A"))
    if pcs_to_device is None:
        return None
    if pcs_to_device["input_channels"] != 3:
        return None
    if pcs_to_device["output_channels"] != device_to_pcs["input_channels"]:
        return None
    if pcs == "Lab":
        # L* = 0 sits at the bottom of the range, but a* = b* = 0 sits at the
        # neutral midpoint, which is 128 of 255 for an 8-bit table and 0x8000 of
        # 0xFFFF for a legacy 16-bit one.
        neutral = 128.0 / 255.0 if pcs_to_device["sample_bytes"] == 1 else 32768.0 / 65535.0
        black_pcs = numpy.asarray([[0.0, neutral, neutral]], dtype=numpy.float32)
    else:
        black_pcs = numpy.zeros((1, 3), dtype=numpy.float32)
    device = numpy.clip(internal_evaluate_lut_tag(pcs_to_device, black_pcs), 0.0, 1.0)
    measured = internal_evaluate_lut_tag(device_to_pcs, numpy.ascontiguousarray(device))
    if pcs == "Lab":
        if legacy_lab:
            measured *= numpy.float32(INTERNAL_LEGACY_LAB_SCALE)
        xyz = lab_to_xyz(measured, INTERNAL_D50_WHITE)
    else:
        xyz = measured
    black = tuple(float(value) for value in xyz[0][:3])
    if not all(0.0 < value < 0.5 for value in black):
        # A black point at or below the origin leaves compensation as a no-op,
        # and an implausibly light one means the tables disagree; skip both.
        return None
    return (black[0], black[1], black[2])


def internal_evaluate_lut_tag(tag: IccLutTag, values: ColorSamples) -> ColorSamples:
    """Run samples through one parsed LUT tag: input curves, CLUT, output curves.

    The tag's matrix is deliberately skipped. It only applies to an XYZ-encoded
    PCS input, and the one caller feeds either Lab or the XYZ origin, which the
    matrix maps to itself.
    """
    input_tables = tuple(internal_curve_table_arrays(table) for table in tag["input_tables"])
    curved = numpy.column_stack(
        [
            numpy.interp(values[:, index], axis, table)
            for index, (axis, table) in enumerate(input_tables)
        ]
    ).astype(numpy.float32)
    clut = interpolate_lut_array(
        internal_readonly_float32(tag["clut"]),
        tag["grid_points"],
        curved,
    )
    output_tables = tuple(internal_curve_table_arrays(table) for table in tag["output_tables"])
    return numpy.column_stack(
        [
            numpy.interp(clut[:, index], axis, table)
            for index, (axis, table) in enumerate(output_tables)
        ]
    ).astype(numpy.float32)


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


def select_icc_lut_tag(tags: dict[bytes, bytes], direction: bytes) -> bytes | None:
    """Pick a LUT tag, preferring the relative colorimetric one.

    PDF 32000-1 8.6.5.8 makes RelativeColorimetric the default rendering intent,
    which is a different table from the one the profile header nominates -- and
    the difference is not cosmetic. A perceptual table black-points its output,
    reporting L* = 0 for the darkest ink mix, so black point detection reads a
    profile that already reaches true black and compensation silently does
    nothing. The relative colorimetric table reports the ink as the dark grey it
    measures, which is the number compensation needs.
    """
    for suffix in (b"1", b"0", b"2"):
        payload = tags.get(direction + suffix)
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
        "sample_bytes": 1 if tag_type == b"mft1" else 2,
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
    if rows < 0 or columns < 0:
        return None, offset
    end = offset + rows * columns * sample_bytes
    if end > len(payload):
        return None, offset
    # A press profile's colour table runs to tens of thousands of grid nodes, so
    # decode the whole block at once rather than a sample at a time.
    dtype = numpy.dtype(numpy.uint8) if sample_bytes == 1 else numpy.dtype(">u2")
    scale = 255.0 if sample_bytes == 1 else 65535.0
    samples = numpy.frombuffer(payload[offset:end], dtype=dtype)
    grid = (samples.astype(numpy.float64) / scale).reshape(rows, columns)
    return tuple(map(tuple, grid.tolist())), end


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
