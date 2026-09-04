# SPDX-License-Identifier: AGPL-3.0-only
"""ICC colour management, delegated to Little-CMS through imagecodecs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import imagecodecs
import numpy

ColorSamples = numpy.ndarray[Any, numpy.dtype[numpy.float32]]
ByteSamples = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


class IccProfileError(ValueError):
    """Raised when an ICC profile cannot be converted by core-color."""


class IccSampleError(ValueError):
    """Raised when sample arrays do not match an ICC transform."""


# Rendering intent 1 is relative colorimetric (ICC.1:2010, 6.1.11), which with
# black point compensation is what the previous in-tree converter implemented:
# it walked the A2B tag, then scaled the result off the profile's detected
# black point. Across the whole 8-bit CMYK cube this pairing reproduces that
# converter to a maximum of 2/255 per channel, mean 0.22.
INTERNAL_RENDERING_INTENT = 1

# lcms2.h flags. cmsFLAGS_NOOPTIMIZE is not a quality trade here, it is the
# opposite: it stops lcms precomputing a device-link LUT and evaluates the full
# pipeline per sample instead, which is both more faithful (without it the same
# cube deviates by up to 21/255) and vastly cheaper to set up -- 1.7ms against
# 18ms for the 2.7MB SWOP profile. Setup cost dominates because imagecodecs
# exposes no reusable transform handle, so every call rebuilds the transform;
# the single-colour call sites in render/ would otherwise pay 18ms each. The
# cost is ~6x more per sample, which only overtakes the LUT past ~150k samples.
INTERNAL_BLACK_POINT_COMPENSATION = 0x2000
INTERNAL_NO_OPTIMIZE = 0x0100
INTERNAL_TRANSFORM_FLAGS = INTERNAL_BLACK_POINT_COMPENSATION | INTERNAL_NO_OPTIMIZE

# ICC data colour space signatures (ICC.1:2010, Table 19) as lcms reports them,
# mapped to the names the colour-space code above this module already uses.
INTERNAL_COLOR_SPACE_NAMES = {
    "gray": "GRAY",
    "rgb": "RGB",
    "cmyk": "CMYK",
    "xyz": "XYZ",
}
INTERNAL_PCS_NAMES = {"xyz": "XYZ", "lab": "Lab"}
INTERNAL_ALTERNATE_COLOR_SPACES = {
    "GRAY": "DeviceGray",
    "RGB": "DeviceRGB",
    "CMYK": "DeviceCMYK",
}

# lcms floating-point buffers carry each space in its own natural units, not a
# normalized [0, 1]: CMYK is ink percentage. Callers hand us [0, 1] throughout,
# so the input is rescaled and the sRGB result -- already [0, 1], though lcms
# lets it run slightly outside on out-of-gamut input -- is clamped.
INTERNAL_FLOAT_INPUT_SCALE = {"CMYK": numpy.float32(100.0)}

# Deduplicating a byte batch costs one sort, and pays for itself many times over
# on real images: a CMYK page photograph runs around 10% distinct colours, so
# the transform shrinks by an order of magnitude. Below the row floor the sort
# is not worth setting up, and above the ratio floor the batch is close enough
# to all-distinct that deduplicating would be pure overhead.
INTERNAL_DEDUPLICATE_MIN_ROWS = 4096
INTERNAL_DEDUPLICATE_MIN_RATIO = 2


@cache
def internal_srgb_profile() -> bytes:
    """Return the sRGB profile every transform in this module converts to."""
    return bytes(imagecodecs.cms_profile("srgb"))


@dataclass(frozen=True, slots=True, eq=False)
class IccTransform:
    """A compiled ICC transform from profile samples to sRGB."""

    profile: bytes
    color_space: str
    pcs: str
    input_channels: int
    internal_cms_color_space: str

    @property
    def output_channels(self) -> int:
        return 3

    @property
    def alternate_color_space(self) -> str:
        return INTERNAL_ALTERNATE_COLOR_SPACES.get(self.color_space, "DeviceRGB")

    def apply(self, samples: ColorSamples) -> ColorSamples:
        """Convert an (n, channels) block of [0, 1] samples to [0, 1] sRGB."""
        validate_color_samples(samples, self.input_channels)
        scale = INTERNAL_FLOAT_INPUT_SCALE.get(self.color_space)
        source = samples if scale is None else samples * scale
        result = internal_transform(self, source, numpy.float32)
        return numpy.clip(result, 0.0, 1.0, out=result)

    def apply_uint8(self, samples: ByteSamples) -> ByteSamples:
        """Convert an (n, channels) block of 8-bit device samples to 8-bit sRGB.

        Byte inputs are the overwhelmingly common case -- every image sample
        arrives this way -- and a byte row is small enough to deduplicate, so
        only the distinct colours in a batch need to reach lcms at all.
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
        return internal_transform(self, samples, numpy.uint8)


def internal_transform(
    transform: IccTransform,
    samples: numpy.ndarray[Any, Any],
    dtype: type[numpy.uint8] | type[numpy.float32],
) -> numpy.ndarray[Any, Any]:
    """Run one batch through lcms, as an (n, 1, channels) single-column image.

    Whole batches go in one call. The converter this replaced split them into
    256k-row blocks because its multilinear interpolation materialised
    2**input_channels intermediate (rows, 3) float32 arrays at once -- sixteen
    of them for CMYK -- so a full-page scan in one shot cost hundreds of
    megabytes. lcms streams sample by sample in C and holds no such
    intermediates, and blocking here would instead rebuild the transform once
    per block, which is the dominant cost of a conversion.
    """
    rows, channels = samples.shape
    if rows == 0:
        return numpy.empty((0, 3), dtype=dtype)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(samples).reshape(rows, 1, channels),
            transform.profile,
            internal_srgb_profile(),
            colorspace=transform.internal_cms_color_space,
            outcolorspace="rgb",
            outdtype=dtype,
            intent=INTERNAL_RENDERING_INTENT,
            flags=INTERNAL_TRANSFORM_FLAGS,
        )
    except imagecodecs.CmsError as error:
        raise IccProfileError("ICC profile cannot be converted to sRGB") from error
    return numpy.asarray(converted, dtype=dtype).reshape(rows, 3)


def parse_icc_transform(profile: bytes) -> IccTransform:
    """Compile an embedded ICC profile into a transform to sRGB."""
    profile_bytes = bytes(profile)
    try:
        info = imagecodecs.cms_info(profile_bytes)
    except imagecodecs.CmsError as error:
        raise IccProfileError("unsupported or malformed ICC profile") from error
    cms_color_space = str(info.get("colorspace") or "")
    color_space = INTERNAL_COLOR_SPACE_NAMES.get(cms_color_space)
    pcs = INTERNAL_PCS_NAMES.get(str(info.get("pcs") or ""))
    channels = int(info.get("channels") or 0)
    if color_space is None or pcs is None or channels < 1:
        raise IccProfileError("unsupported or malformed ICC profile")
    return IccTransform(
        profile=profile_bytes,
        color_space=color_space,
        pcs=pcs,
        input_channels=channels,
        internal_cms_color_space=cms_color_space,
    )


def validate_color_samples(samples: ColorSamples, channels: int) -> None:
    if not isinstance(samples, numpy.ndarray):
        raise IccSampleError("samples must be a NumPy array")
    if samples.dtype != numpy.dtype(numpy.float32):
        raise IccSampleError("samples must have float32 dtype")
    if samples.ndim != 2 or samples.shape[1] != channels:
        raise IccSampleError(f"samples must have shape (count, {channels})")
    if not samples.flags.c_contiguous:
        raise IccSampleError("samples must be C-contiguous")


def internal_distinct_byte_rows(
    samples: ByteSamples,
) -> tuple[ByteSamples, numpy.ndarray[Any, Any]]:
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
