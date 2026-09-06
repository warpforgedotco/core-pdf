# SPDX-License-Identifier: AGPL-3.0-only
"""ICC colour management, delegated to Little-CMS through imagecodecs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import imagecodecs
import numpy

ByteSamples = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


class IccProfileError(ValueError):
    """Raised when an ICC profile cannot be converted by core-color."""


class IccSampleError(ValueError):
    """Raised when sample arrays do not match an ICC transform."""


# Rendering intent 1 is relative colorimetric (ICC.1:2010, 6.1.11). With black
# point compensation it reproduces the in-tree converter this replaced to
# within 2/255 per channel across the whole 8-bit CMYK cube.
INTERNAL_RENDERING_INTENT = 1

# cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_NOOPTIMIZE (lcms2.h). NOOPTIMIZE is
# not a quality trade here but the opposite: it stops lcms precomputing a
# device-link LUT, which is both more faithful (with the LUT the same cube
# deviates by up to 21/255) and an order of magnitude cheaper to set up. Setup
# dominates because imagecodecs exposes no reusable transform handle, so every
# call rebuilds the transform.
INTERNAL_TRANSFORM_FLAGS = 0x2000 | 0x0100

# ICC data colour space signatures (ICC.1:2010, Table 19) as lcms reports them,
# mapped to the names the colour-space code above this module uses. Lowercasing
# a name recovers the lcms spelling, which is what `cms_transform` wants.
INTERNAL_COLOR_SPACE_NAMES = {
    "gray": "GRAY",
    "rgb": "RGB",
    "cmyk": "CMYK",
    "xyz": "XYZ",
}
INTERNAL_PCS_NAMES = {"xyz", "lab"}
INTERNAL_ALTERNATE_COLOR_SPACES = {
    "GRAY": "DeviceGray",
    "RGB": "DeviceRGB",
    "CMYK": "DeviceCMYK",
}

# Real images repeat colours heavily -- a CMYK page photograph runs around 10%
# distinct -- so deduplicating a large batch shrinks the transform by an order
# of magnitude. Below this floor the sort is not worth setting up.
INTERNAL_DEDUPLICATE_MIN_ROWS = 4096

# Once the sort is paid for, scattering the result back costs about a twelfth
# of what transforming the duplicate rows would, so deduplicating pays until
# the batch is within a few percent of all-distinct. Measured on this profile:
# ~0.19us per row transformed against ~0.015us per row gathered.
INTERNAL_DEDUPLICATE_MAX_DISTINCT = 0.9


@cache
def internal_srgb_profile() -> bytes:
    """Return the sRGB profile every transform in this module converts to."""
    return bytes(imagecodecs.cms_profile("srgb"))


@dataclass(frozen=True, slots=True, eq=False)
class IccTransform:
    """An ICC profile and the sRGB conversion it defines."""

    profile: bytes
    color_space: str
    input_channels: int

    @property
    def alternate_color_space(self) -> str:
        return INTERNAL_ALTERNATE_COLOR_SPACES.get(self.color_space, "DeviceRGB")

    def apply_uint8(self, samples: ByteSamples) -> ByteSamples:
        """Convert an (n, channels) block of 8-bit device samples to 8-bit sRGB.

        Distinct colours are the only ones worth sending to lcms, so a batch
        large enough to be worth sorting is deduplicated first and the result
        scattered back.
        """
        channels = self.input_channels
        if samples.dtype != numpy.dtype(numpy.uint8):
            raise IccSampleError("samples must have uint8 dtype")
        if samples.ndim != 2 or samples.shape[1] != channels:
            raise IccSampleError(f"samples must have shape (count, {channels})")
        if channels <= 4 and len(samples) > INTERNAL_DEDUPLICATE_MIN_ROWS:
            distinct, inverse = internal_distinct_byte_rows(samples)
            # The sort is already paid for by this point, so the test only has
            # to clear the gather that is still ahead, not the sort behind.
            if len(distinct) < len(samples) * INTERNAL_DEDUPLICATE_MAX_DISTINCT:
                return internal_transform(self, distinct)[inverse]
        return internal_transform(self, samples)


def internal_transform(transform: IccTransform, samples: ByteSamples) -> ByteSamples:
    """Run one batch through lcms, as an (n, 1, channels) single-column image.

    Whole batches go in one call: lcms streams sample by sample in C, and
    blocking would instead rebuild the transform once per block, which is the
    dominant cost of a conversion.
    """
    rows, channels = samples.shape
    if rows == 0:
        return numpy.empty((0, 3), dtype=numpy.uint8)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(samples).reshape(rows, 1, channels),
            transform.profile,
            internal_srgb_profile(),
            colorspace=transform.color_space.lower(),
            outcolorspace="rgb",
            outdtype=numpy.uint8,
            intent=INTERNAL_RENDERING_INTENT,
            flags=INTERNAL_TRANSFORM_FLAGS,
        )
    except imagecodecs.CmsError as error:
        raise IccProfileError("ICC profile cannot be converted to sRGB") from error
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(rows, 3)


def parse_icc_transform(profile: bytes) -> IccTransform:
    """Compile an embedded ICC profile into a transform to sRGB."""
    return internal_parse_icc_transform(bytes(profile))


# Every ICCBased image re-parses its profile just to read the alternate colour
# space, and the built-in CMYK profile is 2.7MB. Bounded because the cached
# value retains the profile bytes, which `cms_transform` needs on every call.
@lru_cache(maxsize=32)
def internal_parse_icc_transform(profile: bytes) -> IccTransform:
    try:
        info = imagecodecs.cms_info(profile)
    except imagecodecs.CmsError as error:
        raise IccProfileError("unsupported or malformed ICC profile") from error
    color_space = INTERNAL_COLOR_SPACE_NAMES.get(str(info.get("colorspace") or ""))
    channels = int(info.get("channels") or 0)
    if color_space is None or channels < 1:
        raise IccProfileError("unsupported or malformed ICC profile")
    if str(info.get("pcs") or "") not in INTERNAL_PCS_NAMES:
        raise IccProfileError("unsupported or malformed ICC profile")
    return IccTransform(profile=profile, color_space=color_space, input_channels=channels)


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
