# SPDX-License-Identifier: AGPL-3.0-only
"""Built-in profiles for the device colour spaces PDF leaves undefined."""

from __future__ import annotations

from functools import cache, lru_cache
from importlib import resources

import numpy

from core_pdf.impl.spec.s_08_graphics.icc_profiles import (
    ByteSamples,
    IccProfileError,
    IccSampleError,
    IccTransform,
    parse_icc_transform,
)

INTERNAL_DEFAULT_CMYK_PROFILE = "SWOP2006_Coated5v2.icc"


@cache
def default_cmyk_transform() -> IccTransform | None:
    """Return the built-in DeviceCMYK profile, or None if it cannot be used.

    PDF 32000-1 makes DeviceCMYK device-dependent and defines no conversion to
    RGB, so every renderer supplies its own default profile; ours is vendored in
    `_vendor/icc/`, and `_vendor/icc/README.md` records where it came from.
    Returning None rather than raising keeps a damaged or stripped install
    rendering -- the callers below fall back to the naive ink formula.

    Cached because the profile is 2.7MB: every DeviceCMYK colour and image
    sample funnels through here, and re-reading and re-parsing it per call cost
    more than the conversion itself.
    """
    try:
        profile = (
            resources.files("core_pdf._vendor.icc")
            .joinpath(INTERNAL_DEFAULT_CMYK_PROFILE)
            .read_bytes()
        )
    except (OSError, ModuleNotFoundError):
        return None
    try:
        transform = parse_icc_transform(profile)
    except IccProfileError:
        return None
    if transform.color_space != "CMYK" or transform.input_channels != 4:
        return None
    return transform


def cmyk_bytes_to_srgb(samples: ByteSamples) -> ByteSamples:
    """Convert an (n, 4) block of 8-bit DeviceCMYK samples to (n, 3) sRGB."""
    transform = default_cmyk_transform()
    if transform is not None:
        try:
            return transform.apply_uint8(samples)
        except IccSampleError:
            pass
    # Keep the uncalibrated ink formula as a direct fallback for a damaged or
    # stripped installation. There is no retained channel table to manage.
    inks = samples[:, :3].astype(numpy.float64) / 255.0
    black = samples[:, 3:].astype(numpy.float64) / 255.0
    return numpy.floor(255.0 * (1.0 - inks) * (1.0 - black)).astype(numpy.uint8)


def cmyk_floats_to_srgb(
    cyan: float,
    magenta: float,
    yellow: float,
    black: float,
) -> tuple[int, int, int]:
    """Convert one DeviceCMYK colour given as four floats in [0, 1] to sRGB."""
    return internal_cmyk_bytes_to_srgb(
        internal_component_byte(cyan),
        internal_component_byte(magenta),
        internal_component_byte(yellow),
        internal_component_byte(black),
    )


# Keyed on the quantized inks, not the floats they came from. The hot caller is
# `render/patterns`, which asks for one colour per pixel of an axial or radial
# shading: the floats vary continuously, so a float key misses on essentially
# every pixel and pays a fresh lcms transform build for each, while the byte
# key collapses a whole gradient ramp to a few hundred entries.
@lru_cache(maxsize=8192)
def internal_cmyk_bytes_to_srgb(
    cyan: int,
    magenta: int,
    yellow: int,
    black: int,
) -> tuple[int, int, int]:
    sample = numpy.asarray([[cyan, magenta, yellow, black]], dtype=numpy.uint8)
    red, green, blue = cmyk_bytes_to_srgb(sample)[0]
    return int(red), int(green), int(blue)


def internal_component_byte(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))
