# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
lazy from importlib import resources

lazy import imagecodecs
lazy import numpy

from core_pdf.impl.graphics.icc_profiles import (
    ByteSamples,
    IccProfileError,
    IccSampleError,
    IccTransform,
    cms_options,
    parse_icc_transform,
    srgb_profile,
)
from core_pdf.impl.spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering

INTERNAL_DEFAULT_CMYK_PROFILE = "SWOP2006_Coated5v2.icc"


@cache
def default_cmyk_transform() -> IccTransform | None:
    try:
        profile = (
            resources.files("core_pdf.impl.data.icc")
            .joinpath(INTERNAL_DEFAULT_CMYK_PROFILE)
            .read_bytes()
        )
    except OSError, ModuleNotFoundError:
        return None
    try:
        transform = parse_icc_transform(profile)
    except IccProfileError:
        return None
    if transform.color_space != "CMYK" or transform.input_channels != 4:
        return None
    return transform


def cmyk_profile_transform(profile: bytes | None) -> IccTransform | None:
    """The transform of a CMYK ICC profile, or with None (or a profile that is not one) the
    built-in DeviceCMYK profile's."""
    if profile is not None:
        try:
            transform = parse_icc_transform(profile)
        except IccProfileError:
            transform = None
        if transform is not None and transform.color_space == "CMYK":
            return transform
    return default_cmyk_transform()


def cmyk_bytes_to_srgb(
    samples: ByteSamples,
    profile: bytes | None = None,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    """(n, 4) CMYK bytes as (n, 3) sRGB bytes, through profile or the built-in DeviceCMYK
    profile; the same bytes cmyk_byte_tuple_to_srgb gives a DeviceCMYK colour."""
    if profile is None:
        return cmyk_components_to_srgb(samples.astype(numpy.float64) / 255.0, rendering=rendering)
    transform = cmyk_profile_transform(profile)
    if transform is not None:
        try:
            words = numpy.rint(samples.astype(numpy.float64) / 255.0 * 65535).astype(numpy.uint16)
            return transform.apply_uint16(words, rendering=rendering)
        except IccProfileError, IccSampleError:
            pass
    return cmyk_components_to_srgb(samples.astype(numpy.float64) / 255.0, rendering=rendering)


def srgb_bytes_to_cmyk(
    samples: ByteSamples,
    profile: bytes | None = None,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    """(n, 3) sRGB bytes as (n, 4) CMYK bytes: through profile, or the built-in DeviceCMYK
    profile, the inverse of cmyk_bytes_to_srgb, each distinct colour once; without a profile,
    the classic conversion of ISO 32000-2 10.4.2.4 with black generation and undercolour
    removal the identity."""
    rows = numpy.ascontiguousarray(samples, dtype=numpy.uint8).reshape(-1, 3)
    if len(rows) == 0:
        return numpy.empty((0, 4), dtype=numpy.uint8)
    packed = (
        rows[:, 0].astype(numpy.uint32) << 16
        | rows[:, 1].astype(numpy.uint32) << 8
        | rows[:, 2].astype(numpy.uint32)
    )
    distinct, inverse = numpy.unique(packed, return_inverse=True)
    colors = numpy.column_stack(
        ((distinct >> 16) & 255, (distinct >> 8) & 255, distinct & 255)
    ).astype(numpy.uint8)
    transform = cmyk_profile_transform(profile)
    converted = None
    if transform is not None:
        intent, flags = cms_options(rendering)
        try:
            converted = numpy.asarray(
                imagecodecs.cms_transform(
                    colors.reshape(-1, 1, 3),
                    srgb_profile(),
                    transform.profile,
                    colorspace="rgb",
                    outcolorspace="cmyk",
                    outdtype=numpy.uint8,
                    intent=intent,
                    flags=flags,
                ),
                dtype=numpy.uint8,
            ).reshape(-1, 4)
        except imagecodecs.CmsError:
            converted = None
    if converted is None:
        cmy = 255 - colors.astype(numpy.int16)
        black = cmy.min(axis=1, keepdims=True)
        converted = numpy.column_stack((cmy - black, black)).astype(numpy.uint8)
    return converted[inverse.reshape(-1)]


def cmyk_components_to_srgb(
    values: numpy.ndarray, *, rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> ByteSamples:
    values = numpy.clip(values, 0, 1)
    transform = default_cmyk_transform()
    if transform is not None:
        try:
            words = numpy.rint(values * 65535).astype(numpy.uint16)
            return transform.apply_uint16(words, rendering=rendering)
        except IccProfileError, IccSampleError:
            pass
    return numpy.rint(255 * (1 - values[:, :3]) * (1 - values[:, 3:])).astype(numpy.uint8)


def cmyk_floats_to_srgb(
    cyan: float,
    magenta: float,
    yellow: float,
    black: float,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[int, int, int]:
    return cmyk_byte_tuple_to_srgb(
        component_byte(cyan),
        component_byte(magenta),
        component_byte(yellow),
        component_byte(black),
        rendering,
    )


@lru_cache(maxsize=8192)
def cmyk_byte_tuple_to_srgb(
    cyan: int,
    magenta: int,
    yellow: int,
    black: int,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[int, int, int]:
    sample = numpy.asarray([[cyan, magenta, yellow, black]], dtype=numpy.uint8)
    components = sample.astype(numpy.float64) / 255
    red, green, blue = cmyk_components_to_srgb(components, rendering=rendering)[0]
    return int(red), int(green), int(blue)


def component_byte(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))
