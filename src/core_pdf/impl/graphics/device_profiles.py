# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
lazy from importlib import resources

lazy import imagecodecs
lazy import numpy

from core_pdf.impl.graphics.icc_profiles import (
    INTENT_CODES,
    ByteSamples,
    IccProfileError,
    IccSampleError,
    IccTransform,
    cms_options,
    parse_icc_transform,
    srgb_profile,
)
from core_pdf.impl.spec.citations import cites
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


def blending_cmyk_profile(profile: bytes | None) -> bytes | None:
    """The ICC profile whose CMYK a CMYK blending space holds: profile, the space's own, or
    for DeviceCMYK (None) the built-in DeviceCMYK profile; None if neither can be read."""
    transform = cmyk_profile_transform(profile)
    return None if transform is None else transform.profile


@cites(
    "ISO 32000-2",
    "11.7.2",
    introduced="1.4",
    rule="a colour not in its group's colour space converts to it",
)
def convert_cmyk_words(
    words: numpy.ndarray,
    source: bytes | None,
    target: bytes | None,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> numpy.ndarray:
    """(n, 4) uint16 CMYK of the ICC profile source as the CMYK of target, the profile of a
    CMYK blending space (None: the built-in DeviceCMYK profile), through the PCS. The words
    are returned as they are when source is target, or either cannot be read as a CMYK
    profile."""
    if source is None:
        return words
    destination = blending_cmyk_profile(target)
    if destination is None or source is destination or source == destination:
        return words
    try:
        if parse_icc_transform(source).color_space != "CMYK":
            return words
    except IccProfileError:
        return words
    intent, flags = cms_options(rendering)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(words, dtype=numpy.uint16).reshape(-1, 1, 4),
            source,
            destination,
            colorspace="cmyk",
            outcolorspace="cmyk",
            outdtype=numpy.uint16,
            intent=intent,
            flags=flags,
        )
    except imagecodecs.CmsError:
        return words
    return numpy.asarray(converted, dtype=numpy.uint16).reshape(-1, 4)


def convert_cmyk_bytes(
    samples: ByteSamples,
    source: bytes | None,
    target: bytes | None,
    *,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    """convert_cmyk_words of (n, 4) CMYK bytes, each distinct colour once."""
    rows = numpy.ascontiguousarray(samples, dtype=numpy.uint8).reshape(-1, 4)
    if source is None or not len(rows):
        return rows
    distinct, inverse = numpy.unique(rows.view(numpy.uint32).reshape(-1), return_inverse=True)
    words = distinct.view(numpy.uint8).reshape(-1, 4).astype(numpy.uint16) * 257
    converted = convert_cmyk_words(words, source, target, rendering=rendering)
    if converted is words:
        return rows
    found = numpy.rint(converted.astype(numpy.float64) / 257.0).astype(numpy.uint8)
    return found[inverse.reshape(-1)]


@cites(
    "ISO 32000-2",
    "11.5.3",
    introduced="1.4",
    rule="a CIE-based group's luminosity is the CIE 1931 Y of its colour",
)
def cmyk_bytes_luminance(samples: ByteSamples, profile: bytes | None) -> numpy.ndarray:
    """The CIE 1931 Y, relative to the medium's white, of each (n, 4) CMYK byte row of the
    ICC profile, through the profile's colorimetric transform to the PCS (no black point
    compensation, so the profile's darkest black keeps its Y), each distinct colour once."""
    rows = numpy.ascontiguousarray(samples, dtype=numpy.uint8).reshape(-1, 4)
    transform = cmyk_profile_transform(profile)
    if transform is None:
        raise IccProfileError("no CMYK profile")
    distinct, inverse = numpy.unique(rows.view(numpy.uint32).reshape(-1), return_inverse=True)
    try:
        xyz = imagecodecs.cms_transform(
            numpy.ascontiguousarray(distinct.view(numpy.uint8).reshape(-1, 1, 4)),
            transform.profile,
            xyz_profile(),
            colorspace="cmyk",
            outcolorspace="xyz",
            outdtype=numpy.float64,
            intent=INTENT_CODES["RelativeColorimetric"],
            flags=0x0100,
        )
    except imagecodecs.CmsError as error:
        raise IccProfileError("ICC profile cannot be converted to XYZ") from error
    luminance = numpy.asarray(xyz, dtype=numpy.float64).reshape(-1, 3)[:, 1]
    return luminance[inverse.reshape(-1)]


@cache
def xyz_profile() -> bytes:
    return bytes(imagecodecs.cms_profile("xyz"))


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
