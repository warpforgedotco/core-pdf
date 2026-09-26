# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
from importlib import resources

import numpy

from core_pdf.impl.graphics.icc_profiles import (
    ByteSamples,
    IccProfileError,
    IccSampleError,
    IccTransform,
    parse_icc_transform,
)
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering

INTERNAL_DEFAULT_CMYK_PROFILE = "SWOP2006_Coated5v2.icc"


@cache
def default_cmyk_transform() -> IccTransform | None:
    try:
        profile = (
            resources.files("core_pdf._vendor.icc")
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
