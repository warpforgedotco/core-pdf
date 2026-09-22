# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import imagecodecs
import numpy

from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    use_black_point_compensation,
)

ByteSamples = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


class IccProfileError(ValueError):
    pass


class IccSampleError(ValueError):
    pass


internal_INTENT_CODES = {
    "Perceptual": 0,
    "RelativeColorimetric": 1,
    "Saturation": 2,
    "AbsoluteColorimetric": 3,
}


def internal_cms_options(rendering: ColorRendering) -> tuple[int, int]:
    flags = 0x0100 | (0x2000 if use_black_point_compensation(rendering, default=True) else 0)
    return internal_INTENT_CODES[rendering.intent], flags


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


@cache
def internal_srgb_profile() -> bytes:
    return bytes(imagecodecs.cms_profile("srgb"))


@dataclass(frozen=True, slots=True, eq=False)
class IccTransform:
    profile: bytes
    color_space: str
    input_channels: int

    @property
    def alternate_color_space(self) -> str:
        return INTERNAL_ALTERNATE_COLOR_SPACES.get(self.color_space, "DeviceRGB")

    def apply_uint16(
        self,
        samples: numpy.ndarray[Any, Any],
        *,
        rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    ) -> ByteSamples:
        if samples.dtype != numpy.dtype(numpy.uint16):
            raise IccSampleError("samples must have uint16 dtype")
        if samples.ndim != 2 or samples.shape[1] != self.input_channels:
            raise IccSampleError(f"samples must have shape (count, {self.input_channels})")
        return internal_transform(self, samples, rendering)


def internal_transform(
    transform: IccTransform,
    samples: numpy.ndarray[Any, Any],
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    rows, channels = samples.shape
    if rows == 0:
        return numpy.empty((0, 3), dtype=numpy.uint8)
    intent, flags = internal_cms_options(rendering)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(samples).reshape(rows, 1, channels),
            transform.profile,
            internal_srgb_profile(),
            colorspace=transform.color_space.lower(),
            outcolorspace="rgb",
            outdtype=numpy.uint8,
            intent=intent,
            flags=flags,
        )
    except imagecodecs.CmsError as error:
        raise IccProfileError("ICC profile cannot be converted to sRGB") from error
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(rows, 3)


def parse_icc_transform(profile: bytes) -> IccTransform:
    return internal_parse_icc_transform(bytes(profile))


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
