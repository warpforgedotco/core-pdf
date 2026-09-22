# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any, ClassVar, Self

import imagecodecs
import numpy

from core_pdf.impl.records import Record
from core_pdf_spec.s_08_graphics.color_rendering import (
    DEFAULT_COLOR_RENDERING,
    ColorRendering,
    use_black_point_compensation,
)

frozen_setattr = object.__setattr__


ByteSamples = numpy.ndarray[Any, numpy.dtype[numpy.uint8]]


class IccProfileError(ValueError):
    pass


class IccSampleError(ValueError):
    pass


INTENT_CODES = {
    "Perceptual": 0,
    "RelativeColorimetric": 1,
    "Saturation": 2,
    "AbsoluteColorimetric": 3,
}


def cms_options(rendering: ColorRendering) -> tuple[int, int]:
    flags = 0x0100 | (0x2000 if use_black_point_compensation(rendering, default=True) else 0)
    return INTENT_CODES[rendering.intent], flags


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
def srgb_profile() -> bytes:
    return bytes(imagecodecs.cms_profile("srgb"))


class IccTransform(Record):
    __slots__ = ("profile", "color_space", "input_channels")

    profile: bytes
    color_space: str
    input_channels: int

    __fields__: ClassVar[tuple[str, ...]] = ("profile", "color_space", "input_channels")
    __match_args__ = ("profile", "color_space", "input_channels")

    def __init__(self, profile: bytes, color_space: str, input_channels: int) -> None:
        frozen_setattr(self, "profile", profile)
        frozen_setattr(self, "color_space", color_space)
        frozen_setattr(self, "input_channels", input_channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"profile={self.profile!r}, "
            f"color_space={self.color_space!r}, "
            f"input_channels={self.input_channels!r}"
            ")"
        )

    def __replace__(self, /, **changes: Any) -> Self:
        profile = changes.pop("profile", self.profile)
        color_space = changes.pop("color_space", self.color_space)
        input_channels = changes.pop("input_channels", self.input_channels)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(profile, color_space, input_channels)

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
        return transform(self, samples, rendering)


def transform(
    transform: IccTransform,
    samples: numpy.ndarray[Any, Any],
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    rows, channels = samples.shape
    if rows == 0:
        return numpy.empty((0, 3), dtype=numpy.uint8)
    intent, flags = cms_options(rendering)
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(samples).reshape(rows, 1, channels),
            transform.profile,
            srgb_profile(),
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
