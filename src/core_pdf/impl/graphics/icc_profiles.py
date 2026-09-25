# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any, ClassVar

import imagecodecs
import numpy

from core_pdf.impl.types import Record, frozen_setattr
from core_pdf_cythonized import distinct_uint16_rows, gather_uint8_rows
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


# A colour operand reaches lcms as one sample, and lcms builds a transform from
# the two profiles on every call: 0.4 to 2.9 ms for a single colour, while the
# conversion itself is nothing. A page sets the same few colours over and over
# -- one corpus page 768 times across 6 colours, another 286 across 31 -- so a
# small sample set is converted once per profile, rendering and value. Images
# are never cached: each is one call over all its pixels, and rarely repeats.
SMALL_SAMPLE_ROWS = 16


def transform(
    transform: IccTransform,
    samples: numpy.ndarray[Any, Any],
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> ByteSamples:
    rows, channels = samples.shape
    if rows == 0:
        return numpy.empty((0, 3), dtype=numpy.uint8)
    intent, flags = cms_options(rendering)
    contiguous = numpy.ascontiguousarray(samples)
    if rows <= SMALL_SAMPLE_ROWS:
        converted = cached_cms_transform(
            transform.profile,
            transform.color_space,
            intent,
            flags,
            contiguous.tobytes(),
            rows,
            channels,
        )
        # A copy, so a caller that writes into the result cannot reach the cache.
        return numpy.frombuffer(converted, dtype=numpy.uint8).reshape(rows, 3).copy()
    if rows > DISTINCT_ROWS_MINIMUM and channels <= 4:
        # lcms converts each pixel on its own, so an image converted by its
        # distinct colours and scattered back is the same image. A photograph
        # uses few of the colours it could -- 2.5% of the pixels on one corpus
        # page -- so this skips most of lcms's work; past a quarter distinct,
        # the scatter would cost more than it saves.
        found = distinct_uint16_rows(contiguous, rows // 4)
        if found is not None:
            distinct, inverse = found
            distinct_colors = cms_transform(
                transform.profile, transform.color_space, intent, flags, distinct
            )
            return gather_uint8_rows(numpy.ascontiguousarray(distinct_colors), inverse)
    return cms_transform(transform.profile, transform.color_space, intent, flags, contiguous)


# Below this many pixels an image goes straight to lcms.
DISTINCT_ROWS_MINIMUM = 1 << 16


@lru_cache(maxsize=4096)
def cached_cms_transform(
    profile: bytes,
    color_space: str,
    intent: int,
    flags: int,
    samples: bytes,
    rows: int,
    channels: int,
) -> bytes:
    """cms_transform for a few samples, keyed on everything its output depends on."""
    values = numpy.frombuffer(samples, dtype=numpy.uint16).reshape(rows, channels)
    return cms_transform(profile, color_space, intent, flags, values).tobytes()


def cms_transform(
    profile: bytes,
    color_space: str,
    intent: int,
    flags: int,
    samples: numpy.ndarray[Any, Any],
) -> ByteSamples:
    rows, channels = samples.shape
    try:
        converted = imagecodecs.cms_transform(
            samples.reshape(rows, 1, channels),
            profile,
            srgb_profile(),
            colorspace=color_space.lower(),
            outcolorspace="rgb",
            outdtype=numpy.uint8,
            intent=intent,
            flags=flags,
        )
    except imagecodecs.CmsError as error:
        raise IccProfileError("ICC profile cannot be converted to sRGB") from error
    return numpy.asarray(converted, dtype=numpy.uint8).reshape(rows, 3)


def parse_icc_transform(profile: bytes) -> IccTransform:
    return parse_icc_transform_cached(bytes(profile))


@lru_cache(maxsize=32)
def parse_icc_transform_cached(profile: bytes) -> IccTransform:
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
