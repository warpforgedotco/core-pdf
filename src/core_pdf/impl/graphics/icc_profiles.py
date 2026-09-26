# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import cache, lru_cache
from typing import Any, ClassVar

import imagecodecs
import numpy

from core_pdf.impl.graphics.codec_backends import thread_count
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


# lcms builds a transform from the two profiles on every call: 0.4 to 2.9 ms
# even for a single colour, while the conversion itself is nothing. A page
# sets the same few colours over and over -- one corpus page 768 times across
# 6 colours, another 286 across 31 -- and its small images repeat colours
# too: 104 of the 176 conversions PyMuPDF test_3806 makes, of spot-colour
# images through one CMYK profile, hold no colour an earlier one did not. lcms
# converts each sample row on its own, so every row converted is kept, per
# profile, colour space, intent and flags, and a call of up to MEMO_ROWS rows
# converts only the rows not yet kept, in one call. Larger images go to lcms
# whole, by their distinct colours.
MEMO_ROWS = 4096
# Rows kept per transform, and transforms kept, before starting over.
MEMO_LIMIT = 1 << 16
MEMO_TRANSFORMS = 32

type RowMemo = dict[bytes, bytes]
row_memos: dict[tuple[bytes, str, int, int], RowMemo] = {}


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
    if rows <= MEMO_ROWS and contiguous.dtype == numpy.uint16:
        return memoized_cms_transform(
            transform.profile, transform.color_space, intent, flags, contiguous
        )
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


def memoized_cms_transform(
    profile: bytes,
    color_space: str,
    intent: int,
    flags: int,
    samples: numpy.ndarray[Any, Any],
) -> ByteSamples:
    """cms_transform of uint16 `samples`, converting only the rows not converted before.

    Returns a new array the caller may write into. A conversion that fails
    keeps nothing, so it fails again the next time.
    """
    key = (profile, color_space, intent, flags)
    memo = row_memos.get(key)
    if memo is None:
        if len(row_memos) >= MEMO_TRANSFORMS:
            row_memos.clear()
        memo = row_memos[key] = {}
    rows, channels = samples.shape
    width = channels * 2
    raw = samples.tobytes()
    keys = [raw[start : start + width] for start in range(0, rows * width, width)]
    missing = [index for index, row in enumerate(keys) if row not in memo]
    if missing:
        converted = cms_transform(profile, color_space, intent, flags, samples[missing]).tobytes()
        for position, index in enumerate(missing):
            memo[keys[index]] = converted[3 * position : 3 * position + 3]
    result = numpy.frombuffer(bytearray(b"".join(map(memo.__getitem__, keys))), dtype=numpy.uint8)
    if len(memo) > MEMO_LIMIT:
        memo.clear()
    return result.reshape(rows, 3)


# lcms converts each sample row on its own and releases the GIL while it
# does, so a large conversion is split into row blocks converted on threads
# and joined: the same array. A 3.2-megapixel CMYK image through the 2.7 MB
# default profile took 593 ms in one call on PyMuPDF test_4466.
PARALLEL_ROWS = 1 << 18


def cms_thread_count() -> int:
    return thread_count("CORE_PDF_CMS_THREADS")


def cms_transform(
    profile: bytes,
    color_space: str,
    intent: int,
    flags: int,
    samples: numpy.ndarray[Any, Any],
) -> ByteSamples:
    rows = samples.shape[0]
    workers = cms_thread_count() if rows >= PARALLEL_ROWS else 1
    if workers > 1:
        blocks = numpy.array_split(samples, workers)
        with ThreadPoolExecutor(workers) as executor:
            converted = list(
                executor.map(
                    lambda block: cms_transform_block(profile, color_space, intent, flags, block),
                    blocks,
                )
            )
        return numpy.concatenate(converted)
    return cms_transform_block(profile, color_space, intent, flags, samples)


def cms_transform_block(
    profile: bytes,
    color_space: str,
    intent: int,
    flags: int,
    samples: numpy.ndarray[Any, Any],
) -> ByteSamples:
    rows, channels = samples.shape
    try:
        converted = imagecodecs.cms_transform(
            numpy.ascontiguousarray(samples).reshape(rows, 1, channels),
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
