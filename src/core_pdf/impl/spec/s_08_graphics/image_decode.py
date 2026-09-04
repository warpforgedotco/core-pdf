# SPDX-License-Identifier: AGPL-3.0-only
"""Decode PDF image samples independently of downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy

from core_pdf.impl.runtime.array_views import readonly
from core_pdf.impl.spec.s_07_filters.errors import FilterError
from core_pdf.impl.spec.s_07_filters.models import DecodedImage
from core_pdf.impl.spec.s_07_filters.pipeline import (
    decode_stream_data,
    decode_stream_image_data,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_int
from core_pdf.impl.spec.s_08_graphics.color import (
    internal_convert_cmyk,
    internal_convert_image_data,
)


@dataclass(frozen=True, slots=True)
class DecodedRaster:
    data: bytes | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int


@dataclass(frozen=True, slots=True)
class ImageRaster:
    """Immutable canonical samples for one decoded image."""

    array: numpy.ndarray[Any, Any]
    color_model: str
    has_alpha: bool = False

    def __post_init__(self) -> None:
        array = numpy.asarray(self.array, dtype=numpy.uint8)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3 or array.shape[2] not in {1, 2, 3, 4}:
            raise ValueError("image raster must have one, two, three, or four channels")
        expected_alpha = array.shape[2] in {2, 4}
        if expected_alpha != self.has_alpha:
            raise ValueError("image raster alpha flag does not match its channel layout")
        if self.color_model not in {"gray", "rgb"}:
            raise ValueError("unsupported image raster color model")
        expected_channels = 1 if self.color_model == "gray" else 3
        if array.shape[2] not in {expected_channels, expected_channels + 1}:
            raise ValueError("image raster channel layout does not match its color model")
        object.__setattr__(self, "array", readonly(numpy.ascontiguousarray(array)))

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def channels(self) -> int:
        return int(self.array.shape[2])

    @property
    def stride(self) -> int:
        return int(self.array.strides[0])


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A decoded image plus its optional native-resolution soft mask."""

    raster: ImageRaster
    soft_mask: ImageRaster | None = None
    is_stencil: bool = False

    def __post_init__(self) -> None:
        soft_mask = self.soft_mask
        if soft_mask is not None and (soft_mask.color_model != "gray" or soft_mask.has_alpha):
            raise ValueError("prepared image soft mask must be grayscale without alpha")


@dataclass(frozen=True, slots=True)
class SoftMask:
    """The /SMask plane accompanying an image XObject.

    Carried as its own field rather than smuggled through the image's PDF
    dictionary: that dictionary is the real object dictionary and is exported
    verbatim to display consumers, so private keys in it leak downstream.
    """

    raw: bytes | memoryview
    dictionary: dict[Any, Any]


@dataclass(slots=True, eq=False)
class ImageSource:
    """Raw inputs and preparation logic for one embedded image source."""

    raw: bytes | memoryview
    dictionary: dict[Any, Any]
    soft_mask: SoftMask | None = field(default=None, kw_only=True)

    def prepare(self) -> PreparedImage | None:
        """Decode and return an immutable prepared image."""
        is_stencil = self.dictionary.get("ImageMask") is True
        decoded = (
            self.internal_decode_mask()
            if is_stencil
            else decode_pdf_image(self.raw, self.dictionary)
        )
        if decoded is None:
            return None
        array: numpy.ndarray[Any, Any]
        if isinstance(decoded.data, numpy.ndarray):
            decoded_array = numpy.asarray(decoded.data)
            array = decoded_array.reshape(decoded.height, decoded.width, decoded.channels)
        else:
            flat_array = numpy.frombuffer(decoded.data, dtype=numpy.uint8)
            array = flat_array.reshape(decoded.height, decoded.width, decoded.channels)
        color_model = "gray" if decoded.channels in {1, 2} else "rgb"
        raster = ImageRaster(
            array,
            color_model,
            has_alpha=decoded.channels in {2, 4},
        )
        if is_stencil:
            return PreparedImage(raster, is_stencil=True)
        soft_mask = self.internal_decode_soft_mask()
        if soft_mask is None:
            return PreparedImage(raster)
        return PreparedImage(
            self.internal_apply_soft_mask(raster, soft_mask),
            soft_mask=soft_mask,
        )

    def decode(self) -> ImageRaster | None:
        """Decode and return the canonical raster for extraction consumers."""
        prepared = self.prepare()
        return prepared.raster if prepared is not None else None

    def internal_decode_mask(self) -> DecodedRaster | None:
        width = parse_int(self.dictionary.get("Width"), 0)
        height = parse_int(self.dictionary.get("Height"), 0)
        if width <= 0 or height <= 0:
            return None
        try:
            decoded = decode_stream_data(self.raw, self.dictionary)
        except FilterError:
            # Falling back to self.raw here unpacked the still-encoded bytes as
            # a bitmap, painting compressed noise into the alpha plane. A mask
            # that cannot be decoded is dropped instead. An unfiltered stream
            # does not reach this path -- decode_stream_data returns it as-is.
            return None
        row_bytes = (width + 7) // 8
        if len(decoded) < row_bytes * height:
            return None
        packed = numpy.frombuffer(decoded, dtype=numpy.uint8)[: row_bytes * height]
        bits = numpy.unpackbits(packed).reshape(height, row_bytes * 8)[:, :width]
        # ISO 32000-1 8.9.6.2: "If the Decode array is [ 0 1 ] (the default for
        # an image mask), a sample value of 0 shall mark the page with the
        # current colour, and a 1 shall leave the previous contents unchanged.
        # If the Decode array is [ 1 0 ], these meanings shall be reversed."
        # Alpha is the marking mask here, so sample 0 is the opaque one.
        alpha = (1 - bits) * 255
        decode = self.dictionary.get("Decode")
        if isinstance(decode, (list, tuple)) and len(decode) >= 2:
            try:
                if float(decode[0]) > float(decode[1]):
                    alpha = 255 - alpha
            except (TypeError, ValueError):
                pass
        array = numpy.zeros((height, width, 2), dtype=numpy.uint8)
        array[:, :, 1] = alpha
        return DecodedRaster(array, width, height, 2)

    def internal_decode_soft_mask(self) -> ImageRaster | None:
        soft_mask = self.soft_mask
        if soft_mask is None:
            return None
        mask_dictionary = dict(soft_mask.dictionary)
        mask_dictionary.setdefault("ColorSpace", "DeviceGray")
        mask_dictionary.setdefault("BitsPerComponent", 8)
        prepared = ImageSource(soft_mask.raw, mask_dictionary).prepare()
        if prepared is None:
            return None
        mask = prepared.raster
        if mask.color_model == "gray" and not mask.has_alpha:
            return mask
        return ImageRaster(mask.array[:, :, :1], "gray")

    def internal_apply_soft_mask(self, raster: ImageRaster, mask: ImageRaster) -> ImageRaster:
        mask_array = mask.array[:, :, 0]
        y = numpy.minimum(
            mask.height - 1,
            (numpy.arange(raster.height) * mask.height) // raster.height,
        )
        x = numpy.minimum(
            mask.width - 1,
            (numpy.arange(raster.width) * mask.width) // raster.width,
        )
        alpha = mask_array[y[:, None], x[None, :]]
        array = numpy.empty((raster.height, raster.width, raster.channels + 1), dtype=numpy.uint8)
        array[:, :, : raster.channels] = raster.array
        array[:, :, raster.channels] = alpha
        return ImageRaster(array, raster.color_model, has_alpha=True)


def internal_canonical_image_array(
    samples: DecodedImage,
    dictionary: dict[Any, Any],
) -> tuple[numpy.ndarray[Any, Any], int] | None:
    """Normalize a decoded image to a contiguous grayscale/RGB sample array."""
    array = numpy.asarray(samples.array, dtype=numpy.uint8)
    if array.ndim == 2:
        channels = 1
    elif array.ndim == 3:
        channels = int(array.shape[-1])
    else:
        return None
    if channels == 4:
        converted = internal_convert_image_data(array.reshape(-1), dictionary)
        expected_rgb = int(array.shape[0]) * int(array.shape[1]) * 3
        if converted is not None and len(converted) == expected_rgb:
            array = numpy.asarray(converted, dtype=numpy.uint8).reshape(
                int(array.shape[0]), int(array.shape[1]), 3
            )
        else:
            array = array[..., :3]
        channels = 3
    if channels not in {1, 3}:
        return None
    array = numpy.ascontiguousarray(array)
    return array.reshape(-1), channels


def internal_decode_image_samples(
    raw: bytes | memoryview,
    dictionary: dict[Any, Any],
) -> bytes | memoryview | DecodedImage | None:
    width = parse_int(dictionary.get("Width"), 0)
    height = parse_int(dictionary.get("Height"), 0)
    if width <= 0 or height <= 0:
        return None
    native = decode_stream_image_data(raw, dictionary)
    if native is not None and native.width == width and native.height == height:
        return native
    bits_per_component = parse_int(dictionary.get("BitsPerComponent"), 8)
    expected_gray = width * height
    expected_rgb = expected_gray * 3
    if len(raw) in {expected_gray, expected_rgb}:
        return raw
    try:
        decoded = decode_stream_data(raw, dictionary)
    except Exception:
        return None
    if len(decoded) in {expected_gray, expected_rgb}:
        return decoded
    if bits_per_component in {1, 2, 4}:
        row_bytes = (width * bits_per_component + 7) // 8
        if len(decoded) >= row_bytes * height:
            return decoded
    return None


def decode_pdf_image(raw: bytes | memoryview, dictionary: dict[Any, Any]) -> DecodedRaster | None:
    width = parse_int(dictionary.get("Width"), 0)
    height = parse_int(dictionary.get("Height"), 0)
    if width <= 0 or height <= 0:
        return None
    samples = internal_decode_image_samples(raw, dictionary)
    if samples is None:
        return None
    if isinstance(samples, DecodedImage):
        canonical = internal_canonical_image_array(samples, dictionary)
        if canonical is None:
            return None
        array, channels = canonical
        return DecodedRaster(array, width, height, channels)
    try:
        converted = internal_convert_image_data(samples, dictionary)
    except ValueError:
        # Broken PDFs sometimes retain an unresolved or malformed ICCBased
        # reference even though the decoded stream contains ordinary device
        # samples. The exact sample width is unambiguous here, so preserve the
        # image instead of failing the whole page.
        pixels = width * height
        if len(samples) in {pixels, pixels * 3}:
            converted = samples
        elif len(samples) == pixels * 4:
            converted = internal_convert_cmyk(samples)
        else:
            return None
    if converted is None:
        return None
    if isinstance(converted, bytearray):
        converted = bytes(converted)
    pixels = width * height
    channels = len(converted) // pixels
    if channels not in {1, 3} or len(converted) != pixels * channels:
        return None
    return DecodedRaster(converted, width, height, channels)


__all__ = (
    "DecodedRaster",
    "ImageRaster",
    "ImageSource",
    "PreparedImage",
    "SoftMask",
    "decode_pdf_image",
)
