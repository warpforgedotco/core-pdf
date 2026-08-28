# SPDX-License-Identifier: AGPL-3.0-only
"""Decode PDF image samples independently of downstream consumers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, cast

import numpy

from core_pdf.impl.runtime.image_cache import ImageCache, ImageCacheKey
from core_pdf.impl.spec.s_07_filters.models import DecodedImage
from core_pdf.impl.spec.s_07_filters.pipeline import (
    decode_stream_data,
    decode_stream_image_data,
)
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.spec.s_08_graphics.color import ImageColorManager
from core_pdf.impl.spec.s_08_graphics.image_metadata import pdf_int


@dataclass(frozen=True, slots=True)
class DecodedRaster:
    data: bytes | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int


@dataclass(frozen=True, slots=True)
class ImageRaster:
    """Immutable canonical samples shared by all consumers of one image source."""

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
        array = numpy.ascontiguousarray(array)
        array.flags.writeable = False
        object.__setattr__(self, "array", array)

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

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Consumer-ready image data owned and cached by :class:`ImageSource`.

    ``raster`` preserves the existing shared-image contract: when a colour image
    carries a soft mask it includes a same-resolution alpha channel.  Renderers
    additionally need the mask at its native resolution so a high-resolution
    scan mask is not reduced to the colour image's dimensions.  Keeping that
    immutable plane here gives every consumer one preparation and cache owner.
    """

    raster: ImageRaster
    soft_mask: ImageRaster | None = None
    is_stencil: bool = False

    def __post_init__(self) -> None:
        soft_mask = self.soft_mask
        if soft_mask is not None and (soft_mask.color_model != "gray" or soft_mask.has_alpha):
            raise ValueError("prepared image soft mask must be grayscale without alpha")

    @property
    def nbytes(self) -> int:
        soft_mask = self.soft_mask
        return self.raster.nbytes + (soft_mask.nbytes if soft_mask is not None else 0)


class ImageSource:
    """Thread-safe lazy preparation owner for one embedded image source."""

    __slots__ = (
        "raw",
        "dictionary",
        "internal_lock",
        "internal_prepared",
        "internal_prepared_once",
        "cache",
        "cache_key",
    )

    def __init__(
        self,
        raw: bytes | memoryview,
        dictionary: dict[Any, Any],
        *,
        cache: ImageCache | None = None,
        cache_key: tuple[object, ...] = (),
    ) -> None:
        self.raw = raw
        self.dictionary = dictionary
        self.internal_lock = threading.Lock()
        self.internal_prepared: PreparedImage | None = None
        self.internal_prepared_once = False
        self.cache = cache
        self.cache_key = cache_key

    def prepare(self) -> PreparedImage | None:
        """Return the immutable prepared image, decoding at most once per cache key."""
        if self.cache is not None:
            key = ImageCacheKey("prepared-image", self.cache_key or (id(self),))
            value = self.cache.get_or_create(key, self.internal_prepare)
            return value if isinstance(value, PreparedImage) else None
        if self.internal_prepared_once:
            return self.internal_prepared
        with self.internal_lock:
            if self.internal_prepared_once:
                return self.internal_prepared
            self.internal_prepared = self.internal_prepare()
            self.internal_prepared_once = True
            return self.internal_prepared

    def decode(self) -> ImageRaster | None:
        """Return the canonical raster retained for extraction API compatibility."""
        prepared = self.prepare()
        return prepared.raster if prepared is not None else None

    def internal_prepare(self) -> PreparedImage | None:
        is_stencil = lookup_dict_key(self.dictionary, "ImageMask") is True
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

    def internal_decode_mask(self) -> DecodedRaster | None:
        width = pdf_int(lookup_dict_key(self.dictionary, "Width"), 0)
        height = pdf_int(lookup_dict_key(self.dictionary, "Height"), 0)
        if width <= 0 or height <= 0:
            return None
        try:
            decoded = decode_stream_data(self.raw, self.dictionary)
        except Exception:
            decoded = self.raw
        row_bytes = (width + 7) // 8
        if len(decoded) < row_bytes * height:
            return None
        packed = numpy.frombuffer(decoded, dtype=numpy.uint8)[: row_bytes * height]
        bits = numpy.unpackbits(packed).reshape(height, row_bytes * 8)[:, :width]
        alpha = bits * 255
        decode = lookup_dict_key(self.dictionary, "Decode")
        if isinstance(decode, (list, tuple)) and len(decode) >= 2:
            try:
                if float(cast(Any, decode[0])) > float(cast(Any, decode[1])):
                    alpha = 255 - alpha
            except (TypeError, ValueError):
                pass
        array = numpy.zeros((height, width, 2), dtype=numpy.uint8)
        array[:, :, 1] = alpha
        return DecodedRaster(array, width, height, 2)

    def internal_decode_soft_mask(self) -> ImageRaster | None:
        raw = self.dictionary.get("__soft_mask_raw_data__")
        dictionary = self.dictionary.get("__soft_mask_dictionary__")
        if not isinstance(raw, (bytes, memoryview)) or not isinstance(dictionary, dict):
            return None
        mask_dictionary = dict(dictionary)
        mask_dictionary.setdefault("ColorSpace", "DeviceGray")
        mask_dictionary.setdefault("BitsPerComponent", 8)
        # The parent PreparedImage owns this plane and reports its bytes to the
        # document cache. Caching a second nested PreparedImage would count the
        # same allocation twice and split preparation ownership again.
        prepared = ImageSource(raw, mask_dictionary).prepare()
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
        converted = ImageColorManager.convert_image_data(array.reshape(-1), dictionary)
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


def decode_pdf_image_samples(
    raw: bytes | memoryview,
    dictionary: dict[Any, Any],
) -> tuple[bytes | memoryview | DecodedImage, dict[Any, Any]] | None:
    width = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
    height = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
    if width <= 0 or height <= 0:
        return None
    native = decode_stream_image_data(raw, dictionary)
    if native is not None and native.width == width and native.height == height:
        return native, dictionary
    bits_per_component = pdf_int(lookup_dict_key(dictionary, "BitsPerComponent"), 8)
    expected_gray = width * height
    expected_rgb = expected_gray * 3
    if len(raw) in {expected_gray, expected_rgb}:
        return raw, dictionary
    try:
        decoded = decode_stream_data(raw, dictionary)
    except Exception:
        return None
    if len(decoded) in {expected_gray, expected_rgb}:
        return decoded, dictionary
    if bits_per_component in {1, 2, 4}:
        row_bytes = (width * bits_per_component + 7) // 8
        if len(decoded) >= row_bytes * height:
            return decoded, dictionary
    return None


def decode_pdf_image(raw: bytes | memoryview, dictionary: dict[Any, Any]) -> DecodedRaster | None:
    width = pdf_int(lookup_dict_key(dictionary, "Width"), 0)
    height = pdf_int(lookup_dict_key(dictionary, "Height"), 0)
    if width <= 0 or height <= 0:
        return None
    result = decode_pdf_image_samples(raw, dictionary)
    if result is None:
        return None
    samples, sample_dictionary = result
    if isinstance(samples, DecodedImage):
        canonical = internal_canonical_image_array(samples, sample_dictionary)
        if canonical is None:
            return None
        array, channels = canonical
        return DecodedRaster(array, width, height, channels)
    try:
        converted = ImageColorManager.convert_image_data(samples, sample_dictionary)
    except ValueError:
        # Broken PDFs sometimes retain an unresolved or malformed ICCBased
        # reference even though the decoded stream contains ordinary device
        # samples. The exact sample width is unambiguous here, so preserve the
        # image instead of failing the whole page.
        pixels = width * height
        if len(samples) in {pixels, pixels * 3}:
            converted = samples
        elif len(samples) == pixels * 4:
            converted = ImageColorManager.convert_cmyk(samples)
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
    "decode_pdf_image",
    "decode_pdf_image_samples",
)
