# SPDX-License-Identifier: AGPL-3.0-only
"""Decode PDF image samples independently of downstream consumers."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy

from core_pdf.impl._impl.graphics.color import (
    internal_convert_cmyk,
    internal_convert_image_data,
)
from core_pdf.impl._impl.graphics.color_spec import internal_color_space_paints, parse_color_space
from core_pdf.impl._impl.graphics.image_filters import decode_stream_image_data
from core_pdf.impl._impl.graphics.image_models import DecodedImage
from core_pdf.impl._impl.graphics.image_samples import (
    convert_integer_image,
    convert_integer_samples,
)
from core_pdf.impl._impl.graphics.soft_masks import image_has_color_key_mask
from core_pdf.impl._impl.graphics.stream_decoding import (
    decode_stream_data,
)
from core_pdf.impl._impl.runtime.array_views import readonly
from core_pdf.impl._impl.runtime.scalars import parse_int
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_08_graphics.color_kernels import decode_sample_values, unpack_image_samples
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import (
    ImageSource,
    SoftMask,
    image_color_rendering,
    image_decode_array_applies,
    image_smask_in_data,
)
from core_pdf_spec.standards import SemanticContext


def internal_decode_array_applies(
    dictionary: dict[Any, Any], context: SemanticContext | None
) -> bool:
    # Reader recovery retains the historical interpretation for unknown versions.
    known = context if context and context.version and context.version.recognized else None
    return image_decode_array_applies(dictionary, context=known)


def internal_image_color_space_paints(dictionary: dict[Any, Any]) -> bool:
    """Recognize discarded colourants before decoding samples or associated masks."""
    if dictionary.get("ImageMask") is True:
        return True
    return internal_color_space_paints(dictionary.get("ColorSpace"))


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


class internal_ImagePreparation(ImageSource):
    def prepare(self) -> PreparedImage | None:
        """Decode and return an immutable prepared image."""
        if not internal_image_color_space_paints(self.dictionary):
            return None
        is_stencil = self.dictionary.get("ImageMask") is True
        dictionary = self.dictionary
        try:
            rendering = image_color_rendering(dictionary, self.color_rendering)
        except ValueError:
            rendering = self.color_rendering
        matte = None
        alpha = None
        if self.soft_mask is not None:
            dictionary = dict(dictionary)
            dictionary.pop("Mask", None)
            filters = dictionary.get("Filter", ())
            filters = filters if isinstance(filters, (list, tuple)) else (filters,)
            if self.soft_mask.dictionary.get("Matte") is not None and (
                parse_int(dictionary.get("BitsPerComponent"), 8) == 16 or "JPXDecode" in filters
            ):
                try:
                    matte, alpha = self.internal_decode_matte()
                except (TypeError, ValueError):
                    return None
        decoded = (
            self.internal_decode_mask()
            if is_stencil
            else decode_pdf_image(
                self.raw,
                dictionary,
                matte=matte,
                alpha=alpha,
                semantic_context=self.semantic_context,
                rendering=rendering,
            )
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
        prepared = internal_ImagePreparation(
            soft_mask.raw, mask_dictionary, semantic_context=self.semantic_context
        ).prepare()
        if prepared is None:
            return None
        mask = prepared.raster
        if mask.color_model == "gray" and not mask.has_alpha:
            return mask
        return ImageRaster(mask.array[:, :, :1], "gray")

    def internal_decode_matte(self) -> tuple[tuple[float, ...], numpy.ndarray[Any, Any]]:
        soft_mask = self.soft_mask
        if soft_mask is None:
            raise ValueError("missing image soft mask")
        dictionary = soft_mask.dictionary
        width = parse_int(dictionary.get("Width"), 0)
        height = parse_int(dictionary.get("Height"), 0)
        if (width, height) != (
            parse_int(self.dictionary.get("Width"), 0),
            parse_int(self.dictionary.get("Height"), 0),
        ):
            # ISO 32000-1/2 11.6.5.2 requires matching dimensions with Matte.
            raise ValueError("image matte requires matching soft mask dimensions")
        decoded = internal_decode_image_samples(soft_mask.raw, dictionary)
        decode = dictionary.get("Decode", (0, 1))
        if isinstance(decoded, DecodedImage):
            if decoded.channels != 1:
                raise ValueError("invalid image soft mask channels")
            integers = decoded.array.reshape(-1)
            maximum = 65535 if integers.dtype == numpy.uint16 else 255
            if decoded.source == "jpx" and not internal_decode_array_applies(
                dictionary, self.semantic_context
            ):
                decode = (0, 1)
        elif decoded is not None:
            bits = parse_int(dictionary.get("BitsPerComponent"), 8)
            integers = unpack_image_samples(decoded, bits, width, height, 1)
            maximum = (1 << bits) - 1
        else:
            raise ValueError("invalid image soft mask samples")
        if len(decode) != 2:
            raise ValueError("invalid image soft mask Decode")
        pairs = ((float(decode[0]), float(decode[1])),)
        alpha = numpy.clip(decode_sample_values(integers, pairs, maximum).reshape(-1), 0, 1)
        matte = tuple(float(value) for value in dictionary["Matte"])
        return matte, alpha

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
        channels = raster.channels - int(raster.has_alpha)
        array = numpy.empty((raster.height, raster.width, channels + 1), dtype=numpy.uint8)
        array[:, :, :channels] = raster.array[:, :, :channels]
        array[:, :, channels] = alpha
        return ImageRaster(array, raster.color_model, has_alpha=True)


def internal_canonical_image_array(
    samples: DecodedImage,
    dictionary: dict[Any, Any],
    *,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    semantic_context: SemanticContext | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> tuple[numpy.ndarray[Any, Any], int] | None:
    """Normalize a decoded image to a contiguous grayscale/RGB sample array."""
    explicit_jpx_decode = (
        samples.source == "jpx"
        and dictionary.get("Decode") is not None
        and internal_decode_array_applies(dictionary, semantic_context)
    )
    encoded_alpha = None
    sample_array = samples.array
    high_depth_dictionary = dict(dictionary)
    if samples.source == "jpx":
        try:
            selector = image_smask_in_data(dictionary)
        except ValueError:
            selector = 0
        active_alpha = selector in {1, 2}
        ordinary_channels = samples.channels - int(active_alpha)
        inferred_space = {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(ordinary_channels)
        high_depth_dictionary.setdefault("ColorSpace", inferred_space)
        try:
            space = parse_color_space(high_depth_dictionary.get("ColorSpace"))
        except ValueError:
            return None
        count = len(space.component_ranges)
        if samples.channels == count + 1:
            # ISO 32000-1/2 Table 89/87: selector 0 ignores an opacity channel;
            # 1 uses straight colours; 2 first undoes opacity premultiplication.
            words = samples.array.reshape(-1, samples.channels)
            sample_array = words[:, :count]
            if active_alpha:
                maximum = 65535 if samples.array.dtype == numpy.uint16 else 255
                encoded_alpha = words[:, count].astype(numpy.float64) / maximum
                high_depth_dictionary.pop("Mask", None)
                if selector == 2:
                    matte_space = space.base if space.kind == "Indexed" else space
                    if matte_space is None:
                        return None
                    matte = (0.0,) * len(matte_space.component_ranges)
                    alpha = encoded_alpha
        elif samples.channels != count or active_alpha:
            return None
    if (
        samples.array.dtype == numpy.uint16
        or explicit_jpx_decode
        or sample_array is not samples.array
        or rendering != DEFAULT_COLOR_RENDERING
        or image_has_color_key_mask(high_depth_dictionary)
    ):
        if samples.source == "jpx" and not explicit_jpx_decode:
            high_depth_dictionary.pop("Decode", None)
        high_depth_dictionary.setdefault(
            "ColorSpace", "DeviceGray" if samples.channels == 1 else "DeviceRGB"
        )
        try:
            converted_words = convert_integer_samples(
                sample_array,
                high_depth_dictionary,
                bits_per_component=16 if samples.array.dtype == numpy.uint16 else 8,
                matte=matte,
                alpha=alpha,
                rendering=rendering,
            )
        except (TypeError, ValueError):
            return None
        if converted_words.shape[0] != samples.width * samples.height:
            return None
        if encoded_alpha is not None:
            converted_words = numpy.column_stack(
                (converted_words, numpy.rint(encoded_alpha * 255).astype(numpy.uint8))
            )
        return converted_words.reshape(-1), int(converted_words.shape[1])
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
    if bits_per_component == 16:
        # Validate the complete word layout after filters/predictors. Byte-length
        # heuristics for 8-bit recovery must not reinterpret compressed words.
        try:
            return decode_stream_data(raw, dictionary)
        except Exception:
            return None
    expected_gray = width * height
    expected_rgb = expected_gray * 3
    expected_source = 0
    with suppress(ValueError):
        expected_source = expected_gray * len(
            parse_color_space(dictionary.get("ColorSpace")).component_ranges
        )
    if bits_per_component == 8 and dictionary.get("Filter") is None and len(raw) == expected_source:
        return raw
    if len(raw) in {expected_gray, expected_rgb}:
        return raw
    try:
        decoded = decode_stream_data(raw, dictionary)
    except Exception:
        return None
    if len(decoded) in {expected_gray, expected_rgb, expected_source}:
        return decoded
    if bits_per_component in {1, 2, 4}:
        row_bytes = (width * bits_per_component + 7) // 8
        if len(decoded) >= row_bytes * height:
            return decoded
    return None


def decode_pdf_image(
    raw: bytes | memoryview,
    dictionary: dict[Any, Any],
    *,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    semantic_context: SemanticContext | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
) -> DecodedRaster | None:
    if not internal_image_color_space_paints(dictionary):
        return None
    with suppress(ValueError):
        rendering = image_color_rendering(dictionary, rendering)
    width = parse_int(dictionary.get("Width"), 0)
    height = parse_int(dictionary.get("Height"), 0)
    if width <= 0 or height <= 0:
        return None
    samples = internal_decode_image_samples(raw, dictionary)
    if samples is None:
        return None
    if isinstance(samples, DecodedImage):
        canonical = internal_canonical_image_array(
            samples,
            dictionary,
            matte=matte,
            alpha=alpha,
            semantic_context=semantic_context,
            rendering=rendering,
        )
        if canonical is None:
            return None
        array, channels = canonical
        return DecodedRaster(array, width, height, channels)
    bits_per_component = parse_int(dictionary.get("BitsPerComponent"), 8)
    if bits_per_component == 16 or image_has_color_key_mask(dictionary):
        # Color-key masks compare original integers before Decode or
        # conversion (8.9.6.4), and their holes are intrinsic shape.
        try:
            converted_words = convert_integer_image(
                samples,
                dictionary,
                bits_per_component=bits_per_component,
                matte=matte,
                alpha=alpha,
                rendering=rendering,
            )
        except (TypeError, ValueError):
            return None
        return DecodedRaster(converted_words.reshape(-1), width, height, converted_words.shape[1])
    try:
        converted = internal_convert_image_data(samples, dictionary, rendering=rendering)
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
    if channels not in {1, 2, 3, 4} or len(converted) != pixels * channels:
        return None
    return DecodedRaster(converted, width, height, channels)


__all__ = (
    "DecodedRaster",
    "ImageRaster",
    "ImageSource",
    "PreparedImage",
    "SoftMask",
    "decode_image",
    "decode_pdf_image",
    "prepare_image",
)


def prepare_image(source: ImageSource) -> PreparedImage | None:
    """Prepare a PDF image using the configured device and recovery behavior."""
    return internal_ImagePreparation(
        source.raw,
        source.dictionary,
        soft_mask=source.soft_mask,
        semantic_context=source.semantic_context,
        color_rendering=source.color_rendering,
    ).prepare()


def decode_image(source: ImageSource) -> ImageRaster | None:
    prepared = prepare_image(source)
    return prepared.raster if prepared is not None else None
