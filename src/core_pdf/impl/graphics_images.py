# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from contextlib import suppress
from itertools import batched
from typing import Any

import numpy

from core_pdf.impl.array_views import readonly
from core_pdf.impl.graphics_color import (
    convert_cmyk,
    convert_image_data,
)
from core_pdf.impl.graphics_color_spec import ColorSpace, parse_color_space, raw_color_space_paints
from core_pdf.impl.graphics_decode_compat import (
    normalize_stream_decode_spec,
)
from core_pdf.impl.graphics_filter_registry import FilterDecoder, NativeImageCodec
from core_pdf.impl.graphics_image_header import ImageHeader
from core_pdf.impl.graphics_image_samples import (
    convert_integer_image,
    convert_integer_samples,
)
from core_pdf.impl.graphics_soft_masks import image_has_color_key_mask
from core_pdf.impl.graphics_stream_decoding import (
    TOLERANT_FILTER_BY_NAME,
    decode_one_filter,
    decode_stream_data,
)
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import GeneratedRecord
from core_pdf_cythonized import interleave_soft_mask
from core_pdf_spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf_spec.s_07_filters.errors import FilterError
from core_pdf_spec.s_08_graphics.color_kernels import (
    decode_sample_values,
    unpack_image_samples,
    unpack_subbyte_image_samples,
)
from core_pdf_spec.s_08_graphics.color_rendering import DEFAULT_COLOR_RENDERING, ColorRendering
from core_pdf_spec.s_08_graphics.image_spec import (
    ImageSource,
    SoftMask,
    image_color_rendering,
    image_decode_array_applies,
    image_smask_in_data,
)
from core_pdf_spec.standards import SemanticContext


def decode_array_applies(dictionary: dict[Any, Any], context: SemanticContext | None) -> bool:
    known = context if context and context.version and context.version.recognized else None
    return image_decode_array_applies(dictionary, context=known)


def image_color_space_paints(dictionary: dict[Any, Any]) -> bool:
    if dictionary.get("ImageMask") is True:
        return True
    return raw_color_space_paints(dictionary.get("ColorSpace"))


def samples_equal(left: object, right: object) -> bool:
    if isinstance(left, numpy.ndarray) or isinstance(right, numpy.ndarray):
        return (
            isinstance(left, numpy.ndarray)
            and isinstance(right, numpy.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(numpy.array_equal(left, right))
        )
    return bool(left == right)


class DecodedRaster(GeneratedRecord):
    data: bytes | memoryview | numpy.ndarray[Any, Any]
    width: int
    height: int
    channels: int

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.channels == other.channels
            and samples_equal(self.data, other.data)
        )

    def __hash__(self) -> int:
        return hash((self.width, self.height, self.channels))


class ImageRaster(GeneratedRecord):
    array: numpy.ndarray[Any, Any]
    color_model: str

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.color_model == other.color_model and samples_equal(self.array, other.array)

    def __hash__(self) -> int:
        return hash((self.array.shape, self.array.dtype.str, self.color_model))

    def __post_init__(self) -> None:
        array = numpy.asarray(self.array, dtype=numpy.uint8)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3 or array.shape[2] not in {1, 2, 3, 4}:
            raise ValueError("image raster must have one, two, three, or four channels")
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
    def has_alpha(self) -> bool:
        return self.channels in {2, 4}

    @property
    def stride(self) -> int:
        return int(self.array.strides[0])


class PreparedImage(GeneratedRecord):
    raster: ImageRaster
    soft_mask: ImageRaster | None = None
    is_stencil: bool = False

    def __post_init__(self) -> None:
        soft_mask = self.soft_mask
        if soft_mask is not None and (soft_mask.color_model != "gray" or soft_mask.has_alpha):
            raise ValueError("prepared image soft mask must be grayscale without alpha")


def decode_mask(source: ImageSource) -> DecodedRaster | None:
    header = ImageHeader(source.dictionary)
    width = header.width
    height = header.height
    if width <= 0 or height <= 0:
        return None
    try:
        decoded = decode_stream_data(source.raw, source.dictionary)
    except FilterError:
        return None
    row_bytes = (width + 7) // 8
    if len(decoded) < row_bytes * height:
        return None
    bits = unpack_subbyte_image_samples(decoded, 1, width, height, 1).reshape(height, width)
    alpha = (1 - bits) * 255
    decode = header.decode
    if isinstance(decode, (list, tuple)) and len(decode) >= 2:
        try:
            if float(decode[0]) > float(decode[1]):
                alpha = 255 - alpha
        except TypeError, ValueError:
            pass
    array = numpy.zeros((height, width, 2), dtype=numpy.uint8)
    array[:, :, 1] = alpha
    return DecodedRaster(array, width, height, 2)


def decode_soft_mask(source: ImageSource, soft_mask: SoftMask) -> ImageRaster | None:
    mask_dictionary = dict(soft_mask.dictionary)
    mask_dictionary.setdefault("ColorSpace", "DeviceGray")
    mask_dictionary.setdefault("BitsPerComponent", 8)
    prepared = prepare_image(
        ImageSource(soft_mask.raw, mask_dictionary, semantic_context=source.semantic_context)
    )
    if prepared is None:
        return None
    mask = prepared.raster
    if mask.color_model == "gray" and not mask.has_alpha:
        return mask
    return ImageRaster(mask.array[:, :, :1], "gray")


def decode_matte(
    source: ImageSource, soft_mask: SoftMask, source_header: ImageHeader
) -> tuple[tuple[float, ...], numpy.ndarray[Any, Any]]:
    dictionary = soft_mask.dictionary
    header = ImageHeader(dictionary)
    width = header.width
    height = header.height
    if (width, height) != (source_header.width, source_header.height):
        raise ValueError("image matte requires matching soft mask dimensions")
    decoded = decode_image_samples(soft_mask.raw, dictionary, size=(width, height), header=header)
    decode = dictionary.get("Decode", (0, 1))
    if isinstance(decoded, DecodedImage):
        if decoded.channels != 1:
            raise ValueError("invalid image soft mask channels")
        integers = decoded.array.reshape(-1)
        maximum = 65535 if integers.dtype == numpy.uint16 else 255
        if decoded.source == "jpx" and not decode_array_applies(
            dictionary, source.semantic_context
        ):
            decode = (0, 1)
    elif decoded is not None:
        bits = header.bits
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


def apply_soft_mask(raster: ImageRaster, mask: ImageRaster) -> ImageRaster:
    mask_array = mask.array[:, :, 0]
    y = numpy.minimum(
        mask.height - 1,
        (numpy.arange(raster.height) * mask.height) // raster.height,
    )
    x = numpy.minimum(
        mask.width - 1,
        (numpy.arange(raster.width) * mask.width) // raster.width,
    )
    channels = raster.channels - int(raster.has_alpha)
    array = interleave_soft_mask(
        raster.array,
        channels,
        mask_array,
        numpy.ascontiguousarray(y, dtype=numpy.intp),
        numpy.ascontiguousarray(x, dtype=numpy.intp),
    )
    return ImageRaster(array, raster.color_model)


def canonical_image_array(
    samples: DecodedImage,
    dictionary: dict[Any, Any],
    *,
    matte: tuple[float, ...] | None = None,
    alpha: numpy.ndarray[Any, Any] | None = None,
    semantic_context: SemanticContext | None = None,
    rendering: ColorRendering = DEFAULT_COLOR_RENDERING,
    header: ImageHeader | None = None,
) -> tuple[numpy.ndarray[Any, Any], int] | None:
    explicit_jpx_decode = (
        samples.source == "jpx"
        and dictionary.get("Decode") is not None
        and decode_array_applies(dictionary, semantic_context)
    )
    encoded_alpha = None
    sample_array = samples.array
    high_depth_dictionary = dict(dictionary)
    space: ColorSpace | None = None
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
                space=space,
            )
        except TypeError, ValueError:
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
        converted = convert_image_data(array.reshape(-1), dictionary, header=header)
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


class FilterChainOutput:
    __slots__ = ("raw", "spec", "output")

    def __init__(self, raw: bytes | memoryview, spec: StreamDecodeSpec) -> None:
        self.raw = raw
        self.spec = spec
        self.output: bytes | None = None

    def decoded(self) -> bytes:
        output = self.output
        if output is None:
            output = self.output = decode_stream_data(self.raw, self.spec)
        return output


def decode_image_samples(
    raw: bytes | memoryview,
    dictionary: dict[Any, Any],
    *,
    size: tuple[int, int] | None = None,
    header: ImageHeader | None = None,
) -> bytes | memoryview | DecodedImage | None:
    if header is None:
        header = ImageHeader(dictionary)
    if size is None:
        width = header.width
        height = header.height
    else:
        width, height = size
    if width <= 0 or height <= 0:
        return None
    chain = FilterChainOutput(raw, normalize_stream_decode_spec(dictionary))
    native = decode_stream_image_data(raw, dictionary, chain)
    if native is not None and native.width == width and native.height == height:
        return native
    bits_per_component = header.bits
    if bits_per_component == 16:
        try:
            return chain.decoded()
        except Exception:
            return None
    expected_gray = width * height
    expected_rgb = expected_gray * 3
    expected_source = 0
    with suppress(ValueError):
        expected_source = expected_gray * len(header.space().component_ranges)
    if header.filter is None and (
        len(raw) in {expected_gray, expected_rgb}
        or (bits_per_component == 8 and len(raw) == expected_source)
    ):
        return raw
    try:
        decoded = chain.decoded()
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
    header: ImageHeader | None = None,
) -> DecodedRaster | None:
    if not image_color_space_paints(dictionary):
        return None
    with suppress(ValueError):
        rendering = image_color_rendering(dictionary, rendering)
    if header is None:
        header = ImageHeader(dictionary)
    width = header.width
    height = header.height
    if width <= 0 or height <= 0:
        return None
    samples = decode_image_samples(raw, dictionary, size=(width, height), header=header)
    if samples is None:
        return None
    if isinstance(samples, DecodedImage):
        canonical = canonical_image_array(
            samples,
            dictionary,
            matte=matte,
            alpha=alpha,
            semantic_context=semantic_context,
            rendering=rendering,
            header=header,
        )
        if canonical is None:
            return None
        array, channels = canonical
        return DecodedRaster(array, width, height, channels)
    bits_per_component = header.bits
    if bits_per_component == 16 or header.has_color_key_mask:
        try:
            converted_words = convert_integer_image(
                samples,
                dictionary,
                bits_per_component=bits_per_component,
                matte=matte,
                alpha=alpha,
                rendering=rendering,
                space=header.space(),
            )
        except TypeError, ValueError:
            return None
        return DecodedRaster(converted_words.reshape(-1), width, height, converted_words.shape[1])
    try:
        converted = convert_image_data(samples, dictionary, rendering=rendering, header=header)
    except ValueError:
        pixels = width * height
        if len(samples) in {pixels, pixels * 3}:
            converted = samples
        elif len(samples) == pixels * 4:
            converted = convert_cmyk(samples)
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
    if not image_color_space_paints(source.dictionary):
        return None
    is_stencil = source.dictionary.get("ImageMask") is True
    dictionary = source.dictionary
    try:
        rendering = image_color_rendering(dictionary, source.color_rendering)
    except ValueError:
        rendering = source.color_rendering
    matte = None
    alpha = None
    soft_mask_source = source.soft_mask
    if soft_mask_source is not None:
        dictionary = dict(dictionary)
        dictionary.pop("Mask", None)
    header = ImageHeader(dictionary)
    if soft_mask_source is not None:
        filters = header.filter if isinstance(header.filter, (list, tuple)) else (header.filter,)
        if soft_mask_source.dictionary.get("Matte") is not None and (
            header.bits == 16 or "JPXDecode" in filters
        ):
            try:
                matte, alpha = decode_matte(source, soft_mask_source, header)
            except TypeError, ValueError:
                return None
    decoded = (
        decode_mask(source)
        if is_stencil
        else decode_pdf_image(
            source.raw,
            dictionary,
            matte=matte,
            alpha=alpha,
            semantic_context=source.semantic_context,
            rendering=rendering,
            header=header,
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
    raster = ImageRaster(array, color_model)
    if is_stencil:
        return PreparedImage(raster, is_stencil=True)
    if soft_mask_source is None:
        return PreparedImage(raster)
    soft_mask = decode_soft_mask(source, soft_mask_source)
    if soft_mask is None:
        return PreparedImage(raster)
    return PreparedImage(
        apply_soft_mask(raster, soft_mask),
        soft_mask=soft_mask,
    )


def decode_image(source: ImageSource) -> ImageRaster | None:
    prepared = prepare_image(source)
    return prepared.raster if prepared is not None else None


class DecodedImage(GeneratedRecord):
    array: numpy.ndarray[Any, Any]
    source: str

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.source == other.source and samples_equal(self.array, other.array)

    def __hash__(self) -> int:
        return hash((self.array.shape, self.array.dtype.str, self.source))

    def __post_init__(self) -> None:
        if self.array.ndim not in {2, 3}:
            raise ValueError("decoded image must have two or three dimensions")
        if self.array.dtype not in (numpy.uint8, numpy.uint16):
            raise ValueError("decoded image samples must be uint8 or uint16")
        if not self.array.flags.c_contiguous:
            raise ValueError("decoded image must be C-contiguous")

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.array.ndim == 2 else int(self.array.shape[2])


class NativeImagePlan(GeneratedRecord):
    decoder: FilterDecoder
    native: NativeImageCodec
    params: object
    output_shape: tuple[int, ...] | None


def prepare_native_image(
    dictionary: object, stream_spec: StreamDecodeSpec | None = None
) -> NativeImagePlan | None:
    if stream_spec is None:
        stream_spec = normalize_stream_decode_spec(dictionary)
    if len(stream_spec.steps) != 1:
        return None
    step = stream_spec.steps[0]
    tolerant = TOLERANT_FILTER_BY_NAME.get(step.name)
    native = tolerant.native if tolerant is not None else None
    if tolerant is None or native is None:
        return None
    if native.requires_identity_decode and not image_decode_is_identity(dictionary):
        return None
    spec = native.spec
    image_dictionary = dictionary if isinstance(dictionary, dict) else {}
    color_space = image_dictionary.get("ColorSpace")
    if isinstance(color_space, (list, tuple, dict)):
        return None
    color_name = recover_pdf_name(color_space) if color_space is not None else None
    bits = image_dictionary.get("BitsPerComponent")
    if color_name not in spec.color_names or (spec.bits is not None and bits not in spec.bits):
        return None
    width = image_dictionary.get("Width")
    height = image_dictionary.get("Height")
    components = spec.channels.get(color_name)
    shape = None
    if (
        type(width) is int
        and type(height) is int
        and width > 0
        and height > 0
        and components is not None
    ):
        shape = (height, width) if components == 1 else (height, width, components)
    return NativeImagePlan(tolerant.decoder, native, step.params, shape)


def decode_stream_image_data(
    data: bytes | memoryview,
    dictionary: object,
    chain: FilterChainOutput,
) -> DecodedImage | None:
    stream_spec = chain.spec
    steps = stream_spec.steps
    last = TOLERANT_FILTER_BY_NAME.get(steps[-1].name) if steps else None
    last_native = last.native if last is not None else None
    if last is not None and last_native is not None and last_native.after_filters:
        try:
            compressed = bytes(data)
            for step in steps[:-1]:
                compressed = decode_one_filter(
                    compressed,
                    step.name,
                    step.params,
                    dictionary=dictionary,
                    parent_dictionary=None,
                )
            array = last_native.decode(compressed, steps[-1].params, None, chain.decoded)
            if array is None:
                return None
            color_space = dictionary.get("ColorSpace") if isinstance(dictionary, dict) else None
            if array.dtype == numpy.uint16 or not isinstance(color_space, (list, tuple, dict)):
                return DecodedImage(array, last.decoder)
            chain.output = array.tobytes()
        except Exception:
            return None
    plan = prepare_native_image(dictionary, stream_spec)
    if plan is None:
        return None
    try:
        array = plan.native.decode(data, plan.params, plan.output_shape, chain.decoded)
        return DecodedImage(array, plan.decoder) if array is not None else None
    except Exception:
        return None


def image_decode_is_identity(dictionary: object) -> bool:

    decode = dictionary.get("Decode") if isinstance(dictionary, dict) else None
    if decode is None:
        return True
    if not isinstance(decode, (list, tuple)) or len(decode) == 0 or len(decode) % 2:
        return False
    for lower, upper in batched(decode, 2, strict=True):
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            return False
        if float(lower) != 0.0 or float(upper) != 1.0:
            return False
    return True
