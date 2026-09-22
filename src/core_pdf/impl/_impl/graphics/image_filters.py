# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from dataclasses import dataclass
from itertools import batched

import numpy

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    FilterFn = Callable[[bytes, object], bytes]

from core_pdf.impl._impl.graphics.codec_dispatch import (
    decode_ccitt_fax_image,
    decode_jpeg_image,
    decode_jpx_image,
)
from core_pdf.impl._impl.graphics.decode_compat import (
    FilterParams,
    normalize_stream_decode_spec,
)
from core_pdf.impl._impl.graphics.filter_registry import (
    FILTER_DESCRIPTOR_BY_NAME,
    NATIVE_IMAGE_SPECS,
    FilterDecoder,
)
from core_pdf.impl._impl.graphics.image_models import DecodedImage
from core_pdf.impl._impl.graphics.stream_decoding import decode_one_filter, decode_stream_data
from core_pdf.impl._impl.pdf_names import recover_pdf_name

internal_NATIVE_ARRAY_DECODERS = {
    "jpeg": decode_jpeg_image,
    "jpx": decode_jpx_image,
}


@dataclass(frozen=True, slots=True)
class internal_NativeImagePlan:
    decoder: FilterDecoder
    params: object
    output_shape: tuple[int, ...] | None


def internal_prepare_native_image(dictionary: object) -> internal_NativeImagePlan | None:
    stream_spec = normalize_stream_decode_spec(dictionary)
    if len(stream_spec.steps) != 1:
        return None
    step = stream_spec.steps[0]
    descriptor = FILTER_DESCRIPTOR_BY_NAME.get(step.name)
    decoder = descriptor.decoder if descriptor is not None else None
    if decoder is None or (decoder != "jpx" and not image_decode_is_identity(dictionary)):
        return None
    spec = NATIVE_IMAGE_SPECS.get(decoder)
    if spec is None:
        return None
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
    return internal_NativeImagePlan(decoder, step.params, shape)


def decode_stream_image_data(
    data: bytes | memoryview,
    dictionary: object,
) -> DecodedImage | None:

    stream_spec = normalize_stream_decode_spec(dictionary)
    if stream_spec.steps and stream_spec.steps[-1].name == "JPXDecode":
        try:
            compressed = bytes(data)
            for step in stream_spec.steps[:-1]:
                compressed = decode_one_filter(
                    compressed,
                    step.name,
                    step.params,
                    dictionary=dictionary,
                    parent_dictionary=None,
                )
            array = decode_jpx_image(compressed, preserve_precision=True)
            color_space = dictionary.get("ColorSpace") if isinstance(dictionary, dict) else None
            if array.dtype == numpy.uint16 or not isinstance(color_space, (list, tuple, dict)):
                return DecodedImage(array, "jpx")
        except Exception:
            return None
    plan = internal_prepare_native_image(dictionary)
    if plan is None:
        return None
    decoder = plan.decoder
    params = plan.params
    output_shape = plan.output_shape
    source = data
    try:
        array_decoder = internal_NATIVE_ARRAY_DECODERS.get(decoder)
        if decoder == "jpx":
            return DecodedImage(decode_jpx_image(source, preserve_precision=True), decoder)
        if array_decoder is not None:
            output = numpy.empty(output_shape, dtype=numpy.uint8) if output_shape else None
            return DecodedImage(array_decoder(source, out=output), decoder)
        if decoder == "ccitt":
            filter_params = (
                params if type(params) is FilterParams else FilterParams.from_parms(params)
            )
            output = (
                numpy.empty((filter_params.rows, filter_params.columns), dtype=numpy.uint8)
                if filter_params.rows > 0 and filter_params.columns > 0
                else None
            )
            return DecodedImage(
                decode_ccitt_fax_image(source, filter_params, out=output),
                "ccitt",
            )
        if decoder in {"flate", "lzw"}:
            if output_shape is None:
                return None
            decoded = decode_stream_data(data, dictionary)
            expected_size = int(numpy.prod(output_shape, dtype=numpy.int64))
            if len(decoded) != expected_size:
                return None
            array = numpy.frombuffer(decoded, dtype=numpy.uint8).reshape(output_shape)
            return DecodedImage(array, decoder)
    except Exception:
        return None
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
