# SPDX-License-Identifier: AGPL-3.0-only
"""Filter dispatch from normalized stream filter names to decoders."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy

if typing.TYPE_CHECKING:
    from typing import Callable

    FilterFn = Callable[[bytes, object], bytes]

from core_pdf.impl.spec.s_07_filters.codecs import (
    apply_ascii85,
    apply_ascii_hex,
    apply_flate,
    apply_lzw,
    apply_run_length,
    looks_like_pdf_content_stream,
)
from core_pdf.impl.spec.s_07_filters.decode_spec import (
    FilterParams,
    StreamDecodeSpec,
    normalize_stream_decode_spec,
)
from core_pdf.impl.spec.s_07_filters.decoders import (
    decode_ccitt_fax,
    decode_ccitt_fax_image,
    decode_crypt,
    decode_jbig2,
    decode_jpeg,
    decode_jpeg_image,
    decode_jpx,
    decode_jpx_image,
)
from core_pdf.impl.spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf.impl.spec.s_07_filters.models import DecodedImage
from core_pdf.impl.spec.s_07_filters.predictors import apply_predictor
from core_pdf.impl.spec.s_07_filters.registry import (
    FILTER_DESCRIPTOR_BY_NAME,
    FILTER_DESCRIPTORS,
    NATIVE_IMAGE_SPECS,
    PREDICTOR_FILTERS,
    FilterDecoder,
)
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import full_source_bytes

internal_FILTER_DECODERS: dict[str, FilterFn] = {
    "flate": apply_flate,
    "ascii_hex": apply_ascii_hex,
    "ascii85": apply_ascii85,
    "run_length": apply_run_length,
    "lzw": apply_lzw,
    "jpeg": decode_jpeg,
    "ccitt": decode_ccitt_fax,
    "crypt": decode_crypt,
    "jpx": decode_jpx,
    "jbig2": decode_jbig2,
}
FILTER_MAP: dict[str, FilterFn] = {
    descriptor.name: internal_FILTER_DECODERS[descriptor.decoder]
    for descriptor in FILTER_DESCRIPTORS
    if descriptor.decoder is not None
}


def internal_coerce_decoder_bytes(result: object) -> bytes:
    if type(result) is bytearray:
        return bytes(result)
    if type(result) is not bytes:
        raise ValueError("invalid stream decoder result type")
    return result


def decode_one_filter(
    data: bytes,
    filter_name: str,
    parms: object,
    *,
    dictionary: object,
    parent_dictionary: object | None,
    allow_content_stream_passthrough: bool = False,
) -> bytes:
    if filter_name in {"None", "Identity"}:
        return data
    descriptor = FILTER_DESCRIPTOR_BY_NAME.get(filter_name)
    fn = FILTER_MAP.get(filter_name)
    if fn is None:
        raise FilterUnsupportedError(f"stream filter {filter_name} is not implemented yet")
    try:
        decoder_context = (
            (parent_dictionary if parent_dictionary is not None else dictionary)
            if descriptor is not None and descriptor.wants_image_dictionary
            else parms
        )
        result = internal_coerce_decoder_bytes(fn(data, decoder_context))
        if filter_name in PREDICTOR_FILTERS:
            if (
                allow_content_stream_passthrough
                and filter_name in {"FlateDecode", "Fl"}
                and result == data
                and looks_like_pdf_content_stream(result)
            ):
                return result
            result = internal_coerce_decoder_bytes(apply_predictor(result, parms))
        return result
    except ValueError as exc:
        raise FilterParseError("invalid stream data") from exc


def decode_stream_data(
    data: bytes | memoryview,
    dictionary: object | StreamDecodeSpec | None,
    *,
    parent_dictionary: object | None = None,
) -> bytes:
    if type(data) is memoryview:
        source_bytes = full_source_bytes(data)
        data = source_bytes if source_bytes is not None else data.tobytes()
    if dictionary is None:
        return data
    if isinstance(dictionary, StreamDecodeSpec):
        filters = dictionary.filters
        normalized_parms = dictionary.params
    else:
        spec = normalize_stream_decode_spec(dictionary)
        filters = spec.filters
        normalized_parms = spec.params
    if normalized_parms and len(normalized_parms) != len(filters):
        raise FilterParseError("invalid stream decode parameters")
    result = data
    for index, flt in enumerate(filters):
        parms = normalized_parms[index] if index < len(normalized_parms) else None
        result = decode_one_filter(
            result,
            flt,
            parms,
            dictionary=dictionary,
            parent_dictionary=parent_dictionary,
            allow_content_stream_passthrough=len(filters) == 1,
        )
    return result


# Decoders whose native path is exactly "preallocate a shape, hand it the buffer".
# CCITT (needs FilterParams) and flate/lzw (decode then reshape) stay explicit below.
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
    """Select a safe native decoder and its optional preallocation shape.

    Missing dimensions are not rejection: JPEG/JPX can read them from their
    headers and CCITT uses DecodeParms. Raw flate/lzw still require a shape.
    """
    stream_spec = normalize_stream_decode_spec(dictionary)
    if len(stream_spec.filters) != 1 or (stream_spec.params and len(stream_spec.params) != 1):
        return None
    descriptor = FILTER_DESCRIPTOR_BY_NAME.get(stream_spec.filters[0])
    decoder = descriptor.decoder if descriptor is not None else None
    if decoder is None or not image_decode_is_identity(dictionary):
        return None
    spec = NATIVE_IMAGE_SPECS.get(decoder)
    if spec is None:
        return None
    image_dictionary = dictionary if isinstance(dictionary, dict) else {}
    color_space = image_dictionary.get("ColorSpace")
    if isinstance(color_space, (list, tuple, dict)):
        return None
    color_name = normalize_pdf_name(color_space) if color_space is not None else None
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
    return internal_NativeImagePlan(
        decoder, stream_spec.params[0] if stream_spec.params else None, shape
    )


def decode_stream_image_data(
    data: bytes | memoryview,
    dictionary: object,
) -> DecodedImage | None:
    """Decode supported image filters directly to native sample arrays.

    Ordinary stream decoding remains bytes-valued. This opt-in path is for
    image consumers that can preserve array-backed samples through rendering.
    """

    plan = internal_prepare_native_image(dictionary)
    if plan is None:
        return None
    decoder = plan.decoder
    params = plan.params
    output_shape = plan.output_shape
    source = data
    try:
        array_decoder = internal_NATIVE_ARRAY_DECODERS.get(decoder)
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
    """Return whether an image's PDF Decode array leaves samples unchanged."""

    decode = dictionary.get("Decode") if isinstance(dictionary, dict) else None
    if decode is None:
        return True
    if not isinstance(decode, (list, tuple)) or len(decode) == 0 or len(decode) % 2:
        return False
    for index in range(0, len(decode), 2):
        lower = decode[index]
        upper = decode[index + 1]
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            return False
        if float(lower) != 0.0 or float(upper) != 1.0:
            return False
    return True
