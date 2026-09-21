# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    FilterFn = Callable[[bytes, object], bytes]

from core_pdf.impl._impl.graphics.codec_dispatch import (
    decode_ccitt_fax,
    decode_crypt,
    decode_jbig2,
    decode_jpeg,
    decode_jpx,
)
from core_pdf.impl._impl.graphics.decode_compat import normalize_stream_decode_spec
from core_pdf.impl._impl.graphics.filter_recovery import (
    apply_ascii85,
    apply_ascii_hex,
    apply_flate,
    apply_lzw,
    apply_run_length,
    looks_like_pdf_content_stream,
)
from core_pdf.impl._impl.graphics.filter_registry import (
    FILTER_DESCRIPTOR_BY_NAME,
    FILTER_DESCRIPTORS,
    PREDICTOR_FILTERS,
)
from core_pdf.impl._impl.graphics.predictor_backends import apply_predictor
from core_pdf_spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf_spec.s_07_filters.errors import FilterParseError, FilterUnsupportedError
from core_pdf_spec.s_07_syntax_primitives.scanning import full_source_bytes

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
    spec = (
        dictionary
        if isinstance(dictionary, StreamDecodeSpec)
        else normalize_stream_decode_spec(dictionary)
    )
    result = data
    for step in spec.steps:
        result = decode_one_filter(
            result,
            step.name,
            step.params,
            dictionary=dictionary,
            parent_dictionary=parent_dictionary,
            allow_content_stream_passthrough=len(spec.steps) == 1,
        )
    return result
