# SPDX-License-Identifier: AGPL-3.0-only
"""Specification filter ordering with explicitly supplied external decoders."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from core_pdf_spec.s_07_filters.codecs import (
    apply_ascii85,
    apply_ascii_hex,
    apply_flate,
    apply_lzw,
    apply_run_length,
)
from core_pdf_spec.s_07_filters.decode_spec import (
    StreamDecodeSpec,
    normalize_stream_decode_spec,
)
from core_pdf_spec.s_07_filters.errors import FilterUnsupportedError
from core_pdf_spec.s_07_filters.jbig2 import decode_jbig2
from core_pdf_spec.s_07_filters.predictors import apply_predictor
from core_pdf_spec.s_07_filters.registry import FILTER_DESCRIPTOR_BY_NAME, PREDICTOR_FILTERS

FilterFn = Callable[[bytes, object], bytes]
INTERNAL_DECODERS: Mapping[str, FilterFn] = {
    "flate": apply_flate,
    "ascii_hex": apply_ascii_hex,
    "ascii85": apply_ascii85,
    "run_length": apply_run_length,
    "lzw": apply_lzw,
    "jbig2": decode_jbig2,
}


def decode_stream_data(
    data: bytes | memoryview,
    dictionary: object | StreamDecodeSpec | None,
    *,
    parent_dictionary: object | None = None,
    filter_decoders: Mapping[str, FilterFn] | None = None,
) -> bytes:
    result = bytes(data)
    if dictionary is None:
        return result
    spec = (
        dictionary
        if isinstance(dictionary, StreamDecodeSpec)
        else normalize_stream_decode_spec(dictionary)
    )
    decoders = INTERNAL_DECODERS if filter_decoders is None else filter_decoders
    for step in spec.steps:
        name = step.name
        descriptor = FILTER_DESCRIPTOR_BY_NAME.get(name)
        decoder = (
            decoders.get(descriptor.decoder)
            if descriptor is not None and descriptor.decoder is not None
            else None
        )
        if decoder is None:
            raise FilterUnsupportedError(f"stream filter {name} is not implemented")
        params = step.params
        context = (
            parent_dictionary
            if descriptor is not None
            and descriptor.wants_image_dictionary
            and parent_dictionary is not None
            else params
        )
        result = decoder(result, context)
        if name in PREDICTOR_FILTERS:
            result = apply_predictor(result, params)
    return result


__all__ = (
    "FilterFn",
    "decode_stream_data",
)
