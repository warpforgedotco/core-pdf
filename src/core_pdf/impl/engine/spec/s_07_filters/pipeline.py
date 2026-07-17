# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import os
import threading
import typing
from collections import OrderedDict

if typing.TYPE_CHECKING:
    from typing import Callable

    FilterFn = Callable[[bytes, object], bytes]

from core_pdf.impl.engine.spec.s_07_filters.codecs import (
    apply_ascii85 as apply_ascii85,
)
from core_pdf.impl.engine.spec.s_07_filters.codecs import (
    apply_ascii_hex as apply_ascii_hex,
)
from core_pdf.impl.engine.spec.s_07_filters.codecs import (
    apply_lzw as apply_lzw,
)
from core_pdf.impl.engine.spec.s_07_filters.codecs import (
    apply_run_length as apply_run_length,
)
from core_pdf.impl.engine.spec.s_07_filters.decode_spec import (
    StreamDecodeSpec,
    normalize_stream_decode_spec,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_ccitt_fax as decode_ccitt_fax,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_crypt as decode_crypt,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_jbig2 as decode_jbig2,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_jpeg as decode_jpeg,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    decode_jpx as decode_jpx,
)
from core_pdf.impl.engine.spec.s_07_filters.decoders import (
    jpx_parent_uses_explicit_colorspace,
)
from core_pdf.impl.engine.spec.s_07_filters.flate import (
    apply_flate as apply_flate,
)
from core_pdf.impl.engine.spec.s_07_filters.flate import (
    looks_like_pdf_content_stream,
)
from core_pdf.impl.engine.spec.s_07_filters.predictors import (
    apply_predictor as apply_predictor,
)
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError

FILTER_MAP: dict[str, FilterFn] = {
    "FlateDecode": apply_flate,
    "Fl": apply_flate,
    "ASCIIHexDecode": apply_ascii_hex,
    "AHx": apply_ascii_hex,
    "ASCII85Decode": apply_ascii85,
    "A85": apply_ascii85,
    "RunLengthDecode": apply_run_length,
    "RL": apply_run_length,
    "LZWDecode": apply_lzw,
    "LZW": apply_lzw,
    "DCT": decode_jpeg,
    "CCITTFaxDecode": decode_ccitt_fax,
    "CCF": decode_ccitt_fax,
    "DCTDecode": decode_jpeg,
    "Crypt": decode_crypt,
    "JPXDecode": decode_jpx,
    "JBIG2Decode": decode_jbig2,
}

PREDICTOR_FILTERS = {"FlateDecode", "Fl", "LZWDecode", "LZW"}
EXPENSIVE_DECODE_CACHE_FILTERS = {
    "CCF",
    "CCITTFaxDecode",
    "DCT",
    "DCTDecode",
    "JBIG2Decode",
    "JPXDecode",
}
EXPENSIVE_DECODE_CACHE_MAX_BYTES = int(
    os.environ.get("CORE_PDF_EXPENSIVE_DECODE_CACHE_BYTES", str(384 * 1024 * 1024))
)
_EXPENSIVE_DECODE_CACHE: OrderedDict[tuple[object, ...], tuple[bytes, bytes]] = OrderedDict()
_EXPENSIVE_DECODE_CACHE_BYTES = 0
_EXPENSIVE_DECODE_CACHE_LOCK = threading.Lock()


def expensive_decode_cache_key(
    data: bytes,
    filters: typing.Sequence[str],
    params: typing.Sequence[object],
    context: object = None,
) -> tuple[object, ...]:
    return (
        id(data),
        len(data),
        tuple(filters),
        repr(params),
        context,
    )


def cached_expensive_decode(key: tuple[object, ...], data: bytes) -> bytes | None:
    with _EXPENSIVE_DECODE_CACHE_LOCK:
        entry = _EXPENSIVE_DECODE_CACHE.get(key)
        if entry is None:
            return None
        source, decoded = entry
        if source is not data:
            _EXPENSIVE_DECODE_CACHE.pop(key, None)
            return None
        _EXPENSIVE_DECODE_CACHE.move_to_end(key)
        return decoded


def store_expensive_decode(key: tuple[object, ...], data: bytes, decoded: bytes) -> None:
    entry_size = len(data) + len(decoded)
    if EXPENSIVE_DECODE_CACHE_MAX_BYTES <= 0 or entry_size > EXPENSIVE_DECODE_CACHE_MAX_BYTES:
        return
    global _EXPENSIVE_DECODE_CACHE_BYTES
    with _EXPENSIVE_DECODE_CACHE_LOCK:
        previous = _EXPENSIVE_DECODE_CACHE.pop(key, None)
        if previous is not None:
            _EXPENSIVE_DECODE_CACHE_BYTES -= len(previous[0]) + len(previous[1])
        _EXPENSIVE_DECODE_CACHE[key] = (data, decoded)
        _EXPENSIVE_DECODE_CACHE_BYTES += entry_size
        while (
            _EXPENSIVE_DECODE_CACHE
            and _EXPENSIVE_DECODE_CACHE_BYTES > EXPENSIVE_DECODE_CACHE_MAX_BYTES
        ):
            ignored_key, evicted = _EXPENSIVE_DECODE_CACHE.popitem(last=False)
            _EXPENSIVE_DECODE_CACHE_BYTES -= len(evicted[0]) + len(evicted[1])


def decode_stream_data(
    data: bytes | memoryview,
    dictionary: object | StreamDecodeSpec | None,
    *,
    parent_dictionary: object | None = None,
) -> bytes:
    if type(data) is memoryview:
        source = data.obj
        data = (
            source
            if (
                type(source) is bytes
                and data.c_contiguous
                and data.itemsize == 1
                and data.nbytes == len(source)
            )
            else data.tobytes()
        )
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
        raise PdfParseError("invalid stream decode parameters")
    parent_context = parent_dictionary if parent_dictionary is not None else dictionary
    cache_context = (
        jpx_parent_uses_explicit_colorspace(parent_context) if "JPXDecode" in filters else None
    )
    decode_cache_key = (
        expensive_decode_cache_key(data, filters, normalized_parms, cache_context)
        if EXPENSIVE_DECODE_CACHE_FILTERS.intersection(filters)
        else None
    )
    if decode_cache_key is not None:
        cached = cached_expensive_decode(decode_cache_key, data)
        if cached is not None:
            return cached

    if len(filters) == 1:
        flt = filters[0]
        if flt in {"None", "Identity"}:
            return data
        fn = FILTER_MAP.get(flt)
        if fn is None:
            raise PdfUnsupportedError(f"stream filter {flt} is not implemented yet")
        parms = normalized_parms[0] if normalized_parms else None
        try:
            decoder_context = (
                (parent_dictionary if parent_dictionary is not None else dictionary)
                if flt == "JPXDecode"
                else parms
            )
            result = fn(data, decoder_context)
            result_type = type(result)
            if result_type is bytearray:
                result = bytes(result)
            elif result_type is not bytes:
                raise ValueError("invalid stream decoder result type")
            if flt in PREDICTOR_FILTERS:
                if (
                    flt in {"FlateDecode", "Fl"}
                    and result == data
                    and looks_like_pdf_content_stream(result)
                ):
                    return result
                result = apply_predictor(result, parms)
                result_type = type(result)
                if result_type is bytearray:
                    result = bytes(result)
                elif result_type is not bytes:
                    raise ValueError("invalid stream decoder result type")
            if decode_cache_key is not None:
                store_expensive_decode(decode_cache_key, data, result)
            return result
        except ValueError as exc:
            raise PdfParseError("invalid stream data") from exc

    result = data
    for index, flt in enumerate(filters):
        parms = normalized_parms[index] if index < len(normalized_parms) else None
        if flt in {"None", "Identity"}:
            continue
        fn = FILTER_MAP.get(flt)
        if fn is None:
            raise PdfUnsupportedError(f"stream filter {flt} is not implemented yet")
        try:
            decoder_context = (
                (parent_dictionary if parent_dictionary is not None else dictionary)
                if flt == "JPXDecode"
                else parms
            )
            result = fn(result, decoder_context)
            result_type = type(result)
            if result_type is bytearray:
                result = bytes(result)
            elif result_type is not bytes:
                raise ValueError("invalid stream decoder result type")
            if flt in PREDICTOR_FILTERS:
                result = apply_predictor(result, parms)
                result_type = type(result)
                if result_type is bytearray:
                    result = bytes(result)
                elif result_type is not bytes:
                    raise ValueError("invalid stream decoder result type")
        except ValueError as exc:
            raise PdfParseError("invalid stream data") from exc
    if decode_cache_key is not None:
        store_expensive_decode(decode_cache_key, data, result)
    return result
