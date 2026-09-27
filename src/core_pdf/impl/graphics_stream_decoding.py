# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import binascii
import zlib
from collections.abc import Callable
from typing import Any

import imagecodecs
import numpy

import core_pdf_spec.s_07_filters.codecs as strict
from core_jbig2.bitmap import compose_packed_bitmap_data
from core_jbig2.codec import (
    GENERIC_TEMPLATE_0_DEFAULT_AT,
    JBIG2_IMMEDIATE_GENERIC_REGION,
    JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION,
    JBIG2_IMMEDIATE_TEXT_REGION,
    JBIG2_PAGE_INFO,
    JBIG2GenericRegionHeader,
    JBIG2PageDecoder,
    Jbig2ParseError,
    JBIG2Region,
    Jbig2UnsupportedError,
    compose_packed_bitmap_region,
)
from core_pdf.impl import graphics_codec_backends as codec_backends
from core_pdf.impl.graphics_codec_backends import (
    png_predict_codec,
    tiff_predict_codec,
)
from core_pdf.impl.graphics_decode_compat import (
    FilterParams,
    filter_params,
    normalize_stream_decode_spec,
)
from core_pdf.impl.graphics_filter_registry import (
    CCITT_IMAGE,
    FILTER_DESCRIPTOR_BY_NAME,
    FILTER_DESCRIPTORS,
    JPEG_IMAGE,
    JPX_IMAGE,
    PREDICTOR_FILTERS,
    RAW_SAMPLE_IMAGE,
    FilterDecoder,
    NativeImageCodec,
    TolerantFilter,
)
from core_pdf_cythonized import decode_arithmetic_generic_template0
from core_pdf_spec.s_07_filters.decode_spec import FilterParams as PdfFilterParams
from core_pdf_spec.s_07_filters.decode_spec import StreamDecodeSpec
from core_pdf_spec.s_07_filters.errors import (
    FilterParseError,
    FilterUnsupportedError,
    PredictorError,
)
from core_pdf_spec.s_07_filters.jbig2 import decode_jbig2 as decode_strict_jbig2
from core_pdf_spec.s_07_filters.predictors import (
    SUPPORTED_PREDICTOR_BITS,
    png_predict,
    tiff_predict,
)
from core_pdf_spec.s_07_filters.predictors import (
    apply_predictor as strict_apply_predictor,
)
from core_pdf_spec.s_07_syntax_primitives.coercion import is_pdf_null
from core_pdf_spec.s_07_syntax_primitives.content_operators import PDF_CONTENT_OPERATOR_BYTES
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    full_source_bytes,
    skip_comment,
    skip_hex_string,
    skip_literal_string,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import (
    DELIMITERS,
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
)


def coerce_decoder_bytes(result: object) -> bytes:
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
    tolerant = TOLERANT_FILTER_BY_NAME.get(filter_name)
    if tolerant is None:
        raise FilterUnsupportedError(f"stream filter {filter_name} is not implemented yet")
    try:
        decoder_context = (
            (parent_dictionary if parent_dictionary is not None else dictionary)
            if descriptor is not None and descriptor.wants_image_dictionary
            else parms
        )
        passed_through = False
        passthrough = tolerant.passthrough
        if passthrough is not None:
            inflated, passed_through = passthrough(data)
            result = coerce_decoder_bytes(inflated)
        else:
            result = coerce_decoder_bytes(tolerant.decode(data, decoder_context))
        if filter_name in PREDICTOR_FILTERS and not (
            passed_through and allow_content_stream_passthrough
        ):
            result = coerce_decoder_bytes(apply_predictor(result, parms))
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


class RecoveryJBIG2PageDecoder(JBIG2PageDecoder):
    supported_segment_types = frozenset(
        {
            JBIG2_PAGE_INFO,
            JBIG2_IMMEDIATE_TEXT_REGION,
            JBIG2_IMMEDIATE_GENERIC_REGION,
            JBIG2_IMMEDIATE_LOSSLESS_GENERIC_REGION,
        }
    )

    def decode_text_region(self, region: JBIG2Region) -> None:
        image = self.image
        if image is None:
            return
        if len(region.raw) < 20:
            raise Jbig2ParseError("truncated JBIG2 text region")
        bitmap = region.raw[20:]
        row_bytes = max(1, (region.width + 7) // 8)
        compose_packed_bitmap_data(
            bitmap,
            min(region.height, len(bitmap) // row_bytes),
            region.width,
            region.x,
            region.y,
            image.width,
            image.height,
            image.stride,
            image.data,
            0,
        )

    def decode_generic_region(self, header: JBIG2GenericRegionHeader) -> None:
        region = header.region
        if region.width <= 0 or region.height <= 0:
            return
        if header.mmr:
            if self.image is not None:
                compose_packed_bitmap_region(
                    region,
                    region.raw[header.bitmap_start :],
                    self.image,
                    self.page_info,
                )
            return
        if (
            header.template != 0
            or header.prediction
            or header.adaptive_pixels != GENERIC_TEMPLATE_0_DEFAULT_AT
        ):
            raise Jbig2UnsupportedError("unsupported JBIG2 generic bitmap template")
        bitmap = decode_arithmetic_generic_template0(
            region.raw[header.bitmap_start :], region.width, region.height
        )
        if self.image is not None:
            compose_packed_bitmap_region(region, bitmap, self.image, self.page_info)


def decode_jpeg_image(
    data: bytes | memoryview, *, out: numpy.ndarray[Any, Any] | None = None
) -> numpy.ndarray[Any, Any]:
    return codec_backends.decode_jpeg_image(data, out=out)


def decode_jpx_image(
    data: bytes | memoryview,
    *,
    out: numpy.ndarray[Any, Any] | None = None,
    preserve_precision: bool = False,
) -> numpy.ndarray[Any, Any]:
    return codec_backends.decode_jpx_image(data, out=out, preserve_precision=preserve_precision)


def decode_jpeg(data: bytes, parms: object) -> bytes:
    return decode_jpeg_image(data).tobytes()


def decode_jpx(data: bytes, parms: object) -> bytes:
    return decode_jpx_image(data).tobytes()


def decode_ccitt_fax_image(
    data: bytes | memoryview, parms: FilterParams, *, out: numpy.ndarray[Any, Any] | None = None
) -> numpy.ndarray[Any, Any]:
    array = codec_backends.decode_ccitt_fax_image(
        data,
        width=parms.columns if parms.has_columns else 1728,
        height=parms.rows,
        group4=parms.k < 0,
        t4options=(1 if parms.k > 0 else 0) | (4 if parms.encoded_byte_align else 0),
        out=out,
    )
    if not parms.black_is_1:
        numpy.bitwise_xor(array, 1, out=array)
    numpy.multiply(array, 255, out=array)
    return array


def decode_ccitt_fax(data: bytes, parms: object) -> bytes:
    params = filter_params(parms)
    return numpy.packbits(
        decode_ccitt_fax_image(data, params) != 0, axis=1, bitorder="big"
    ).tobytes()


def decode_crypt(data: bytes, parms: object) -> bytes:
    if is_pdf_null(parms):
        return data
    if not isinstance(parms, dict):
        raise FilterParseError("invalid Crypt filter params")
    return data


def decode_jbig2(data: bytes, parms: object) -> bytes:
    params = filter_params(parms)
    return decode_strict_jbig2(data, params, decoder_type=RecoveryJBIG2PageDecoder)


def png_predict_tolerant(
    data: bytes | memoryview,
    *,
    columns: int,
    colors: int,
    bits_per_component: int,
    damaged_rows_before_error: int = 0,
) -> bytes:
    if bits_per_component not in SUPPORTED_PREDICTOR_BITS:
        raise PredictorError(f"invalid PNG predictor bits {bits_per_component}")
    decoded = png_predict_codec(
        data, columns=columns, colors=colors, bits_per_component=bits_per_component
    )
    if decoded is not None:
        return decoded
    stride = max(1, (colors * columns * bits_per_component + 7) // 8) + 1
    stop = len(data) // stride * stride
    if damaged_rows_before_error:
        for start in range(0, stop, stride):
            if data[start] > 4:
                stop = start
                break
    return png_predict(
        data[:stop], columns=columns, colors=colors, bits_per_component=bits_per_component
    )


def png_predictor_tolerant(data: bytes | memoryview, params: PdfFilterParams) -> bytes:
    return png_predict_tolerant(
        data,
        columns=params.columns,
        colors=params.colors,
        bits_per_component=params.bits_per_component,
        damaged_rows_before_error=params.damaged_rows_before_error,
    )


def tiff_predictor_tolerant(data: bytes | memoryview, params: PdfFilterParams) -> bytes:
    columns = params.columns
    colors = params.colors
    bits_per_component = params.bits_per_component
    decoded = tiff_predict_codec(
        data, columns=columns, colors=colors, bits_per_component=bits_per_component
    )
    if decoded is not None:
        return decoded
    return tiff_predict(data, columns=columns, colors=colors, bits_per_component=bits_per_component)


def apply_predictor(data: bytes | memoryview, parms: object) -> bytes:
    params = filter_params(parms)
    return strict_apply_predictor(
        data, params, png=png_predictor_tolerant, tiff=tiff_predictor_tolerant
    )


ASCII_HEX_DIGITS = b"0123456789ABCDEFabcdef"
ASCII_HEX_INVALID_BYTES = bytes(byte for byte in range(256) if byte not in ASCII_HEX_DIGITS)
MIN_TRUNCATED_RAW_FLATE_BYTES = 8


def apply_ascii_hex(data: bytes, parms: object) -> bytes:
    terminator = data.find(b">")
    if terminator >= 0:
        data = data[:terminator]
    filtered = data.translate(None, ASCII_HEX_INVALID_BYTES)
    if len(filtered) & 1:
        filtered += b"0"
    return binascii.unhexlify(filtered)


def apply_run_length(data: bytes, parms: object) -> bytes:
    try:
        return strict.apply_run_length(data, parms)
    except strict.IncompleteRunLengthError as exc:
        return exc.decoded


def apply_ascii85(data: bytes | memoryview, parms: object) -> bytes:
    clean = bytes(data).lstrip(WHITESPACE)
    if clean.startswith(b"<~"):
        clean = clean[2:]
    if b"~>" not in clean:
        clean += b"~>"
    return strict.apply_ascii85(clean, parms)


def apply_lzw(data: bytes | memoryview, parms: object) -> bytes:
    params = filter_params(parms)
    if params.early_change == 1 and imagecodecs.LZW.available:
        try:
            return bytes(imagecodecs.lzw_decode(data))
        except Exception as exc:
            raise ValueError("invalid LZW stream") from exc
    try:
        return strict.apply_lzw(data, params)
    except strict.IncompleteLzwError as exc:
        return exc.decoded


def apply_flate(data: bytes, parms: object) -> bytes:
    return inflate(data)[0]


def inflate(data: bytes) -> tuple[bytes, bool]:
    if not data:
        return b"", False
    candidates: tuple[int, ...]
    if len(data) >= 2:
        cmf = data[0]
        flg = data[1]
        if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
            candidates = (zlib.MAX_WBITS, zlib.MAX_WBITS | 32, -15)
        elif cmf == 0x1F and flg == 0x8B:
            candidates = (zlib.MAX_WBITS | 32, zlib.MAX_WBITS, -15)
        else:
            candidates = (-15, zlib.MAX_WBITS | 32, zlib.MAX_WBITS)
    else:
        candidates = (-15, zlib.MAX_WBITS | 32, zlib.MAX_WBITS)

    tried_default = candidates[0] == zlib.MAX_WBITS
    if tried_default:
        try:
            return zlib.decompress(data, zlib.MAX_WBITS), False
        except zlib.error:
            pass
        try:
            return bytes(imagecodecs.zlib_decode(data)), False
        except Exception:
            pass

    for wbits in candidates:
        if not (tried_default and wbits == zlib.MAX_WBITS):
            try:
                return zlib.decompress(data, wbits), False
            except zlib.error:
                pass
        recovered = recover_flate(data, wbits)
        if recovered is not None:
            return recovered, False

    if looks_like_pdf_content_stream(data):
        return bytes(data), True
    raise FilterParseError("invalid FlateDecode stream")


def recover_flate(data: bytes, wbits: int = zlib.MAX_WBITS) -> bytes | None:
    try:
        decoder = zlib.decompressobj(wbits)
        decoded = decoder.decompress(data) + decoder.flush()
        minimum = MIN_TRUNCATED_RAW_FLATE_BYTES if wbits < 0 else 2
        if not decoder.eof and len(data) < minimum:
            return None
        return decoded
    except zlib.error:
        if wbits > 0 and len(data) > 6:
            cmf = data[0]
            flg = data[1]
            if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
                try:
                    return zlib.decompress(data[2:-4], -15)
                except zlib.error:
                    return None
        return None


def looks_like_pdf_content_stream(data: bytes | memoryview) -> bool:
    data_len = len(data)
    scan_limit = min(data_len, 1024)
    source_bytes = full_source_bytes(data)
    raw = source_bytes[:scan_limit] if source_bytes is not None else bytes(data[:scan_limit])
    end = len(raw)
    pos = 0
    token_count = 0
    container_depth = 0
    while pos < end and token_count < 64:
        byte = raw[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        if byte == 37:
            pos = skip_comment(raw, pos, end)
            continue
        if byte == 40:
            pos = skip_literal_string(raw, pos, end)
            continue
        if byte == 60:
            if pos + 1 < end and raw[pos + 1] == 60:
                container_depth += 1
                pos += 2
            else:
                pos = skip_hex_string(raw, pos, end)
            continue
        if byte == 62 and pos + 1 < end and raw[pos + 1] == 62:
            container_depth = max(0, container_depth - 1)
            pos += 2
            continue
        if byte == 91:
            container_depth += 1
            pos += 1
            continue
        if byte == 93:
            container_depth = max(0, container_depth - 1)
            pos += 1
            continue
        if byte == 47:
            pos += 1
            while pos < end:
                byte = raw[pos]
                if SEPARATOR_TABLE[byte]:
                    break
                pos += 1
            continue
        if byte in DELIMITERS:
            pos += 1
            continue
        start = pos
        while pos < end:
            byte = raw[pos]
            if SEPARATOR_TABLE[byte]:
                break
            pos += 1
        token_count += 1
        if container_depth == 0 and raw[start:pos] in PDF_CONTENT_OPERATOR_BYTES:
            return True
    return False


def decode_native_jpx(
    data: bytes | memoryview,
    params: object,
    output_shape: tuple[int, ...] | None,
    decoded: Callable[[], bytes],
) -> numpy.ndarray[Any, Any] | None:
    return decode_jpx_image(data, preserve_precision=True)


def decode_native_jpeg(
    data: bytes | memoryview,
    params: object,
    output_shape: tuple[int, ...] | None,
    decoded: Callable[[], bytes],
) -> numpy.ndarray[Any, Any] | None:
    output = numpy.empty(output_shape, dtype=numpy.uint8) if output_shape else None
    return decode_jpeg_image(data, out=output)


def decode_native_ccitt(
    data: bytes | memoryview,
    params: object,
    output_shape: tuple[int, ...] | None,
    decoded: Callable[[], bytes],
) -> numpy.ndarray[Any, Any] | None:
    ccitt_params = filter_params(params)
    output = (
        numpy.empty((ccitt_params.rows, ccitt_params.columns), dtype=numpy.uint8)
        if ccitt_params.rows > 0 and ccitt_params.columns > 0
        else None
    )
    return decode_ccitt_fax_image(data, ccitt_params, out=output)


def decode_native_samples(
    data: bytes | memoryview,
    params: object,
    output_shape: tuple[int, ...] | None,
    decoded: Callable[[], bytes],
) -> numpy.ndarray[Any, Any] | None:
    if output_shape is None:
        return None
    samples = decoded()
    expected_size = int(numpy.prod(output_shape, dtype=numpy.int64))
    if len(samples) != expected_size:
        return None
    return numpy.frombuffer(samples, dtype=numpy.uint8).reshape(output_shape)


NATIVE_SAMPLES = NativeImageCodec(RAW_SAMPLE_IMAGE, decode_native_samples)

TOLERANT_FILTERS: dict[FilterDecoder, TolerantFilter] = {
    "flate": TolerantFilter("flate", apply_flate, passthrough=inflate, native=NATIVE_SAMPLES),
    "ascii_hex": TolerantFilter("ascii_hex", apply_ascii_hex),
    "ascii85": TolerantFilter("ascii85", apply_ascii85),
    "run_length": TolerantFilter("run_length", apply_run_length),
    "lzw": TolerantFilter("lzw", apply_lzw, native=NATIVE_SAMPLES),
    "jpeg": TolerantFilter(
        "jpeg", decode_jpeg, native=NativeImageCodec(JPEG_IMAGE, decode_native_jpeg)
    ),
    "ccitt": TolerantFilter(
        "ccitt", decode_ccitt_fax, native=NativeImageCodec(CCITT_IMAGE, decode_native_ccitt)
    ),
    "crypt": TolerantFilter("crypt", decode_crypt),
    "jpx": TolerantFilter(
        "jpx",
        decode_jpx,
        native=NativeImageCodec(
            JPX_IMAGE, decode_native_jpx, requires_identity_decode=False, after_filters=True
        ),
    ),
    "jbig2": TolerantFilter("jbig2", decode_jbig2),
}

TOLERANT_FILTER_BY_NAME: dict[str, TolerantFilter] = {
    descriptor.name: TOLERANT_FILTERS[descriptor.decoder]
    for descriptor in FILTER_DESCRIPTORS
    if descriptor.decoder is not None
}
