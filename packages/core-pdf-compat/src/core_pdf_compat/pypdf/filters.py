# The decoders' error, limit and logging behaviour is ported from pypdf, Copyright (c)
# 2006, Mathieu Fenniak and 2006-2008, Martin Thoma, under its BSD-style license.
# core_pdf_spec does the LZW and predictor decoding where the two agree.
"""Stream filters with pypdf's signatures and limits."""

from __future__ import annotations

import binascii
import logging
import math
import os
import shutil
import struct
import subprocess
import warnings
import zlib
from base64 import a85decode
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NoReturn

from core_pdf_compat.pypdf.errors import (
    DependencyError,
    DeprecationError,
    LimitReachedError,
    PdfReadError,
    PdfStreamError,
)
from core_pdf_compat.pypdf.generic import (
    IndirectObject,
    NullObject,
    StreamObject,
    is_null_or_none,
)
from core_pdf_spec.s_07_filters.codecs import (
    IncompleteLzwError,
    LzwOutputLimitError,
    apply_ascii_hex,
    apply_lzw,
    encode_flate,
)
from core_pdf_spec.s_07_filters.decode_spec import FilterParams
from core_pdf_spec.s_07_filters.errors import PredictorError, UnsupportedPngFilterError
from core_pdf_spec.s_07_filters.predictors import png_predict, tiff_predict
from core_records import GeneratedRecord

logger = logging.getLogger(__name__)

MAX_DECLARED_STREAM_LENGTH = 75_000_000
MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH = 75_000_000
JBIG2_MAX_OUTPUT_LENGTH = 75_000_000
LZW_MAX_OUTPUT_LENGTH = 75_000_000
RUN_LENGTH_MAX_OUTPUT_LENGTH = 75_000_000
ZLIB_MAX_OUTPUT_LENGTH = 75_000_000
ZLIB_MAX_RECOVERY_INPUT_LENGTH = 5_000_000
FLATE_MAX_COLUMNS = 250_000
FLATE_MAX_ROW_LENGTH = 4_000_000
FLATE_MAX_BUFFER_SIZE = 75_000_000
MAX_TAIL_CUT_OFF_BYTES = 8
ASCII_WHITESPACE = b"\x00\t\n\f\r "


def _resolved(value: Any) -> Any:
    return value.get_object() if isinstance(value, IndirectObject) else value


def _decompress_with_limit(data: bytes) -> bytes:
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(data, max_length=ZLIB_MAX_OUTPUT_LENGTH)
    if decompressor.unconsumed_tail:
        raise LimitReachedError(
            f"Limit reached while decompressing. "
            f"{len(decompressor.unconsumed_tail)} bytes remaining."
        )
    return result


def _decompress_bytewise(data: bytes) -> bytes:
    # Keeps whatever a damaged stream yields, in a larger window.
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 32)
    output = bytearray()
    remaining = ZLIB_MAX_OUTPUT_LENGTH
    known_errors: set[str] = set()
    for index in range(len(data)):
        try:
            decompressed = decompressor.decompress(data[index : index + 1], max_length=remaining)
        except zlib.error as error:
            if index > ZLIB_MAX_RECOVERY_INPUT_LENGTH:
                raise LimitReachedError(
                    f"Recovery limit reached while decompressing. "
                    f"{len(data) - index} bytes remaining."
                ) from error
            if str(error) not in known_errors:
                logger.warning(str(error))
                known_errors.add(str(error))
            continue
        output += decompressed
        remaining -= len(decompressed)
        if remaining <= 0:
            raise LimitReachedError(
                f"Limit reached while decompressing. {len(data) - index} bytes remaining."
            )
    return bytes(output)


def decompress(data: bytes) -> bytes:
    """Inflate ``data``, trying a cut tail and then a byte-wise pass on a damaged stream."""
    try:
        return _decompress_with_limit(data)
    except zlib.error:
        # Early Distiller and Pitstop versions add a few bad bytes after the stream.
        for cut in range(1, min(MAX_TAIL_CUT_OFF_BYTES + 1, len(data))):
            try:
                return _decompress_with_limit(data[:-cut])
            except zlib.error:
                pass
        return _decompress_bytewise(data)


def _predictor_parameters(parameters: Any) -> tuple[int, int, int]:
    def get(key: str, default: int) -> int:
        value = _resolved(parameters.get(key, default))
        if not isinstance(value, int) or value < 1:
            raise PdfReadError(f"Expected positive number for {key}, got {value}!")
        return value

    columns = get("/Columns", 1)
    if columns > FLATE_MAX_COLUMNS:
        raise LimitReachedError(
            f"Number of columns {columns} exceeds defined limit of {FLATE_MAX_COLUMNS}."
        )
    colors = get("/Colors", 1)
    if colors > 16:
        raise LimitReachedError(
            f"Color value {colors} exceeds limit of 16. "
            f"Please open an issue if this limits valid use cases."
        )
    bits = get("/BitsPerComponent", 8)
    if bits > 16:
        raise PdfReadError(f"More than 16 bits per component are not allowed: {bits}")
    return columns, colors, bits


def _undo_predictor(data: bytes, parameters: Any) -> bytes:
    predictor = _resolved(parameters.get("/Predictor", 1))
    if predictor == 1:
        return data
    columns, colors, bits = _predictor_parameters(parameters)
    row_length = math.ceil(columns * colors * bits / 8) + 1
    if row_length > FLATE_MAX_ROW_LENGTH:
        raise LimitReachedError(
            f"Row length of {row_length} exceeds defined limit of {FLATE_MAX_ROW_LENGTH}."
        )
    try:
        if predictor == 2:
            return tiff_predict(data, columns=columns, colors=colors, bits_per_component=bits)
        if not 10 <= predictor <= 15:
            raise PdfReadError(f"Unsupported flatedecode predictor {predictor!r}")
        if remainder := len(data) % row_length:
            logger.warning("Image data is not rectangular. Adding padding.")
            data += bytes(row_length - remainder)
        return png_predict(data, columns=columns, colors=colors, bits_per_component=bits)
    except UnsupportedPngFilterError as error:
        raise PdfReadError(str(error).replace("predictor filter", "filter")) from error
    except PredictorError as error:
        raise PdfReadError(str(error)) from error


class FlateDecode:
    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        decoded = decompress(data)
        parameters = _resolved(decode_parms)
        if not isinstance(parameters, dict):
            if not is_null_or_none(parameters):
                logger.warning(
                    "Detected invalid /DecodeParms, results might be incorrect: %s (type %s)",
                    decode_parms,
                    type(decode_parms).__name__,
                )
            return decoded
        return _undo_predictor(decoded, parameters)

    @staticmethod
    def encode(data: bytes, level: int = -1) -> bytes:
        return encode_flate(data, level)


class ASCIIHexDecode:
    @staticmethod
    def decode(data: str | bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        encoded = data.encode() if isinstance(data, str) else data
        if b">" not in encoded:
            logger.warning("missing EOD in ASCIIHexDecode, check if output is OK")
            encoded += b">"
        try:
            return apply_ascii_hex(encoded, None)
        except binascii.Error as error:
            raise PdfStreamError(
                f"Invalid hexadecimal character in ASCIIHexDecode stream: {error}"
            ) from error


class RunLengthDecode:
    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        output = bytearray()
        index = 0
        while True:
            if index >= len(data):
                logger.warning("missing EOD in RunLengthDecode, check if output is OK")
                break
            length = data[index]
            index += 1
            if length == 128:
                if index == len(data) - 1 and data[index:] == b"\n":
                    logger.warning("Found trailing newline in stream data, check if output is OK")
                elif index < len(data):
                    # Stopping here keeps the image readable, though it is likely damaged.
                    logger.warning("Early EOD in RunLengthDecode, check if output is OK")
                break
            if length < 128:
                output += data[index : index + length + 1]
                index += length + 1
            elif index >= len(data):
                logger.warning("Missing EOD in RunLengthDecode, check if output is OK")
                break
            else:
                output += data[index : index + 1] * (257 - length)
                index += 1
            if len(output) > RUN_LENGTH_MAX_OUTPUT_LENGTH:
                raise LimitReachedError("Limit reached while decompressing.")
        return bytes(output)


class LZWDecode:
    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        parameters = _resolved(decode_parms)
        parameters = parameters if isinstance(parameters, dict) else {}
        early_change = _resolved(parameters.get("/EarlyChange", 1))
        try:
            decoded = apply_lzw(
                data,
                FilterParams(early_change=early_change),
                max_output_length=LZW_MAX_OUTPUT_LENGTH,
            )
        except IncompleteLzwError as error:
            decoded = error.decoded  # pypdf treats running out of data as the end marker.
        except LzwOutputLimitError as error:
            raise LimitReachedError(
                f"Limit reached while decompressing: {error.length} > {LZW_MAX_OUTPUT_LENGTH}"
            ) from error
        except ValueError as error:
            code = str(error).rpartition(" ")[2]
            raise PdfStreamError(
                f"LZW code {code} out of range with empty base at table index 258."
            ) from error
        return _undo_predictor(decoded, parameters)


class ASCII85Decode:
    @staticmethod
    def decode(data: str | bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        encoded = data.encode() if isinstance(data, str) else data
        encoded = encoded.strip(ASCII_WHITESPACE)
        if len(encoded) > 2 and encoded.endswith(b">"):
            encoded = encoded[:-1].rstrip(ASCII_WHITESPACE) + b">"
        try:
            return a85decode(encoded, adobe=True, ignorechars=ASCII_WHITESPACE)
        except ValueError as error:
            if error.args[0] != "Ascii85 encoded byte sequences must end with b'~>'":
                raise
        logger.warning("Ignoring missing Ascii85 end marker.")
        return a85decode(encoded, adobe=False, ignorechars=ASCII_WHITESPACE)


class DCTDecode:
    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        return data


class JPXDecode:
    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        return data


class CCITTParameters(GeneratedRecord):
    K: int = 0
    columns: int = 1728
    rows: int = 0
    EndOfLine: bool | None = False
    EncodedByteAlign: bool | None = False
    EndOfBlock: bool | None = True
    BlackIs1: bool = False
    DamagedRowsBeforeError: int | None = 0

    @property
    def group(self) -> int:
        return 4 if self.K < 0 else 3


def CCITParameters(*args: Any, **kwargs: Any) -> NoReturn:  # noqa: N802
    del args, kwargs
    raise DeprecationError(
        "CCITParameters is deprecated and was removed in pypdf 6.0.0. Use CCITTParameters instead."
    )


class CCITTFaxDecode:
    @staticmethod
    def _get_parameters(parameters: Any, rows: Any) -> CCITTParameters:
        found: dict[str, Any] = {}
        resolved = _resolved(parameters)
        for item in (resolved if isinstance(resolved, list) else [resolved]) if resolved else ():
            entry = _resolved(item)
            if is_null_or_none(entry):
                continue
            if "/K" in entry:
                found["K"] = _resolved(entry["/K"])
            if "/Columns" in entry:
                found["columns"] = _resolved(entry["/Columns"])
            if "/BlackIs1" in entry:
                found["BlackIs1"] = _resolved(entry["/BlackIs1"]).value
        return CCITTParameters(rows=int(_resolved(rows)), **found)

    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, height: int = 0, **kwargs: Any) -> bytes:  # noqa: ARG004
        params = CCITTFaxDecode._get_parameters(decode_parms, height)
        layout = "<2sHLH" + "HHLL" * 8 + "H"
        header = struct.pack(
            layout,
            b"II",  # Little endian
            42,  # TIFF version
            8,  # Offset of the first image file directory
            8,  # Number of tags
            *(256, 4, 1, params.columns),  # ImageWidth
            *(257, 4, 1, params.rows),  # ImageLength
            *(258, 3, 1, 1),  # BitsPerSample
            *(259, 3, 1, params.group),  # Compression
            *(262, 3, 1, int(params.BlackIs1)),  # PhotometricInterpretation
            *(273, 4, 1, struct.calcsize(layout)),  # StripOffsets
            *(278, 4, 1, params.rows),  # RowsPerStrip
            *(279, 4, 1, len(data)),  # StripByteCounts
            0,  # Last directory
        )
        return header + data


JBIG2DEC_BINARY = shutil.which("jbig2dec")


class JBIG2Decode:
    """jbig2dec, as in pypdf: its PNG output, logging and DependencyError are the public API."""

    @staticmethod
    def decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:  # noqa: ARG004
        if JBIG2DEC_BINARY is None:
            raise DependencyError("jbig2dec binary is not available.")
        with TemporaryDirectory() as directory:
            paths: list[Path] = []
            global_segments = _resolved(decode_parms.get("/JBIG2Globals")) if decode_parms else None
            if isinstance(global_segments, StreamObject):
                paths.append(Path(directory, "globals.jbig2"))
                paths[-1].write_bytes(global_segments.get_data())
            paths.append(Path(directory, "image.jbig2"))
            paths[-1].write_bytes(data)
            result = subprocess.run(  # noqa: S603
                [
                    JBIG2DEC_BINARY,
                    "--embedded",
                    "--format",
                    "png",
                    "--output",
                    "-",
                    "-M",
                    str(JBIG2_MAX_OUTPUT_LENGTH),
                    *paths,
                ],
                capture_output=True,
                env={**os.environ, "LC_ALL": "C"},
                check=False,
            )
        for option in (b"--embedded", b"-M"):
            if b"unrecognized option '" + option + b"'" in result.stderr:
                raise DependencyError("jbig2dec>=0.19 is required.")
        messages = result.stderr.decode("utf-8")
        if b"FATAL ERROR failed to allocate image data buffer" in result.stderr:
            raise LimitReachedError(f"Memory limit reached while reading JBIG2 data:\n{messages}")
        for line in messages.splitlines():
            logger.warning(line)
        if result.returncode != 0:
            raise PdfStreamError(f"Unable to decode JBIG2 data. Exit code: {result.returncode}")
        return result.stdout

    @staticmethod
    def _is_binary_compatible() -> bool:
        if not JBIG2DEC_BINARY:
            return False
        version = subprocess.run(  # noqa: S603
            [JBIG2DEC_BINARY, "--version"], capture_output=True, text=True, check=False
        ).stdout.split(" ", maxsplit=1)[1]
        return tuple(int(part) for part in version.split(".")[:2]) >= (0, 19)


FULL_FILTER_NAMES = {
    "/AHx": "/ASCIIHexDecode",
    "/A85": "/ASCII85Decode",
    "/LZW": "/LZWDecode",
    "/Fl": "/FlateDecode",
    "/RL": "/RunLengthDecode",
    "/CCF": "/CCITTFaxDecode",
    "/DCT": "/DCTDecode",
}


def decode_stream_data(stream: StreamObject) -> bytes:
    """Decode ``stream`` through its /Filter chain."""
    filters = _resolved(stream.get("/Filter", ()))
    if not isinstance(filters, (list, tuple)):
        filters = (filters,)
    parameters = _resolved(stream.get("/DecodeParms", ({},) * len(filters)))
    if not isinstance(parameters, (list, tuple)):
        parameters = (parameters,)
    data: bytes = stream._data
    if not data:
        return data
    for filter_item, parameter_item in zip(filters, parameters, strict=False):
        name, params = _resolved(filter_item), _resolved(parameter_item)
        if isinstance(params, NullObject):
            params = {}
        full_name = FULL_FILTER_NAMES.get(name, name)
        if name in FULL_FILTER_NAMES:
            warnings.warn(
                f"The filter name {name} is deprecated and will be removed in pypdf 7.0.0. "
                f"Use {full_name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if full_name == "/ASCIIHexDecode":
            data = ASCIIHexDecode.decode(data)
        elif full_name == "/ASCII85Decode":
            data = ASCII85Decode.decode(data)
        elif full_name == "/LZWDecode":
            data = LZWDecode.decode(data, params)
        elif full_name == "/FlateDecode":
            data = FlateDecode.decode(data, params)
        elif full_name == "/RunLengthDecode":
            data = RunLengthDecode.decode(data)
        elif full_name == "/CCITTFaxDecode":
            data = CCITTFaxDecode.decode(data, params, stream.get("/Height", 0))
        elif full_name in {"/DCTDecode", "/JPXDecode"}:
            pass  # pypdf leaves these encoded.
        elif full_name == "/JBIG2Decode":
            data = JBIG2Decode.decode(data, params)
        elif full_name == "/Crypt":
            if "/Name" in params or "/Type" in params:
                raise NotImplementedError("/Crypt filter with /Name or /Type not supported yet")
        else:
            raise NotImplementedError(f"Unsupported filter {name}")
    return data


__all__ = (
    "ASCII85Decode",
    "ASCIIHexDecode",
    "CCITParameters",
    "CCITTFaxDecode",
    "CCITTParameters",
    "DCTDecode",
    "FlateDecode",
    "JBIG2Decode",
    "JPXDecode",
    "LZWDecode",
    "RunLengthDecode",
    "decode_stream_data",
    "decompress",
)
