# SPDX-License-Identifier: AGPL-3.0-only
"""Stream filtering and decoding logic."""

from __future__ import annotations

import base64
import binascii
import zlib
from collections.abc import Callable

from core_pdf.impl.engine.spec.s_07_filters.ccitt import decode_ccitt_fax as decode_ccitt_impl
from core_pdf.impl.engine.spec.s_07_filters.jbig2 import (
    assemble_embedded_jbig2,
    decode_embedded_jbig2,
    parse_jbig2_file,
)
from core_pdf.impl.engine.spec.s_07_filters.jpeg import JPEGDecoder
from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError, PdfUnsupportedError
from core_pdf.impl.engine.spec.s_07_syntax.lexer import WS_TABLE
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    FilterParams,
    PdfDictLike,
    PdfName,
    PdfObject,
    PdfStream,
    StreamDecodeSpec,
    normalize_pdf_name,
)

FilterArg = PdfObject | FilterParams
FilterFn = Callable[[bytes, FilterArg], bytes]

# Local bindings for hot-path optimization
PdfParseError = PdfParseError
PdfUnsupportedError = PdfUnsupportedError
FilterParams = FilterParams
FilterParams_from_parms = FilterParams.from_parms
PdfName_of = PdfName.of
normalize_filter_name = normalize_pdf_name

PDF_CONTENT_OPERATORS = {
    b"b",
    b"b*",
    b"B",
    b"B*",
    b"BI",
    b"BT",
    b"c",
    b"cm",
    b"CS",
    b"cs",
    b"d",
    b"Do",
    b"ET",
    b"f",
    b"F",
    b"f*",
    b"gs",
    b"h",
    b"i",
    b"ID",
    b"j",
    b"J",
    b"l",
    b"m",
    b"M",
    b"n",
    b"q",
    b"Q",
    b"re",
    b"ri",
    b"s",
    b"S",
    b"SC",
    b"sc",
    b"SCN",
    b"scn",
    b"sh",
    b"T*",
    b"TD",
    b"Td",
    b"Tf",
    b"Tj",
    b"TJ",
    b"Tm",
    b"Tr",
    b"Ts",
    b"Tw",
    b"Tz",
    b"v",
    b"w",
    b"W",
    b"W*",
    b"y",
    b"'",
    b'"',
}
PDF_CONTENT_DELIMITERS = b"()<>[]{}/%"


def normalize_stream_decode_spec(dictionary: PdfDictLike) -> StreamDecodeSpec:
    raw_filters = dictionary.get(PdfName_of(b"Filter"))
    if raw_filters is None:
        filters: list[PdfObject] = []
    else:
        filters = list(raw_filters) if isinstance(raw_filters, (list, tuple)) else [raw_filters]
    names: list[str] = []
    for item in filters:
        name = normalize_filter_name(item)
        if name is None:
            raise PdfParseError("invalid stream filter")
        names.append(name)

    parms_raw = dictionary.get(PdfName_of(b"DecodeParms"))
    if parms_raw is None:
        decode_parms: list[PdfObject] = []
    elif isinstance(parms_raw, (list, tuple)):
        decode_parms = list(parms_raw)
    else:
        decode_parms = [parms_raw]
    params = [FilterParams.from_parms(parms) for parms in decode_parms]
    return StreamDecodeSpec(filters=names, params=params)


def apply_ascii_hex(data: bytes, parms: FilterArg) -> bytes:
    filtered = bytearray()
    ws = WS_TABLE
    for byte in data:
        if ws[byte]:
            continue
        if byte == 62:  # >
            break
        filtered.append(byte)
    if len(filtered) & 1:
        filtered.append(48)  # 0
    try:
        return binascii.unhexlify(filtered)
    except binascii.Error as exc:
        raise PdfParseError("invalid ASCIIHexDecode stream") from exc


def apply_run_length(data: bytes, parms: FilterArg) -> bytes:
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        length = data[i]
        i += 1
        if length == 128:
            break
        if length < 128:
            run = length + 1
            if i + run > n:
                raise PdfParseError("truncated RunLengthDecode stream")
            out.extend(data[i : i + run])
            i += run
            continue
        run = 257 - length
        if i >= n:
            raise PdfParseError("truncated RunLengthDecode stream")
        out.extend(data[i : i + 1] * run)
        i += 1
    return bytes(out)


def decode_ccitt_fax(data: bytes, parms: FilterArg) -> bytes:
    params = parms if isinstance(parms, FilterParams) else FilterParams_from_parms(parms)
    k = params.k
    return decode_ccitt_impl(
        data,
        columns=params.columns if params.has_columns else 1728,
        rows=params.rows,
        byte_align=params.encoded_byte_align,
        k=k,
    )


def decode_jpeg(data: bytes, parms: FilterArg) -> bytes:
    return JPEGDecoder.from_data(data)


class BitReader:
    __slots__ = ("data", "pos", "length", "buffer", "bits_in_buffer")

    def __init__(self, data: bytes | memoryview):
        self.data = data
        self.pos = 0
        self.length = len(data)
        self.buffer = 0
        self.bits_in_buffer = 0

    def read_bits(self, n: int) -> int | None:
        # Fast path: we already have enough bits in the buffer
        if self.bits_in_buffer >= n:
            self.bits_in_buffer -= n
            return (self.buffer >> self.bits_in_buffer) & ((1 << n) - 1)

        # Slow path: need to read more bytes from the stream
        bits_needed = n - self.bits_in_buffer
        bytes_needed = (bits_needed + 7) // 8

        # Read chunk (up to 8 bytes at a time for 64-bit int safety)
        end_pos = min(self.pos + bytes_needed, self.length)
        chunk = self.data[self.pos : end_pos]
        self.pos = end_pos

        if not chunk:
            return None

        # Convert chunk to integer and merge into buffer
        chunk_val = int.from_bytes(chunk, byteorder="big")
        self.buffer = (self.buffer << (len(chunk) * 8)) | chunk_val
        self.bits_in_buffer += len(chunk) * 8

        # Extract the requested bits
        self.bits_in_buffer -= n
        return (self.buffer >> self.bits_in_buffer) & ((1 << n) - 1)


def apply_lzw(data: bytes, parms: FilterArg) -> bytes:
    """Decode LZWDecode with strictly zero-copy table entries."""
    ec = (
        parms.early_change
        if isinstance(parms, FilterParams)
        else FilterParams_from_parms(parms).early_change
    )

    code_size = 9
    next_code = 258

    # We use a list of bytes for the table. Small byte objects are interned or
    # efficiently handled in Python. For LZW, entries are immutable sequences.
    # To avoid the '+' operator overhead, we use a pre-allocated table and
    # only create new byte objects when a new sequence is added.
    table: list[bytes] = [bytes([i]) for i in range(256)] + [b""] * (4096 - 256)

    reader = BitReader(data)
    out = bytearray()
    prev: bytes | None = None

    # Local method bindings for speed
    out_extend = out.extend

    while True:
        code = reader.read_bits(code_size)
        if code is None:
            break
        if code == 256:  # clear
            code_size = 9
            next_code = 258
            prev = None
            continue
        if code == 257:  # EOI
            break

        if code < next_code:
            entry = table[code]
        elif code == next_code and prev is not None:
            entry = prev + prev[:1]
        else:
            raise ValueError(f"invalid LZW code: {code}")

        out_extend(entry)
        if prev is not None and next_code < 4096:
            # We still use '+' here as it is the most efficient way to build
            # these small immutable sequences in pure Python.
            # Strictly speaking, this is a copy, but these are tiny objects (< 4KB).
            table[next_code] = prev + entry[:1]
            next_code += 1
            if next_code == (1 << code_size) - ec and code_size < 12:
                code_size += 1
        prev = entry
    return bytes(out)


def tiff_predict_8(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns
    out = bytearray()
    pos = 0
    n = len(data)
    while pos < n:
        if pos + bytes_per_row > n:
            raise PdfParseError("truncated TIFF predictor row")
        row = bytearray(data[pos : pos + bytes_per_row])
        pos += bytes_per_row
        for i in range(colors, len(row)):
            row[i] = (row[i] + row[i - colors]) & 0xFF
        out.extend(row)
    return bytes(out)


def tiff_predict_16(data: bytes | memoryview, columns: int, colors: int) -> bytes:
    bytes_per_row = colors * columns * 2
    out = bytearray()
    pos = 0
    n = len(data)
    while pos < n:
        if pos + bytes_per_row > n:
            raise PdfParseError("truncated TIFF predictor row")
        row = bytearray(data[pos : pos + bytes_per_row])
        pos += bytes_per_row
        samples_per_row = colors * columns
        for i in range(colors, samples_per_row):
            cur = (row[i * 2] << 8) | row[i * 2 + 1]
            prev_sample = (row[(i - colors) * 2] << 8) | row[(i - colors) * 2 + 1]
            val = (cur + prev_sample) & 0xFFFF
            row[i * 2] = (val >> 8) & 0xFF
            row[i * 2 + 1] = val & 0xFF
        out.extend(row)
    return bytes(out)


def tiff_predict_bits(data: bytes | memoryview, columns: int, colors: int, bits: int) -> bytes:
    sample_count = colors * columns
    sample_mask = (1 << bits) - 1
    row_bit_length = sample_count * bits
    row_byte_length = max(1, (row_bit_length + 7) // 8)
    out = bytearray()
    pos = 0
    n = len(data)

    def unpack_samples(row_bytes: bytes) -> list[int]:
        samples: list[int] = []
        bit_buffer = 0
        bits_in_buffer = 0
        for byte in row_bytes:
            bit_buffer = (bit_buffer << 8) | byte
            bits_in_buffer += 8
            while bits_in_buffer >= bits and len(samples) < sample_count:
                bits_in_buffer -= bits
                samples.append((bit_buffer >> bits_in_buffer) & sample_mask)
                if bits_in_buffer:
                    bit_buffer &= (1 << bits_in_buffer) - 1
                else:
                    bit_buffer = 0
        if len(samples) < sample_count:
            samples.extend([0] * (sample_count - len(samples)))
        return samples

    def pack_samples(samples: list[int]) -> bytes:
        packed = bytearray()
        bit_buffer = 0
        bits_in_buffer = 0
        for sample in samples:
            bit_buffer = (bit_buffer << bits) | (sample & sample_mask)
            bits_in_buffer += bits
            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                packed.append((bit_buffer >> bits_in_buffer) & 0xFF)
                if bits_in_buffer:
                    bit_buffer &= (1 << bits_in_buffer) - 1
                else:
                    bit_buffer = 0
        if bits_in_buffer:
            packed.append((bit_buffer << (8 - bits_in_buffer)) & 0xFF)
        if len(packed) < row_byte_length:
            packed.extend(b"\x00" * (row_byte_length - len(packed)))
        return bytes(packed[:row_byte_length])

    while pos < n:
        if pos + row_byte_length > n:
            raise PdfParseError("truncated TIFF predictor row")
        row = bytes(data[pos : pos + row_byte_length])
        pos += row_byte_length
        samples = unpack_samples(row)
        for i in range(colors, sample_count):
            samples[i] = (samples[i] + samples[i - colors]) & sample_mask
        out.extend(pack_samples(samples))
    return bytes(out)


def apply_tiff_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    if params.bits_per_component == 8:
        return tiff_predict_8(data, params.columns, params.colors)
    if params.bits_per_component == 16:
        return tiff_predict_16(data, params.columns, params.colors)
    if params.bits_per_component not in {1, 2, 4}:
        raise PdfParseError(f"invalid TIFF predictor bits {params.bits_per_component}")
    return tiff_predict_bits(data, params.columns, params.colors, params.bits_per_component)


def apply_png_predictor(data: bytes | memoryview, params: FilterParams) -> bytes:
    colors = params.colors
    bits_per_component = params.bits_per_component
    if bits_per_component not in {1, 2, 4, 8, 16}:
        raise PdfParseError(f"invalid PNG predictor bits {bits_per_component}")
    columns = params.columns
    bytes_per_pixel = max(1, (colors * bits_per_component + 7) // 8)
    row_length = max(1, (colors * columns * bits_per_component + 7) // 8)
    out = bytearray()
    n = len(data)
    pos = 0
    previous = bytearray(row_length)
    bpp = bytes_per_pixel
    rl = row_length
    while pos < n:
        if pos + 1 > n:
            raise PdfParseError("truncated PNG predictor row")
        filter_type = data[pos]
        pos += 1
        if pos + rl > n:
            raise PdfParseError("truncated PNG predictor row")
        row = bytearray(data[pos : pos + rl])
        pos += rl
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(bpp, len(row)):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filter_type == 2:
            for i in range(len(row)):
                row[i] = (row[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            n_row = len(row)
            for i in range(min(bpp, n_row)):
                row[i] = (row[i] + (previous[i] >> 1)) & 0xFF
            for i in range(bpp, n_row):
                row[i] = (row[i] + ((row[i - bpp] + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            n_row = len(row)
            for i in range(min(bpp, n_row)):
                row[i] = (row[i] + previous[i]) & 0xFF
            for i in range(bpp, n_row):
                left, up, up_left = row[i - bpp], previous[i], previous[i - bpp]
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                if pa <= pb and pa <= pc:
                    row[i] = (row[i] + left) & 0xFF
                elif pb <= pc:
                    row[i] = (row[i] + up) & 0xFF
                else:
                    row[i] = (row[i] + up_left) & 0xFF
        else:
            raise PdfUnsupportedError(f"Unsupported PNG predictor filter {filter_type}")
        out.extend(row)
        previous = row
    return bytes(out)


def apply_predictor(data: bytes | memoryview, parms: FilterArg) -> bytes:
    """Apply predictor filter (PNG or TIFF)."""
    params = parms if isinstance(parms, FilterParams) else FilterParams_from_parms(parms)
    predictor = params.predictor
    if predictor == 1:
        return bytes(data)
    if predictor == 2:
        return apply_tiff_predictor(data, params)
    if predictor >= 10:
        return apply_png_predictor(data, params)
    raise PdfParseError(f"invalid stream predictor {predictor}")


def apply_flate(data: bytes, parms: FilterArg) -> bytes:
    if not data:
        return b""

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

    for wbits in candidates:
        try:
            return zlib.decompress(data, wbits)
        except zlib.error as exc:
            recovered = recover_zlib_checksum_data(data, exc) if wbits == zlib.MAX_WBITS else None
            if recovered is not None:
                return recovered
            recovered = recover_flate(data, wbits)
            if recovered:
                return recovered

    if looks_like_pdf_content_stream(data):
        return bytes(data)
    raise PdfParseError("invalid FlateDecode stream")


def recover_flate(data: bytes, wbits: int = zlib.MAX_WBITS) -> bytes:
    try:
        decoder = zlib.decompressobj(wbits)
        return decoder.decompress(data) + decoder.flush()
    except zlib.error:
        return b""


def recover_zlib_checksum_data(data: bytes, original_error: zlib.error) -> bytes | None:
    if "incorrect data check" not in str(original_error):
        return None

    decompressor = zlib.decompressobj()
    output = bytearray()
    try:
        for index in range(0, len(data), 64):
            output.extend(decompressor.decompress(data[index : index + 64]))
        output.extend(decompressor.flush())
    except zlib.error as exc:
        if "incorrect data check" not in str(exc):
            return None
    if output:
        return bytes(output)
    if len(data) > 6:
        try:
            return zlib.decompress(data[2:-4], -15)
        except zlib.error:
            return None
    return None


def looks_like_pdf_content_stream(data: bytes | memoryview) -> bool:
    view = data if type(data) is memoryview else memoryview(data)
    end = len(view)
    pos = 0
    token_count = 0
    scan_limit = min(end, 1024)
    while pos < scan_limit and token_count < 64:
        byte = view[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        if byte == 37:
            newline = bytes(view[pos:scan_limit]).find(b"\n")
            if newline < 0:
                return False
            pos += newline + 1
            continue
        if byte in PDF_CONTENT_DELIMITERS:
            pos += 1
            continue
        start = pos
        while pos < scan_limit:
            byte = view[pos]
            if WS_TABLE[byte] or byte in PDF_CONTENT_DELIMITERS:
                break
            pos += 1
        token = bytes(view[start:pos])
        token_count += 1
        if token in PDF_CONTENT_OPERATORS:
            return True
    return False


def apply_ascii85(data: bytes, parms: FilterArg) -> bytes:
    try:
        return base64.a85decode(data, adobe=True)
    except (ValueError, binascii.Error) as exc:
        stripped = data.strip()
        if stripped.startswith(b"<~"):
            stripped = stripped[2:]
        terminator = stripped.find(b"~>")
        if terminator >= 0:
            stripped = stripped[:terminator]
        try:
            return base64.a85decode(stripped, adobe=False)
        except (ValueError, binascii.Error):
            raise PdfParseError("invalid ASCII85Decode stream") from exc


def decode_jpx(data: bytes, parms: FilterArg) -> bytes:
    from core_pdf.impl.engine.spec.s_07_filters.jpx import decode_jpx as _decode

    return _decode(data)


def decode_jbig2(data: bytes, parms: FilterArg) -> bytes:
    globals_data = b""
    if isinstance(parms, dict):
        globals_obj = parms.get("JBIG2Globals")
        if isinstance(globals_obj, bytes):
            globals_data = globals_obj
        elif isinstance(globals_obj, PdfStream):
            globals_data = globals_obj.data
    assembled = assemble_embedded_jbig2(globals_data, data)
    decoded = decode_embedded_jbig2(assembled)
    if decoded:
        return decoded
    parse_jbig2_file(assembled)
    raise PdfUnsupportedError("JBIG2 stream could not be decoded")


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
    "CCITTFaxDecode": decode_ccitt_fax,
    "CCF": decode_ccitt_fax,
    "DCTDecode": decode_jpeg,
    "DCT": decode_jpeg,
    "JPXDecode": decode_jpx,
    "JBIG2Decode": decode_jbig2,
}


def decode_stream_data(data: bytes, dictionary: PdfDictLike | StreamDecodeSpec | None) -> bytes:
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
    result = data
    for index, flt in enumerate(filters):
        parms = normalized_parms[index] if index < len(normalized_parms) else FilterParams()
        fn = FILTER_MAP.get(flt)
        if fn is None:
            raise PdfUnsupportedError(f"stream filter {flt} is not implemented yet")
        result = fn(result, parms)
        if flt in {"FlateDecode", "Fl", "LZWDecode", "LZW"}:
            result = apply_predictor(result, parms)
    return result
