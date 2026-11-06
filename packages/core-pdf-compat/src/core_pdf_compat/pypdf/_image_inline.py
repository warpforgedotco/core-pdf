# Ported from pypdf.generic._image_inline and ContentStream._read_inline_image of pypdf
# 6.15.0, Copyright (c) 2024, pypdf contributors, under its BSD-style license. Where an
# inline image's data ends is pypdf's guess, not the specification's, so it is followed
# step by step: images found with EI inside their data, or without a delimited EI, end
# where pypdf ends them.
"""How pypdf finds the end of an inline image's data in a content stream."""

from __future__ import annotations

import logging
from io import BytesIO
from math import ceil
from typing import IO, Any

from core_pdf_compat.pypdf._generic_read import read_non_whitespace
from core_pdf_compat.pypdf.errors import PdfReadError, PdfStreamError

logger = logging.getLogger(__name__)

WHITESPACES = (b"\x00", b"\t", b"\n", b"\f", b"\r", b" ")
WHITESPACES_AS_BYTES = b"".join(WHITESPACES)
# An inline image should be used only for small images (4096 bytes or less),
# but allow twice this for cases where this has been exceeded.
BUFFER_SIZE = 8192


def _check_end_image_marker(stream: IO[bytes]) -> bool:
    ei_tok = read_non_whitespace(stream)
    ei_tok += stream.read(2)
    stream.seek(-3, 1)
    return ei_tok[:2] == b"EI" and (ei_tok[2:3] == b"" or ei_tok[2:3] in WHITESPACES)


def extract_inline__ascii_hex_decode(stream: IO[bytes]) -> bytes:
    """Hex data, up to its > or else an EI. The stream is left on the EI."""
    data_out = bytearray()
    while True:
        data_buffered = read_non_whitespace(stream) + (read_bytes := stream.read(BUFFER_SIZE))
        if not data_buffered or not read_bytes:
            raise PdfReadError("Unexpected end of stream.")
        pos_tok = data_buffered.find(b">")
        if pos_tok >= 0:
            data_out += data_buffered[: pos_tok + 1]
            stream.seek(-len(data_buffered) + pos_tok + 1, 1)
            break
        pos_ei = data_buffered.find(b"EI")
        if pos_ei >= 0:
            stream.seek(-len(data_buffered) + pos_ei - 1, 1)
            c = stream.read(1)
            while c in WHITESPACES:
                stream.seek(-2, 1)
                c = stream.read(1)
                pos_ei -= 1
            data_out += data_buffered[:pos_ei]
            break
        if len(data_buffered) == 2:
            data_out += data_buffered
            raise PdfReadError("Unexpected end of stream.")
        data_out += data_buffered[:-2]
        stream.seek(-2, 1)

    if not _check_end_image_marker(stream):
        raise PdfReadError("EI stream not found.")
    return bytes(data_out)


def extract_inline__ascii85_decode(stream: IO[bytes]) -> bytes:
    """Ascii85 data, up to its ~>. The stream is left on the EI."""
    data_out = bytearray()
    while True:
        data_buffered = read_non_whitespace(stream) + (read_bytes := stream.read(BUFFER_SIZE))
        if not data_buffered or not read_bytes:
            raise PdfReadError("Unexpected end of stream.")
        pos_tok = data_buffered.find(b"~>")
        if pos_tok >= 0:
            data_out += data_buffered[: pos_tok + 2]
            stream.seek(-len(data_buffered) + pos_tok + 2, 1)
            break
        if len(data_buffered) == 2:
            data_out += data_buffered
            raise PdfReadError("Unexpected end of stream.")
        # Keep the last byte, in case it starts a ~>.
        data_out += data_buffered[:-2]
        stream.seek(-2, 1)

    if not _check_end_image_marker(stream):
        raise PdfReadError("EI stream not found.")
    return bytes(data_out)


def extract_inline__run_length_decode(stream: IO[bytes]) -> bytes:
    """Run-length data, up to its 128 end marker. The stream is left on the EI."""
    data_out = bytearray()
    while True:
        data_buffered = stream.read(BUFFER_SIZE)
        if not data_buffered:
            raise PdfReadError("Unexpected end of stream.")
        pos_tok = data_buffered.find(b"\x80")
        if pos_tok >= 0:
            # Some images carry several end markers (pypdf #3517): unless an EI follows
            # the first one, look for the EI instead.
            after_token = data_buffered[pos_tok + 1 : pos_tok + 4]
            if after_token.startswith(b"EI") or after_token.endswith(b"EI"):
                data_out += data_buffered[: pos_tok + 1]
                stream.seek(-len(data_buffered) + pos_tok + 1, 1)
            else:
                logger.warning("Early EOD in RunLengthDecode of inline image, using fallback.")
                ei_marker = data_buffered.find(b"EI")
                if ei_marker > 0:
                    data_out += data_buffered[:ei_marker]
                    stream.seek(-len(data_buffered) + ei_marker - 1, 1)
            break
        data_out += data_buffered

    if not _check_end_image_marker(stream):
        raise PdfReadError("EI stream not found.")
    return bytes(data_out)


def extract_inline__dct_decode(stream: IO[bytes]) -> bytes:
    """JPEG data, read marker by marker up to its end of image. The stream is left on the EI."""

    def read(length: int) -> bytes:
        result = stream.read(length)
        if result is None or len(result) != length:
            raise PdfReadError("Unexpected end of stream.")
        return result

    data_out = bytearray()
    # Blocks of ID, size and data, up to the ID FF D9:
    # https://www.digicamsoft.com/itu/itu-t81-36.html
    not_first = False
    while True:
        c = read(1)
        if not_first or (c == b"\xff"):
            data_out += c
        if c != b"\xff":
            continue
        not_first = True
        c = read(1)
        data_out += c
        if c == b"\xff":
            stream.seek(-1, 1)
        elif c == b"\x00":  # stuffing
            pass
        elif c == b"\xd9":  # end
            break
        elif c in (
            b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc9\xca\xcb\xcc\xcd\xce\xcf"
            b"\xda\xdb\xdc\xdd\xde\xdf"
            b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xfe"
        ):
            c = read(2)
            data_out += c
            sz = c[0] * 256 + c[1]
            data_out += read(sz - 2)

    if not _check_end_image_marker(stream):
        raise PdfReadError("EI stream not found.")
    return bytes(data_out)


def _is_preceded_by_whitespace(stream_out: BytesIO, position: int) -> bool:
    if position <= 0:
        return False
    saved_position = stream_out.tell()
    stream_out.seek(position - 1)
    is_whitespace = stream_out.read(1) in WHITESPACES
    stream_out.seek(saved_position)
    return is_whitespace


def _is_followed_by_closing_operator(stream: IO[bytes], first_byte: bytes) -> bool:
    """Whether the consumed ``first_byte`` starts an exact Q or EMC operator."""
    position = stream.tell()
    if first_byte == b"Q":
        suffix = stream.read(1)
        is_closing_operator = suffix == b"" or suffix in WHITESPACES
    elif first_byte == b"E":
        suffix = stream.read(3)
        is_closing_operator = suffix[:2] == b"MC" and (
            suffix[2:3] == b"" or suffix[2:3] in WHITESPACES
        )
    else:
        is_closing_operator = False
    stream.seek(position, 0)
    return is_closing_operator


def extract_inline_default(stream: IO[bytes]) -> bytes:
    """Data up to an EI that looks like the end of the image. The stream is left on the EI."""
    stream_out = BytesIO()
    while True:
        data_buffered = stream.read(BUFFER_SIZE)
        if not data_buffered:
            raise PdfReadError("Unexpected end of stream.")
        # Look for E alone, as the I may be in the next buffer.
        pos_ei = data_buffered.find(b"E")

        if pos_ei == -1:
            stream_out.write(data_buffered)
        else:
            # Write out everything up to and including the E, removed again at an EI.
            stream_out.write(data_buffered[0 : pos_ei + 1])
            sav_pos_ei = stream_out.tell() - 1
            stream.seek(pos_ei + 1 - len(data_buffered), 1)
            saved_pos = stream.tell()
            tok2 = stream.read(1)
            if tok2 != b"I":
                stream.seek(saved_pos, 0)
                continue
            tok3 = stream.read(1)
            if tok3 == b"":
                # An EI at the very end of the stream ends the image (pypdf #3468).
                stream.seek(saved_pos - 1, 0)
                stream_out.truncate(sav_pos_ei)
                break
            if tok3 not in WHITESPACES:
                stream.seek(saved_pos, 0)
                continue
            while tok3 in WHITESPACES:
                tok3 = stream.read(1)
            # An EI not preceded by white-space ends the image only before Q or EMC.
            if not _is_preceded_by_whitespace(
                stream_out, sav_pos_ei
            ) and not _is_followed_by_closing_operator(stream, tok3):
                stream.seek(saved_pos, 0)
                continue
            if is_followed_by_binary_data(stream):
                # An EI inside the image's data.
                stream.seek(saved_pos, 0)
                continue
            stream.seek(saved_pos - 1, 0)
            stream_out.truncate(sav_pos_ei)
            break

    return stream_out.getvalue()


def is_followed_by_binary_data(stream: IO[bytes], length: int = 10) -> bool:
    """Whether the next bytes look like image data rather than page content, by pypdf's
    heuristic, which follows iText's InlineImageParsingUtils.
    """
    position = stream.tell()
    data = stream.read(length)
    stream.seek(position)
    if not data:
        return False
    operator_start = None
    operator_end = None

    for index, byte in enumerate(data):
        if byte < 32 and byte not in WHITESPACES_AS_BYTES:
            return True
        is_whitespace = byte in WHITESPACES_AS_BYTES
        if operator_start is None and not is_whitespace:
            operator_start = index
        if operator_start is not None and is_whitespace:
            operator_end = index
            break

    if operator_start is None:
        return False
    if operator_end is None:
        operator_end = length
    operator_length = operator_end - operator_start
    operator = data[operator_start:operator_end]
    if operator.startswith(b"/") and operator_length > 1:
        return False
    if operator.replace(b".", b"").isdigit():
        return False
    return operator_length > 3


def _first_name(value: Any, default: str) -> Any:
    if isinstance(value, list):
        value = value[0] if value else default
    return value if isinstance(value, str) else default


def read_inline_image_data(stream: IO[bytes], settings: Any) -> bytes:
    """An inline image's data, read from just after the byte that follows its ID.

    The stream is left after the EI, as pypdf's ContentStream._read_inline_image leaves it.
    """
    filtr = _first_name(settings.get("/F", settings.get("/Filter", "not set")), "not set")
    savpos = stream.tell()
    if "AHx" in filtr or "ASCIIHexDecode" in filtr:
        data = extract_inline__ascii_hex_decode(stream)
    elif "A85" in filtr or "ASCII85Decode" in filtr:
        data = extract_inline__ascii85_decode(stream)
    elif "RL" in filtr or "RunLengthDecode" in filtr:
        data = extract_inline__run_length_decode(stream)
    elif "DCT" in filtr or "DCTDecode" in filtr:
        data = extract_inline__dct_decode(stream)
    elif filtr == "not set":
        cs = _first_name(settings.get("/CS", ""), "")
        if "RGB" in cs:
            lcs: float = 3
        elif "CMYK" in cs:
            lcs = 4
        else:
            bits = settings.get("/BPC", 8 if cs in {"/I", "/G", "/Indexed", "/DeviceGray"} else -1)
            if isinstance(bits, (int, float)) and bits > 0:
                lcs = bits / 8.0
            else:
                data = extract_inline_default(stream)
                lcs = -1
        width = settings.get("/W")
        height = settings.get("/H")
        if lcs > 0 and isinstance(width, int) and isinstance(height, int):
            data = stream.read(ceil(width * lcs) * height)
        elif lcs > 0:
            data = extract_inline_default(stream)
        read_non_whitespace(stream)
        stream.seek(-1, 1)
    else:
        data = extract_inline_default(stream)

    ei = stream.read(3)
    # An EI at the very end of the stream gives two bytes only (pypdf #3468).
    if len(ei) == 3:
        stream.seek(-1, 1)
    ei_trailing = ei[2:3]
    if ei[:2] != b"EI" or (ei_trailing != b"" and ei_trailing not in WHITESPACES):
        # A wrong or missing EI, as after wrong dimensions: look for the EI instead.
        stream.seek(savpos, 0)
        data = extract_inline_default(stream)
        ei = stream.read(3)
        if len(ei) == 3:
            stream.seek(-1, 1)
        ei_trailing = ei[2:3]
        if ei[:2] != b"EI" or (ei_trailing != b"" and ei_trailing not in WHITESPACES):
            raise PdfStreamError(
                f"Could not extract inline image, even using fallback. Expected 'EI', got {ei!r}"
            )
    return data
