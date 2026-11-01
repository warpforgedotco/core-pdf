# Ported from pypdf, Copyright (c) 2006, Mathieu Fenniak, under its BSD-style license.
"""pypdf's object reading: one object at a time from a stream, with pypdf's strictness.

A file goes through core-pdf's parser instead. These back the public read_from_stream
methods and read_object, whose positions, limits and error messages callers rely on.
``generic`` imports this module near its end, so these names see a complete ``generic``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

from core_pdf_compat.pypdf.errors import (
    STREAM_TRUNCATED_PREMATURELY,
    LimitReachedError,
    PdfReadError,
    PdfStreamError,
)
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    StreamObject,
)

logger = logging.getLogger(__name__)

WHITESPACES = (b"\x00", b"\t", b"\n", b"\f", b"\r", b" ")
MAX_DECLARED_STREAM_LENGTH = 75_000_000
INDIRECT_PATTERN = re.compile(rb"[+-]?(\d+)\s+(\d+)\s+R[^a-zA-Z]")
ENDSTREAM = re.compile(b"endstream")


def read_until_regex(stream: Any, regex: re.Pattern[bytes], length: int = sys.maxsize) -> bytes:
    """Read up to a match, or to the end of the stream. The limit is checked per chunk."""
    data = bytearray()
    chunk_size = 16
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return bytes(data)
        if (total := len(data) + len(chunk)) >= length:
            raise LimitReachedError(
                f"Read stream length of {total} exceeds maximum allowed length of {length}."
            )
        # Reread the tail, so a match can span two chunks.
        searched_from = max(0, len(data) - 16)
        data += chunk
        match = regex.search(data, searched_from)
        if match is not None:
            stream.seek(match.start() - total, 1)
            return bytes(data[: match.start()])
        if chunk_size < 8192:
            chunk_size <<= 1


def read_non_whitespace(stream: Any) -> bytes:
    token = stream.read(1)
    while token in WHITESPACES:
        token = stream.read(1)
    return token


def skip_over_comment(stream: Any) -> None:
    token = stream.read(1)
    stream.seek(-1, 1)
    if token == b"%":
        while token not in (b"\n", b"\r"):
            token = stream.read(1)
            if token == b"":
                raise PdfStreamError("File ended unexpectedly.")


def _read_until_whitespace(stream: Any) -> None:
    token = stream.read(1)
    while token and not token.isspace():
        token = stream.read(1)


def read_null(stream: Any) -> NullObject:
    if stream.read(4) != b"null":
        raise PdfReadError("Could not read Null object")
    return NullObject()


def read_boolean(stream: Any) -> BooleanObject:
    word = stream.read(4)
    if word == b"true":
        return BooleanObject(True)
    if word == b"fals":
        stream.read(1)
        return BooleanObject(False)
    raise PdfReadError("Could not read Boolean object")


def _read_field(stream: Any, *, skip_leading: bool = False) -> bytes:
    field = b""
    while True:
        token = stream.read(1)
        if not token:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if token.isspace():
            if field or not skip_leading:
                return field
            continue
        field += token


def read_indirect(stream: Any, pdf: Any) -> IndirectObject:
    number = _read_field(stream)
    generation = _read_field(stream, skip_leading=True)
    if read_non_whitespace(stream) != b"R":
        raise PdfReadError(f"Error reading indirect object reference at byte {hex(stream.tell())}")
    return IndirectObject(int(number), int(generation), pdf)


def read_number(stream: Any) -> NumberObject | FloatObject:
    text = read_until_regex(stream, NumberObject.NumberPattern, NumberObject._LENGTH_LIMIT)
    return FloatObject(text) if b"." in text else NumberObject(text)


def decode_name(data: bytes) -> str | None:
    for charset in NameObject.CHARSETS:
        try:
            return data.decode(charset)
        except UnicodeError, LookupError:
            continue
    return None


def read_name(stream: Any, pdf: Any) -> NameObject:
    name = stream.read(1)
    if name != NameObject.prefix:
        raise PdfReadError("Name read error")
    name += read_until_regex(stream, NameObject.delimiter_pattern, NameObject._LENGTH_LIMIT)
    name = NameObject.unnumber(name)
    text = decode_name(name)
    if text is not None:
        return NameObject(text)
    if getattr(pdf, "strict", False):
        raise PdfReadError(
            f"Illegal character in NameObject ({name!r}). "
            "You may need to adjust NameObject.CHARSETS."
        )
    logger.warning(
        "Illegal character in NameObject (%r), you may need to adjust NameObject.CHARSETS", name
    )
    return NameObject(name.decode("latin-1"))


def read_array(stream: Any, pdf: Any, forced_encoding: Any = None) -> ArrayObject:
    array = ArrayObject()
    if stream.read(1) != b"[":
        raise PdfReadError("Could not read array")
    while True:
        token = stream.read(1)
        while token.isspace():
            token = stream.read(1)
        if token in (b"", b"]"):
            return array
        stream.seek(-1, 1)
        if token == b"%":
            skip_over_comment(stream)
        else:
            array.append(read_object(stream, pdf, forced_encoding))


def _is_strict(pdf: Any) -> bool:
    return pdf is not None and pdf.strict


def _read_key(stream: Any, pdf: Any) -> NameObject | NullObject | None:
    """The next key, null at the end of a broken dictionary, or None after a skipped token."""
    try:
        key = read_object(stream, pdf)
        if isinstance(key, NullObject):
            return key
        if not isinstance(key, NameObject):
            raise PdfReadError(f"Expecting a NameObject for key but found {key!r}")
    except PdfReadError as error:
        if _is_strict(pdf):
            raise
        logger.warning("%r", error)
        return None
    return key


def _read_entries(stream: Any, pdf: Any, forced_encoding: Any) -> dict[Any, Any]:
    """The entries, or a DictionaryObject of those read before a tolerated error."""
    data: dict[Any, Any] = {}
    while True:
        token = read_non_whitespace(stream)
        if token == b"\x00":
            continue
        if token == b"%":
            stream.seek(-1, 1)
            skip_over_comment(stream)
            continue
        if not token:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if token == b">":
            stream.read(1)
            return data
        stream.seek(-1, 1)
        try:
            key = _read_key(stream, pdf)
            if key is None:
                continue
            if isinstance(key, NullObject):
                return data
            read_non_whitespace(stream)
            stream.seek(-1, 1)
            value = read_object(stream, pdf, forced_encoding)
        except (RecursionError, LimitReachedError) as error:
            raise PdfReadError(repr(error)) from error
        except Exception as error:
            if _is_strict(pdf):
                raise PdfReadError(repr(error)) from error
            logger.warning("%r", error)
            partial = DictionaryObject()
            partial.update(data)
            return partial
        if not data.get(key):
            data[key] = value
            continue
        message = f"Multiple definitions in dictionary at byte {hex(stream.tell())} for key {key}"
        if _is_strict(pdf):
            raise PdfReadError(message)
        logger.warning("%s", message)


def _declared_length(stream: Any, pdf: Any, data: dict[Any, Any]) -> int:
    if "/Length" not in data:
        if _is_strict(pdf):
            raise PdfStreamError("Stream length not defined")
        logger.warning("Stream length not defined @pos=%d", stream.tell())
        data[NameObject("/Length")] = NumberObject(-1)
    length = data["/Length"]
    if isinstance(length, IndirectObject):
        position = stream.tell()
        length = pdf.get_object(length)
        stream.seek(position, 0)
    return -1 if length is None else length


def _read_stream_data(stream: Any, pdf: Any, data: dict[Any, Any]) -> None:
    end_of_line = stream.read(1)
    # Some files have spaces between the keyword and its line break.
    while end_of_line == b" ":
        end_of_line = stream.read(1)
    if end_of_line not in (b"\n", b"\r"):
        raise PdfStreamError("Stream data must be followed by a newline")
    if end_of_line == b"\r" and stream.read(1) != b"\n":
        stream.seek(-1, 1)
    length = _declared_length(stream, pdf, data)
    start = stream.tell()
    if length > MAX_DECLARED_STREAM_LENGTH:
        raise LimitReachedError(
            f"Declared stream length of {length} exceeds maximum allowed length."
        )
    if length >= 0:
        data["__streamdata__"] = stream.read(length)
    else:
        data["__streamdata__"] = read_until_regex(stream, ENDSTREAM, MAX_DECLARED_STREAM_LENGTH)
    keyword = read_non_whitespace(stream) + stream.read(8)
    if keyword == b"endstream":
        return
    # A /Length one too long is a common producer bug, so look back one byte.
    position = stream.tell()
    stream.seek(-10, 1)
    found = stream.read(9)
    if found == b"endstream":
        data["__streamdata__"] = data["__streamdata__"][:-1]
    elif pdf is not None and not pdf.strict:
        stream.seek(start, 0)
        data["__streamdata__"] = DictionaryObject._read_unsized_from_stream(
            stream=stream, pdf=pdf, length=MAX_DECLARED_STREAM_LENGTH
        )
    else:
        stream.seek(position, 0)
        raise PdfReadError(
            f"Unable to find 'endstream' marker after stream at byte {hex(stream.tell())} "
            f"(nd='{keyword[1:]!r}', end='{found!r}')."
        )


def read_dictionary(stream: Any, pdf: Any, forced_encoding: Any = None) -> DictionaryObject:
    if stream.read(2) != b"<<":
        raise PdfReadError(
            f"Dictionary read error at byte {hex(stream.tell())}: stream must begin with '<<'"
        )
    data = _read_entries(stream, pdf, forced_encoding)
    if isinstance(data, DictionaryObject):
        return data
    position = stream.tell()
    if read_non_whitespace(stream) == b"s" and stream.read(5) == b"tream":
        _read_stream_data(stream, pdf, data)
        return StreamObject.initialize_from_dictionary(data)
    stream.seek(position, 0)
    output = DictionaryObject()
    output.update(data)
    return output


def read_unsized(stream: Any, pdf: Any, length: int) -> bytes:
    """Read a stream of unknown length, up to the next object in the cross-reference table."""
    position = stream.tell()
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    stream.seek(position)
    following = DictionaryObject._get_next_object_position(position, end, list(pdf.xref), pdf)
    count = following - position
    if count >= length:
        raise LimitReachedError(f"Requested length of {count} exceeds maximum allowed length.")
    data = stream.read(count)
    marker = data.find(b"endstream")
    if marker < 0:
        raise PdfReadError(f"Unable to find 'endstream' marker for obj starting at {position}.")
    stream.seek(position + marker + len(b"endstream"))
    return data[: marker - 1]


def read_object(stream: Any, pdf: Any = None, forced_encoding: Any = None) -> Any:
    from core_pdf_compat.pypdf._generic_trees import (  # noqa: PLC0415
        read_hex_string_from_stream,
        read_string_from_stream,
    )

    token = stream.read(1)
    stream.seek(-1, 1)
    if token == b"/":
        return read_name(stream, pdf)
    if token == b"<":
        is_dictionary = stream.read(2) == b"<<"
        stream.seek(-2, 1)
        if is_dictionary:
            return read_dictionary(stream, pdf, forced_encoding)
        return read_hex_string_from_stream(stream, forced_encoding)
    if token == b"[":
        return read_array(stream, pdf, forced_encoding)
    if token in (b"t", b"f"):
        return read_boolean(stream)
    if token == b"(":
        return read_string_from_stream(stream, forced_encoding)
    if token == b"e" and stream.read(6) == b"endobj":
        return NullObject()
    if token == b"n":
        return read_null(stream)
    if token == b"%":
        skip_over_comment(stream)
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        return read_object(stream, pdf, forced_encoding)
    if token in b"0123456789+-.":
        peek = stream.read(20)
        stream.seek(-len(peek), 1)
        if INDIRECT_PATTERN.match(peek) is not None:
            return read_indirect(stream, pdf)
        return read_number(stream)
    position = stream.tell()
    stream.seek(max(0, position - 20))
    extract = stream.read(80)
    stream.seek(position)
    _read_until_whitespace(stream)
    raise PdfReadError(
        f"Invalid Elementary Object starting with {token!r} @{position}: {extract!r}"
    )
