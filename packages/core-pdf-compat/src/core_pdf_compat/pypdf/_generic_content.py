# ContentStream's attributes, messages and limits are ported from pypdf,
# Copyright (c) 2006-2008, Mathieu Fenniak, under its BSD-style license. Its parsing
# and writing are core_pdf_spec's, except inline images: where their data ends, and how
# they are written, follow pypdf.
"""pypdf's content stream object, and stream decoding through core-pdf's filters."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import IO, Any, cast

from core_pdf.impl.types import PdfString
from core_pdf_compat.pypdf._image_inline import read_inline_image_data
from core_pdf_compat.pypdf.errors import LimitReachedError, PdfReadError
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    DecodedStreamObject,
    DictionaryObject,
    NullObject,
    StreamObject,
    TextStringObject,
    _generic_key,
    create_string_object,
    to_core,
    to_generic,
)
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.inline_images import InlineImage, parse_inline_image_dictionary
from core_pdf_spec.s_07_content.operations import parse_content_token, serialize_content_operation
from core_pdf_spec.s_07_syntax.lexer import PdfLexer

logger = logging.getLogger(__name__)

CONTENT_STREAM_ARRAY_MAX_LENGTH = 10_000
MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH = 75_000_000
INLINE_IMAGE = b"INLINE IMAGE"


def decode_stream(stream: StreamObject) -> bytes:
    from core_pdf_compat.pypdf.filters import decode_stream_data

    try:
        return decode_stream_data(stream)
    except ValueError as error:
        raise PdfReadError(str(error)) from error


class _ContentText(TextStringObject):
    source: PdfString


class _ContentBytes(ByteStringObject):
    source: PdfString


def _content_string(value: PdfString) -> Any:
    # Text decoding loses a string's bytes and form, and those select glyphs here, so the
    # parsed string is kept and written back as it was read.
    text = create_string_object(value.data)
    string: Any = _ContentText(text) if isinstance(text, str) else _ContentBytes(text)
    string.source = value
    return string


def _generic_operand(value: Any) -> Any:
    if isinstance(value, PdfString):
        return _content_string(value)
    if isinstance(value, (list, tuple)):
        array = ArrayObject()
        array.extend(_generic_operand(item) for item in value)
        return array
    if isinstance(value, Mapping):
        dictionary = DictionaryObject()
        for key, item in value.items():
            dict.__setitem__(dictionary, _generic_key(key), _generic_operand(item))
        return dictionary
    return to_generic(value)


def _core_operand(value: Any) -> Any:
    if isinstance(value, (_ContentText, _ContentBytes)):
        return value.source
    if isinstance(value, list):
        return [_core_operand(item) for item in value]
    if isinstance(value, dict) and not isinstance(value, StreamObject):
        return {key[1:]: _core_operand(item) for key, item in dict.items(value)}
    return to_core(value)


def _operator_limit_error(available: int, limit: int) -> LimitReachedError:
    # pypdf reports how far it had read, in chunks of 16 bytes that double in size.
    read = chunk = 16
    while read < limit:
        chunk *= 2
        read += chunk
    return LimitReachedError(
        f"Read stream length of {min(read, available)} exceeds maximum allowed length of {limit}."
    )


def _at_inline_image(lexer: PdfLexer) -> bool:
    data, position = lexer.raw_data, lexer.pos
    if bytes(data[position : position + 2]) != b"BI":
        return False
    after = position + 2
    return after == lexer.data_len or bool(lexer.lexical_rules.separator_table[data[after]])


def _inline_image_parts(
    lexer: PdfLexer, stream: IO[bytes], offset: int
) -> tuple[Any, DictionaryObject, bytes]:
    try:
        settings = parse_inline_image_dictionary(lexer)
    except PdfParseError as error:
        raise PdfReadError("Unexpected end of stream.") from error
    generic = _generic_operand(settings)
    # pypdf reads ID and the byte after it.
    stream.seek(offset + lexer.pos + 1)
    data = read_inline_image_data(stream, generic)
    lexer.pos = stream.tell() - offset
    return settings, generic, data


def _inline_image(lexer: PdfLexer, stream: IO[bytes], offset: int = 0) -> dict[str, Any]:
    """Read an image from just after its BI. ``stream`` holds the lexer's data from ``offset``.

    core-pdf reads the dictionary; the data ends where pypdf's search ends it.
    """
    _, settings, data = _inline_image_parts(lexer, stream, offset)
    return {"settings": settings, "data": data}


def inline_image_spans(data: bytes) -> list[tuple[int, int, InlineImage]]:
    """Each inline image's start at its BI and end after its EI, with its dictionary as
    written and its data, found as ContentStream.operations finds them.

    An image pypdf cannot read raises its error. A malformed token before the next image
    ends the search, leaving the rest to core-pdf's tolerant parser.
    """
    stream = BytesIO(data)
    lexer = PdfLexer(data)
    spans: list[tuple[int, int, InlineImage]] = []
    try:
        while True:
            lexer.skip_ignored()
            if _at_inline_image(lexer):
                start = lexer.pos
                lexer.advance(2)
                settings, _, image_data = _inline_image_parts(lexer, stream, 0)
                spans.append((start, lexer.pos, InlineImage(settings, image_data)))
                continue
            try:
                if parse_content_token(lexer) is None:
                    return spans
            except PdfParseError:
                return spans
    finally:
        lexer.close()


class ContentStream(DecodedStreamObject):
    """Holds raw data or parsed operations, and rebuilds whichever one is asked for.

    Reading .operations parses ._data and clears it, setting them clears ._data, and
    set_data clears ._operations. get_data writes the operations back.
    """

    _OPERATOR_LENGTH_LIMIT = 128

    def __init__(self, stream: Any, pdf: Any, forced_encoding: Any = None) -> None:
        super().__init__()
        self.pdf = pdf
        self.forced_encoding = forced_encoding
        self._operations: list[tuple[Any, bytes]] = []
        if stream is None:
            return
        stream = stream.get_object()
        self._data = self._joined(stream) if isinstance(stream, ArrayObject) else stream.get_data()

    @staticmethod
    def _joined(streams: ArrayObject) -> bytes:
        if (count := len(streams)) > CONTENT_STREAM_ARRAY_MAX_LENGTH:
            raise LimitReachedError(
                f"Array-based stream has {count} > {CONTENT_STREAM_ARRAY_MAX_LENGTH} elements."
            )
        data = bytearray()
        for item in streams:
            resolved = item.get_object()
            if isinstance(resolved, NullObject):
                continue
            if isinstance(resolved, StreamObject):
                data += resolved.get_data()
                if len(data) > MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH:
                    raise LimitReachedError(
                        f"Array-based stream has at least {len(data)} > "
                        f"{MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH} output bytes."
                    )
            else:
                logger.warning(
                    "Expected StreamObject, got %s instead. Data might be wrong.",
                    type(resolved).__name__,
                )
            if data[-1:] != b"\n":
                data += b"\n"
        return bytes(data)

    def _parse_content_stream(self, stream: Any) -> None:
        stream.seek(0, 0)
        lexer = PdfLexer(stream.read())
        operands: list[Any] = []
        try:
            while True:
                lexer.skip_ignored()
                if _at_inline_image(lexer):
                    lexer.advance(2)
                    self._operations.append((_inline_image(lexer, stream), INLINE_IMAGE))
                    continue
                token = parse_content_token(lexer)
                if token is None:
                    return
                if not token.is_operator:
                    operands.append(_generic_operand(token.value))
                    continue
                operator = str(token.value).encode("latin-1")
                if len(operator) >= self._OPERATOR_LENGTH_LIMIT:
                    available = lexer.data_len - token.start
                    raise _operator_limit_error(available, self._OPERATOR_LENGTH_LIMIT)
                self._operations.append((operands, operator))
                operands = []
        except PdfParseError as error:
            raise PdfReadError(str(error)) from error
        finally:
            lexer.close()

    def _read_inline_image(self, stream: Any) -> dict[str, Any]:
        """Read an image from just after its BI, as pypdf does."""
        start = stream.tell()
        lexer = PdfLexer(stream.read())
        try:
            image = _inline_image(lexer, stream, start)
        finally:
            lexer.close()
        return image

    def replicate(self, pdf_dest: Any) -> ContentStream:
        copy = self._reference_clone(self.__class__(None, None), pdf_dest, False)
        self._replicate_decoded(pdf_dest)
        for key, value in dict.items(self):
            copy[key.replicate(pdf_dest)] = (
                value.replicate(pdf_dest) if hasattr(value, "replicate") else value
            )
        copy._clone(self, pdf_dest, False, (), set())
        return copy

    def clone(
        self,
        pdf_dest: Any,
        force_duplicate: bool = False,
        ignore_fields: Sequence[str | int] | None = (),
    ) -> ContentStream:
        reference = self.indirect_reference
        if not force_duplicate and reference is not None and reference.pdf == pdf_dest:
            return self
        copy = self._reference_clone(self.__class__(None, None), pdf_dest, force_duplicate)
        copy._clone(self, pdf_dest, force_duplicate, ignore_fields or [], set())
        return copy

    def _clone(
        self,
        src: DictionaryObject,
        pdf_dest: Any,
        force_duplicate: bool,
        ignore_fields: Any,
        visited: set[tuple[int, int]],
    ) -> None:
        del force_duplicate, ignore_fields, visited
        # As in pypdf, a content stream's dictionary entries are not cloned.
        source = cast("ContentStream", src)
        self._data = source._data
        self.pdf = pdf_dest
        self._operations = list(source._operations)
        self.forced_encoding = source.forced_encoding

    def get_data(self) -> bytes:
        if not self._data:
            self._data = b"".join(map(_serialize_operation, self._operations))
        return self._data

    def set_data(self, data: bytes) -> None:
        self._data = data
        self._operations = []

    def _stored_data(self) -> bytes:
        return self.get_data()

    @property
    def operations(self) -> list[tuple[Any, bytes]]:
        if not self._operations and self._data:
            self._parse_content_stream(BytesIO(self._data))
            self._data = b""
        return self._operations

    @operations.setter
    def operations(self, operations: list[tuple[Any, bytes]]) -> None:
        self._operations = operations
        self._data = b""

    def isolate_graphics_state(self) -> None:
        if self._operations:
            self._operations.insert(0, ([], b"q"))
            self._operations.append(([], b"Q"))
        elif self._data:
            self._data = b"q\n" + self._data + b"\nQ\n"


def _serialize_operation(operation: tuple[Any, bytes]) -> bytes:
    operands, operator = operation
    if operator == INLINE_IMAGE:
        # Written as pypdf writes it, so the data is read back as it was.
        settings = BytesIO()
        operands["settings"].write_to_stream(settings)
        return b"BI" + settings.getvalue()[2:-2] + b"ID " + operands["data"] + b"EI\n"
    return serialize_content_operation(operator, [_core_operand(item) for item in operands])
