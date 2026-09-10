# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import ObjectCache
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import skip_pdf_ignored
from core_pdf_spec.types import PdfReference


class PdfObjectStream:
    __slots__ = ("stream", "objects", "raw_body", "index", "lexer", "lock")

    def __init__(self, stream: PdfStream) -> None:
        first, pairs = self.read_header(stream)
        self.validate_header_pairs(pairs)
        index_map: dict[int, int] = {}
        body = stream.data[first:]
        body_len = len(body)
        for obj_num, offset in pairs:
            if obj_num < 0 or offset < 0 or offset >= body_len:
                raise PdfParseError("invalid object stream header")
            if obj_num in index_map:
                raise PdfParseError("invalid object stream header")
            index_map[obj_num] = offset
        if not index_map:
            raise PdfParseError("invalid object stream header")
        self.stream = stream
        self.objects: ObjectCache = {}
        self.raw_body = body
        self.index = index_map
        self.lexer = self.create_lexer(body)
        self.lock = threading.RLock()

    def read_header(self, stream: PdfStream) -> tuple[int, list[tuple[int, int]]]:
        """Validate the dictionary and return First plus all header pairs."""
        if normalize_pdf_name(stream.dictionary.get("Type")) != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = stream.dictionary.get("N")
        first = stream.dictionary.get("First")
        if (
            type(n) is not int
            or type(first) is not int
            or n < 0
            or first < 0
            or first > len(stream.data)
        ):
            raise PdfParseError("invalid object stream dictionary")
        pairs = parse_object_stream_header(stream.data, first, n)
        if len(pairs) != n:
            raise PdfParseError("object stream header is truncated")
        return first, pairs

    def validate_header_pairs(self, pairs: list[tuple[int, int]]) -> None:
        """Reject reserved object numbers and overlapping object offsets."""
        offsets: set[int] = set()
        for object_number, offset in pairs:
            if object_number <= 0 or offset in offsets:
                raise PdfParseError("invalid object stream header")
            offsets.add(offset)

    def create_lexer(self, body: bytes) -> PdfLexer:
        """Create the parser for decrypted, decoded object-stream body bytes."""
        return PdfLexer(body)

    def get(self, reference: int | PdfReference, default: Any = None) -> Any:
        obj_num = reference.object_number if isinstance(reference, PdfReference) else reference
        if obj_num < 0:
            raise ValueError("invalid object number")
        with self.lock:
            if obj_num in self.objects:
                return self.objects[obj_num]
            if obj_num not in self.index:
                return default
            rel_offset = self.index[obj_num]
            result = self.parse_object_at(rel_offset)
            self.objects[obj_num] = result
            return result

    def parse_object_at(self, offset: int) -> Any:
        """Parse one object at a decoded-body-relative offset; errors propagate.

        ``get`` calls this extension method while holding the object-stream lock
        and caches its successful result.
        """
        return self.lexer.parse_object_at(offset)


def parse_object_stream_pair(lexer: PdfLexer) -> tuple[int, int]:
    """Consume one object-number/offset pair, raising if either integer is absent."""
    values: list[int] = []
    for _ in range(2):
        token = lexer.scan_word()
        value = None if token is None else parse_int(token[0], None)
        if value is None or token is None:
            raise PdfParseError("object stream header is truncated")
        lexer.pos = token[1]
        values.append(value)
    return values[0], values[1]


def parse_object_stream_header(
    data: bytes | memoryview, first: int, n: int
) -> list[tuple[int, int]]:
    if n < 0 or not 0 <= first <= len(data):
        raise PdfParseError("invalid object stream dictionary")
    header = data[:first]
    lexer = PdfLexer(header)
    try:
        pairs = [parse_object_stream_pair(lexer) for _ in range(n)]
        if skip_pdf_ignored(header, lexer.pos, len(header)) != len(header):
            raise PdfParseError("unexpected data after object stream header")
        return pairs
    finally:
        lexer.close()


__all__ = (
    "PdfObjectStream",
    "parse_object_stream_header",
    "parse_object_stream_pair",
)
