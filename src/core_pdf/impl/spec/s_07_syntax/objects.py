# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import ObjectCache
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int,
)
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import skip_pdf_ignored
from core_pdf.impl.types import PdfReference


class PdfObjectStream:
    __slots__ = ("stream", "objects", "raw_body", "index", "lexer", "lock")

    def __init__(self, stream: PdfStream) -> None:
        first, pairs = self.internal_read_header(stream)
        self.internal_validate_header_pairs(pairs)
        index_map: dict[int, int] = {}
        body = stream.data[first:]
        body_len = len(body)
        for obj_num, offset in pairs:
            if obj_num < 0 or offset < 0 or offset >= body_len:
                self.internal_invalid_header_entry()
                continue
            if obj_num in index_map:
                self.internal_invalid_header_entry()
                continue
            index_map[obj_num] = offset
        if not index_map:
            raise PdfParseError("invalid object stream header")
        self.stream = stream
        self.objects: ObjectCache = {}
        self.raw_body = body
        self.index = index_map
        self.lexer = self.internal_create_lexer(body)
        self.lock = threading.RLock()

    def internal_read_header(self, stream: PdfStream) -> tuple[int, list[tuple[int, int]]]:
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

    def internal_invalid_header_entry(self) -> None:
        raise PdfParseError("invalid object stream header")

    def internal_validate_header_pairs(self, pairs: list[tuple[int, int]]) -> None:
        offsets: set[int] = set()
        for object_number, offset in pairs:
            if object_number <= 0 or offset in offsets:
                raise PdfParseError("invalid object stream header")
            offsets.add(offset)

    def internal_create_lexer(self, body: bytes) -> PdfLexer:
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
            try:
                result = self.lexer.parse_object_at(rel_offset)
            except PdfParseError:
                result = self.handle_object_error(rel_offset)
            self.objects[obj_num] = result
            return result

    def handle_object_error(self, rel_offset: int) -> Any:
        raise PdfParseError("invalid object stream object")


def internal_scan_object_stream_pairs(
    data: bytes | memoryview,
    n: int,
) -> tuple[list[tuple[int, int]], int]:
    """Scan up to ``n`` ``objnum offset`` pairs, with the end of the last one."""
    lexer = PdfLexer(data)
    pairs: list[tuple[int, int]] = []
    last_end = 0
    while len(pairs) < n:
        first_token = lexer.scan_word(skip_ignored=True)
        if first_token is None:
            break
        obj_num = parse_int(first_token[0], None)
        if obj_num is None:
            break
        lexer.pos = first_token[1]
        second_token = lexer.scan_word(skip_ignored=True)
        if second_token is None:
            break
        offset = parse_int(second_token[0], None)
        if offset is None:
            break
        lexer.pos = second_token[1]
        last_end = second_token[1]
        pairs.append((obj_num, offset))
    return pairs, last_end


def parse_object_stream_header(
    data: bytes | memoryview,
    first: int,
    n: int,
) -> list[tuple[int, int]]:
    header = data[:first]
    pairs, last_end = internal_scan_object_stream_pairs(header, n)
    if skip_pdf_ignored(header, last_end, len(header)) != len(header):
        raise PdfParseError("unexpected data after object stream header")
    return pairs
