# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream as SyntaxObjectStream
from core_pdf_spec.s_07_syntax.objects import (
    parse_object_stream_pair,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import ObjectCache
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int_strict,
)


class PdfObjectStream(SyntaxObjectStream):
    __slots__ = ()

    def __init__(self, stream: PdfStream) -> None:
        first, pairs = self.read_header(stream)
        self.validate_header_pairs(pairs)
        index_map: dict[int, int] = {}
        body = stream.data[first:]
        body_len = len(body)
        for obj_num, offset in pairs:
            if obj_num < 0 or offset < 0 or offset >= body_len:
                continue
            if obj_num in index_map:
                continue
            index_map[obj_num] = offset
        if not index_map:
            raise PdfParseError("invalid object stream header")
        self.stream = stream
        self.objects: ObjectCache = {}
        self.raw_body = body
        self.index = index_map
        self.lexer = self.create_lexer(body)
        self.lock = threading.RLock()

    def parse_object_at(self, offset: int) -> Any:
        try:
            return super().parse_object_at(offset)
        except PdfParseError:
            return self.handle_object_error(offset)

    def read_header(self, stream: PdfStream) -> tuple[int, list[tuple[int, int]]]:
        type_name = normalize_pdf_name(stream.dictionary.get("Type"))
        if type_name is not None and type_name != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = parse_int_strict(stream.dictionary.get("N"))
        first = parse_int_strict(stream.dictionary.get("First"))
        if n < 0 or first < 0:
            raise PdfParseError("invalid object stream dictionary")
        if first > len(stream.data):
            recovered_first = recover_object_stream_first(stream.data, n)
            if recovered_first is None:
                raise PdfParseError("invalid object stream dictionary")
            first = recovered_first
        pairs = parse_object_stream_header(stream.data, first, n)
        if len(pairs) < n:
            recovered_first = recover_object_stream_first(stream.data, n)
            if recovered_first is not None and recovered_first != first:
                recovered_pairs = parse_object_stream_header(stream.data, recovered_first, n)
                if len(recovered_pairs) > len(pairs):
                    first = recovered_first
                    pairs = recovered_pairs
        if not pairs:
            raise PdfParseError("object stream header is truncated")
        return first, pairs

    def validate_header_pairs(self, pairs: list[tuple[int, int]]) -> None:
        pass

    def create_lexer(self, body: bytes) -> PdfLexer:
        return PdfLexer(body)

    def handle_object_error(self, rel_offset: int) -> Any:
        body = self.raw_body
        n = len(body)
        starts: list[int] = []
        search_start = max(0, rel_offset - 64)
        search_end = min(n, rel_offset + 64)
        start_bytes = b"(<[/+-0123456789tfn"
        for pos in range(search_start, search_end):
            if pos == rel_offset:
                continue
            if body[pos] not in start_bytes:
                continue
            starts.append(pos)
        starts.sort(key=lambda pos: (abs(pos - rel_offset), pos))
        for pos in starts:
            try:
                return self.lexer.parse_object_at(pos)
            except PdfParseError:
                continue
        raise PdfParseError("invalid object stream object")


def recover_object_stream_first(data: bytes | memoryview, n: int) -> int | None:
    pairs, last_end = internal_scan_object_stream_pairs(data, n)
    return last_end if pairs else None


def parse_object_stream_header(
    data: bytes | memoryview,
    first: int,
    n: int,
) -> list[tuple[int, int]]:
    return internal_scan_object_stream_pairs(data[:first], n)[0]


def internal_scan_object_stream_pairs(
    data: bytes | memoryview, n: int
) -> tuple[list[tuple[int, int]], int]:
    lexer = PdfLexer(data)
    pairs: list[tuple[int, int]] = []
    last_end = 0
    try:
        while len(pairs) < n:
            try:
                pair = parse_object_stream_pair(lexer)
            except PdfParseError:
                break
            pairs.append(pair)
            last_end = lexer.pos
        return pairs, last_end
    finally:
        lexer.close()
