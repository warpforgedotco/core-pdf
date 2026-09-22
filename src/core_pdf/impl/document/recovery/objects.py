# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing

from core_pdf.impl.document.recovery.lexer import PdfLexer
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream as SyntaxObjectStream
from core_pdf_spec.s_07_syntax.objects import (
    parse_object_stream_pair,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_int_strict,
)

if typing.TYPE_CHECKING:
    from typing import Any


class PdfObjectStream(SyntaxObjectStream):
    __slots__ = ()

    def build_index(self, pairs: list[tuple[int, int]], body_length: int) -> dict[int, int]:
        index_map: dict[int, int] = {}
        for obj_num, offset in pairs:
            if obj_num < 0 or offset < 0 or offset >= body_length:
                continue
            if obj_num in index_map:
                continue
            index_map[obj_num] = offset
        if not index_map:
            raise PdfParseError("invalid object stream header")
        return index_map

    def parse_object_at(self, offset: int, end: int) -> Any:
        try:
            return super().parse_object_at(offset, end)
        except PdfParseError:
            return self.handle_object_error(offset)

    def read_header(
        self, stream: PdfStream, decoded_data: bytes
    ) -> tuple[int, list[tuple[int, int]]]:
        type_name = recover_pdf_name(stream.dictionary.get("Type"))
        if type_name is not None and type_name != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = parse_int_strict(stream.dictionary.get("N"))
        first = parse_int_strict(stream.dictionary.get("First"))
        if n < 0 or first < 0:
            raise PdfParseError("invalid object stream dictionary")
        if first > len(decoded_data):
            recovered_first = recover_object_stream_first(decoded_data, n)
            if recovered_first is None:
                raise PdfParseError("invalid object stream dictionary")
            first = recovered_first
        pairs = parse_object_stream_header(decoded_data, first, n)
        if len(pairs) < n:
            recovered_first = recover_object_stream_first(decoded_data, n)
            if recovered_first is not None and recovered_first != first:
                recovered_pairs = parse_object_stream_header(decoded_data, recovered_first, n)
                if len(recovered_pairs) > len(pairs):
                    first = recovered_first
                    pairs = recovered_pairs
        if not pairs:
            raise PdfParseError("object stream header is truncated")
        return first, pairs

    def create_lexer(self, body: bytes | memoryview) -> PdfLexer:
        return PdfLexer(body, semantic_context=self.semantic_context)

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
        lexer = self.create_lexer(body)
        try:
            for pos in [rel_offset, *starts]:
                try:
                    return lexer.parse_object_at(pos)
                except PdfParseError:
                    continue
            raise PdfParseError("invalid object stream object")
        finally:
            lexer.close()


def recover_object_stream_first(data: bytes | memoryview, n: int) -> int | None:
    pairs, last_end = scan_object_stream_pairs(data, n)
    return last_end if pairs else None


def parse_object_stream_header(
    data: bytes | memoryview,
    first: int,
    n: int,
) -> list[tuple[int, int]]:
    return scan_object_stream_pairs(data[:first], n)[0]


def scan_object_stream_pairs(data: bytes | memoryview, n: int) -> tuple[list[tuple[int, int]], int]:
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
