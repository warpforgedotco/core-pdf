# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfReference
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.object_cache import ObjectCache
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int,
    parse_int_strict,
)
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key


class PdfObjectStream:
    __slots__ = ("stream", "objects", "raw_body", "index", "lexer", "lock")

    def __init__(self, stream: PdfStream, kw_cache: dict[bytes, object] | None = None) -> None:
        type_name = normalize_pdf_name(lookup_dict_key(stream.dictionary, "Type"))
        if type_name is not None and type_name != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = parse_int_strict(lookup_dict_key(stream.dictionary, "N"))
        first = parse_int_strict(lookup_dict_key(stream.dictionary, "First"))
        if n < 0 or first < 0:
            raise PdfParseError("invalid object stream dictionary")
        if first > len(stream.data):
            recovered_first = recover_object_stream_first(stream.data, n, kw_cache)
            if recovered_first is None:
                raise PdfParseError("invalid object stream dictionary")
            first = recovered_first
        pairs = parse_object_stream_header(stream.data, first, n, kw_cache)
        if len(pairs) < n:
            recovered_first = recover_object_stream_first(stream.data, n, kw_cache)
            if recovered_first is not None and recovered_first != first:
                recovered_pairs = parse_object_stream_header(
                    stream.data, recovered_first, n, kw_cache
                )
                if len(recovered_pairs) > len(pairs):
                    first = recovered_first
                    pairs = recovered_pairs
        if not pairs:
            raise PdfParseError("object stream header is truncated")
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
        self.lexer = PdfLexer(body, kw_cache=kw_cache)
        self.lock = threading.RLock()

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
                result = self.recover_object_at(rel_offset)
            self.objects[obj_num] = result
            return result

    def recover_object_at(self, rel_offset: int) -> Any:
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


def parse_object_stream_header(
    data: bytes | memoryview,
    first: int,
    n: int,
    kw_cache: dict[bytes, object] | None = None,
) -> list[tuple[int, int]]:
    header = data[:first]
    header_lexer = PdfLexer(header, kw_cache=kw_cache)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n:
        token = header_lexer.scan_word(skip_ignored=True)
        if token is None:
            break
        obj_num = parse_int(token[0], None)
        header_lexer.pos = token[1]
        token = header_lexer.scan_word(skip_ignored=True)
        if token is None:
            break
        offset = parse_int(token[0], None)
        header_lexer.pos = token[1]
        if obj_num is None or offset is None:
            break
        pairs.append((obj_num, offset))
    return pairs


def recover_object_stream_first(
    data: bytes | memoryview, n: int, kw_cache: dict[bytes, object] | None = None
) -> int | None:
    lexer = PdfLexer(data, kw_cache=kw_cache)
    pairs = 0
    last_end = 0
    while pairs < n:
        token = lexer.scan_word(skip_ignored=True)
        if token is None:
            break
        if parse_int(token[0], None) is None:
            break
        lexer.pos = token[1]
        token = lexer.scan_word(skip_ignored=True)
        if token is None:
            break
        if parse_int(token[0], None) is None:
            break
        lexer.pos = token[1]
        last_end = token[1]
        pairs += 1
    if pairs == 0:
        return None
    return last_end
