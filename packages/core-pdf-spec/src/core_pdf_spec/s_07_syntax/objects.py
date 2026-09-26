# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import ObjectCache, PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
)
from core_pdf_spec.s_07_syntax_primitives.numbers import is_integer_token
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfReference


class ObjectStreamLexer(PdfLexer):
    __slots__ = ()

    def parse_stream(self, dictionary: PdfDict) -> PdfStream:
        raise PdfParseError("streams cannot be stored in an object stream")


class PdfObjectStream:
    __slots__ = (
        "objects",
        "raw_body",
        "index",
        "lock",
        "semantic_context",
        "object_numbers",
        "ends",
    )

    def __init__(
        self, stream: PdfStream, *, semantic_context: SemanticContext | None = None
    ) -> None:
        self.semantic_context = semantic_context
        decoded_data = stream.data
        first, pairs = self.read_header(stream, decoded_data)
        body = decoded_data[first:]
        index_map = self.build_index(pairs, len(body))
        self.objects: ObjectCache = {}
        self.raw_body = body
        self.index = index_map
        self.object_numbers = tuple(index_map)
        offsets = sorted(set(index_map.values()))
        self.ends = dict(zip(offsets, [*offsets[1:], len(body)], strict=True))
        self.lock = threading.RLock()

    def read_header(
        self, stream: PdfStream, decoded_data: bytes
    ) -> tuple[int, list[tuple[int, int]]]:
        if decoded_name(stream.dictionary.get("Type")) != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = stream.dictionary.get("N")
        first = stream.dictionary.get("First")
        if (
            type(n) is not int
            or type(first) is not int
            or n < 0
            or first < 0
            or first > len(decoded_data)
        ):
            raise PdfParseError("invalid object stream dictionary")
        pairs = parse_object_stream_header(decoded_data, first, n, context=self.semantic_context)
        return first, pairs

    def build_index(self, pairs: list[tuple[int, int]], body_length: int) -> dict[int, int]:
        index: dict[int, int] = {}
        previous_offset = -1
        for object_number, offset in pairs:
            if (
                object_number <= 0
                or object_number in index
                or not previous_offset < offset < body_length
                or (not index and offset != 0)
            ):
                raise PdfParseError("invalid object stream header")
            index[object_number] = offset
            previous_offset = offset
        if not index:
            raise PdfParseError("invalid object stream header")
        return index

    def create_lexer(self, body: bytes | memoryview) -> PdfLexer:
        return ObjectStreamLexer(body, semantic_context=self.semantic_context)

    def close(self) -> None:
        with self.lock:
            self.objects.clear()
            self.index.clear()
            self.object_numbers = ()
            self.ends.clear()
            self.raw_body = b""

    def get(self, reference: int | PdfReference, default: Any = None) -> Any:
        obj_num = stream_object_number(reference)
        if obj_num is None:
            return default
        with self.lock:
            if obj_num in self.objects:
                return self.objects[obj_num]
            if obj_num not in self.index:
                return default
            rel_offset = self.index[obj_num]
            result = self.parse_object_at(rel_offset, self.ends[rel_offset])
            self.objects[obj_num] = result
            return result

    def get_at_index(self, index: int, *, expected_reference: PdfReference) -> Any:
        with self.lock:
            if (
                type(index) is not int
                or not 0 <= index < len(self.object_numbers)
                or expected_reference.generation_number != 0
                or self.object_numbers[index] != expected_reference.object_number
            ):
                raise PdfParseError("invalid compressed object reference")
            return self.get(expected_reference)

    def object_bytes(self, reference: int | PdfReference) -> bytes | None:
        object_number = stream_object_number(reference)
        if object_number is None:
            return None
        with self.lock:
            offset = self.index.get(object_number)
            if offset is None:
                return None
            return self.raw_body[offset : self.ends[offset]]

    def parse_object_at(self, offset: int, end: int) -> Any:
        if not 0 <= offset < end <= len(self.raw_body):
            raise PdfParseError("invalid object stream object bounds")
        with memoryview(self.raw_body)[offset:end] as body:
            lexer = self.create_lexer(body)
            try:
                result = lexer.parse_object()
                if isinstance(result, (PdfStream, PdfReference)):
                    raise PdfParseError("invalid compressed object value")
                lexer.skip_ignored()
                if lexer.pos != lexer.data_len:
                    raise PdfParseError("unexpected data after compressed object")
                return result
            finally:
                lexer.close()


def stream_object_number(reference: int | PdfReference) -> int | None:
    """The object number `reference` names, or None for a nonzero generation.

    Objects in an object stream have generation 0 (ISO 32000-2 7.5.7).
    """
    object_number = reference.object_number if isinstance(reference, PdfReference) else reference
    if object_number < 0:
        raise ValueError("invalid object number")
    if isinstance(reference, PdfReference) and reference.generation_number != 0:
        return None
    return object_number


def parse_object_stream_pair(lexer: PdfLexer) -> tuple[int, int]:
    values: list[int] = []
    for _ in range(2):
        token = lexer.scan_word()
        if token is None or not is_integer_token(token[0]):
            raise PdfParseError("object stream header is truncated")
        lexer.pos = token[1]
        values.append(lexer.parse_integer_token(token[0]))
    return values[0], values[1]


def parse_object_stream_header(
    data: bytes | memoryview, first: int, n: int, *, context: SemanticContext | None = None
) -> list[tuple[int, int]]:
    if n < 0 or not 0 <= first <= len(data):
        raise PdfParseError("invalid object stream dictionary")
    header = data[:first]
    lexer = PdfLexer(header, semantic_context=context)
    try:
        pairs = [parse_object_stream_pair(lexer) for _ in range(n)]
        lexer.skip_ignored()
        if lexer.pos != len(header):
            raise PdfParseError("unexpected data after object stream header")
        return pairs
    finally:
        lexer.close()


__all__ = (
    "PdfObjectStream",
    "parse_object_stream_header",
    "parse_object_stream_pair",
)
