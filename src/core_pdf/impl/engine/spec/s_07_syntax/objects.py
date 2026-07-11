"""PDF object and stream parsing."""

from __future__ import annotations

from typing import TypeVar, overload

from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError
from core_pdf.impl.engine.spec.s_07_syntax.lexer import KeywordCacheValue, PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfObject,
    PdfReference,
    PdfStream,
    parse_int_strict,
    parse_name,
)

DefaultT = TypeVar("DefaultT")


class PdfObjectStream:
    """Decoded object stream with lazy parsing of contained objects."""

    __slots__ = ("stream", "objects", "raw_body", "index", "lexer")

    def __init__(
        self, stream: PdfStream, kw_cache: dict[bytes, KeywordCacheValue] | None = None
    ) -> None:
        type_name = parse_name(stream.dictionary.get("Type"))
        if type_name is not None and type_name != "ObjStm":
            raise PdfParseError("stream is not an object stream")
        n = parse_int_strict(stream.dictionary.get("N"))
        first = parse_int_strict(stream.dictionary.get("First"))
        if n < 0 or first < 0 or first > len(stream.data):
            raise PdfParseError("invalid object stream dictionary")
        header = stream.data[:first]
        fields = header.split()
        if len(fields) != n * 2:
            raise PdfParseError("object stream header is truncated")
        index_map: dict[int, int] = {
            parse_int_strict(fields[i * 2]): parse_int_strict(fields[i * 2 + 1]) for i in range(n)
        }
        if any(obj_num < 0 or offset < 0 for obj_num, offset in index_map.items()):
            raise PdfParseError("invalid object stream header")
        body = stream.data[first:]
        self.stream = stream
        self.objects: dict[int, PdfObject] = {}
        self.raw_body = body
        self.index = index_map
        self.lexer = PdfLexer(body, kw_cache=kw_cache)

    @overload
    def get(self, reference: int | PdfReference) -> PdfObject | None: ...

    @overload
    def get(self, reference: int | PdfReference, default: DefaultT) -> PdfObject | DefaultT: ...

    def get(
        self, reference: int | PdfReference, default: DefaultT | None = None
    ) -> PdfObject | DefaultT | None:
        obj_num = reference.object_number if isinstance(reference, PdfReference) else reference
        if obj_num < 0:
            raise ValueError("invalid object number")
        if obj_num in self.objects:
            return self.objects[obj_num]
        if obj_num not in self.index:
            return default
        rel_offset = self.index[obj_num]
        result = self.lexer.parse_object_at(rel_offset)
        self.objects[obj_num] = result
        return result
