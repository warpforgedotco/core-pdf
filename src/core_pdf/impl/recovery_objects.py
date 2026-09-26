# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import typing

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream as SyntaxObjectStream
from core_pdf_spec.s_07_syntax.objects import (
    parse_object_stream_pair,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_int_strict,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import WHITESPACE

if typing.TYPE_CHECKING:
    from typing import Any

    from core_pdf_spec.s_07_syntax.types import PdfDict
    from core_pdf_spec.standards import SemanticContext


class PdfObjectStream(SyntaxObjectStream):
    __slots__ = ("body_lexer",)

    def __init__(
        self, stream: PdfStream, *, semantic_context: SemanticContext | None = None
    ) -> None:
        self.body_lexer: PdfLexer | None = None
        super().__init__(stream, semantic_context=semantic_context)

    def close(self) -> None:
        with self.lock:
            if self.body_lexer is not None:
                self.body_lexer.close()
                self.body_lexer = None
            super().close()

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
        dictionary = self.scan_dictionary_at(offset, end)
        if dictionary is not None:
            return dictionary
        try:
            return super().parse_object_at(offset, end)
        except PdfParseError:
            return self.handle_object_error(offset)

    def scan_dictionary_at(self, offset: int, end: int) -> PdfDict | None:
        """The dictionary spanning [offset, end), scanned in place, or None.

        The base parses each object through a lexer of its own over just its
        bytes -- on a document with thousands of compressed annotations,
        building those lexers costs as much as parsing. This reads a
        dictionary with one compiled scanner over the whole body instead, and
        keeps the result only where the per-object lexer must agree: the
        object starts with its "<<", and the scanner's closing ">>" lies
        within the object's span followed by nothing but whitespace. The
        scanner reads no byte past that ">>", so it saw what the object's own
        lexer would, which would find the same dictionary, no stream keyword
        after it and no trailing data. Anything else, including every
        declined dictionary, takes the base path unchanged.
        """
        body = self.raw_body
        if not 0 <= offset < end <= len(body) or body[offset : offset + 2] != b"<<":
            return None
        lexer = self.lexer_over_body()
        parsed = lexer.object_scanner().parse_dictionary(offset)
        if parsed is None:
            return None
        dictionary, dictionary_end = parsed
        if dictionary_end > end:
            return None
        if dictionary_end < end and body[dictionary_end:end].strip(lexer.lexical_rules.whitespace):
            return None
        return dictionary

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

    def lexer_over_body(self) -> PdfLexer:
        """One lexer over the whole body, kept until close: every parse
        through it starts from an explicit position."""
        lexer = self.body_lexer
        if lexer is None:
            lexer = self.body_lexer = self.create_lexer(self.raw_body)
        return lexer

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
        lexer = self.lexer_over_body()
        for pos in [rel_offset, *starts]:
            try:
                return lexer.parse_object_at(pos)
            except PdfParseError:
                continue
        raise PdfParseError("invalid object stream object")


def recover_object_stream_first(data: bytes | memoryview, n: int) -> int | None:
    pairs, last_end = scan_object_stream_pairs(data, n)
    return last_end if pairs else None


def parse_object_stream_header(
    data: bytes | memoryview,
    first: int,
    n: int,
) -> list[tuple[int, int]]:
    return scan_object_stream_pairs(data[:first], n)[0]


HEADER_BYTES = b"0123456789" + WHITESPACE
DIGIT_RUN_RE = re.compile(rb"[0-9]+")


def scan_object_stream_pairs(data: bytes | memoryview, n: int) -> tuple[list[tuple[int, int]], int]:
    if type(data) is bytes and not data.translate(None, HEADER_BYTES):
        return scan_digit_pairs(data, n)
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


def scan_digit_pairs(data: bytes, n: int) -> tuple[list[tuple[int, int]], int]:
    """scan_object_stream_pairs for a header of nothing but digits and whitespace.

    The lexer's words there are the digit runs, each an integer token, so the
    pairs are the runs taken two at a time; an unpaired last run ends the
    scan as the lexer's failed second read did.
    """
    pairs: list[tuple[int, int]] = []
    last_end = 0
    runs = DIGIT_RUN_RE.finditer(data)
    while len(pairs) < n:
        first = next(runs, None)
        second = None if first is None else next(runs, None)
        if first is None or second is None:
            break
        pairs.append((int(first[0]), int(second[0])))
        last_end = second.end()
    return pairs, last_end
