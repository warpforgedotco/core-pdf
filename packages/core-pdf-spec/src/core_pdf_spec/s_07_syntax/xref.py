# SPDX-License-Identifier: AGPL-3.0-only
"""PDF cross-reference entries, streams, and revision precedence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    parse_int,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import lexical_rules
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfByteBuffer


class PdfXRefEntry:
    __slots__ = ("offset", "generation", "in_use", "object_stream", "index_in_stream")

    offset: int
    generation: int
    in_use: bool
    object_stream: int | None
    index_in_stream: int | None

    def __init__(
        self,
        offset: int,
        generation: int = 0,
        in_use: bool = True,
        object_stream: int | None = None,
        index_in_stream: int | None = None,
    ) -> None:
        self.offset = offset
        self.generation = generation
        self.in_use = in_use
        self.object_stream = object_stream
        self.index_in_stream = index_in_stream


XRefTable = dict[int, PdfXRefEntry]


def key_for(obj_num: int, gen_num: int = 0) -> int:
    if obj_num < 0 or not 0 <= gen_num <= 65535:
        raise ValueError("invalid PDF reference")
    return (obj_num << 16) | gen_num


def parse_xref_entry_at(data: PdfByteBuffer, pos: int) -> tuple[int, int, bool, int]:
    row = bytes(data[pos : pos + 20])
    if (
        len(row) != 20
        or not row[:10].isdigit()
        or row[10] != 32
        or not row[11:16].isdigit()
        or row[16] != 32
        or row[17] not in (102, 110)
        or row[18:] not in (b" \r", b" \n", b"\r\n")
    ):
        raise PdfParseError("invalid xref table entry")
    generation = int(row[11:16])
    if generation > 65535:
        raise PdfParseError("invalid xref generation number")
    return int(row[:10]), generation, row[17] == 110, pos + 20


def parse_xref_entry_line(line: bytes) -> tuple[int, int, bool]:
    offset, generation, in_use, _ = parse_xref_entry_at(line, 0)
    return offset, generation, in_use


class XRefScanner:
    @staticmethod
    def find_startxref(
        data: PdfByteBuffer, *, semantic_context: SemanticContext | None = None
    ) -> int | None:
        rules = lexical_rules(semantic_context)
        eof = find_eof_marker(data, semantic_context=semantic_context)
        if eof < 0:
            return None
        marker = data.rfind(b"startxref", 0, eof)
        if marker < 0 or (marker > 0 and not rules.whitespace_table[data[marker - 1]]):
            return None
        raw = bytes(data[marker + 9 : eof]).strip(rules.whitespace)
        if not raw.isdigit():
            raise PdfParseError("invalid startxref offset")
        return int(raw)

    @staticmethod
    def skip_ws(
        data: PdfByteBuffer, pos: int, *, semantic_context: SemanticContext | None = None
    ) -> int:
        ws = lexical_rules(semantic_context).whitespace_table
        n = len(data)
        while pos < n and ws[data[pos]]:
            pos += 1
        return pos

    @staticmethod
    def skip_ignored(
        data: PdfByteBuffer,
        pos: int,
        stop: int | None = None,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> int:
        ws = lexical_rules(semantic_context).whitespace_table
        n = len(data)
        while pos < n:
            while pos < n and ws[data[pos]]:
                pos += 1
            if pos == stop:
                return pos
            if data[pos : pos + 5] == b"%%EOF":
                return pos
            if pos >= n or data[pos] != 37:
                return pos
            pos += 1
            while pos < n and data[pos] not in (10, 13):
                pos += 1
        return pos

    @staticmethod
    def read_line(data: PdfByteBuffer, pos: int) -> tuple[bytes, int]:
        n = len(data)
        start = pos
        lf = data.find(b"\n", pos)
        if lf < 0:
            cr = data.find(b"\r", pos)
            end = n if cr < 0 else cr
        else:
            cr = data.find(b"\r", pos, lf)
            end = lf if cr < 0 else cr
        pos = end

        if pos < n and data[pos] == 13:
            pos += 1
            if pos < n and data[pos] == 10:
                pos += 1
        elif pos < n and data[pos] == 10:
            pos += 1
        return bytes(data[start:end]), pos

    @staticmethod
    def parse_subsection_integer(token: bytes) -> int:
        value = parse_int(token, None)
        if value is None:
            raise ValueError("invalid PDF integer")
        return value

    @classmethod
    def parse_table_section(
        cls,
        data: PdfByteBuffer,
        start_pos: int,
        *,
        lexer: PdfLexer | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        rules = lexical_rules(semantic_context)
        options = {"semantic_context": semantic_context} if semantic_context is not None else {}
        # Omit new arguments on the historical no-context path so existing
        # subclass overrides of the parsing extension methods remain callable.
        pos = cls.skip_ws(data, start_pos, **options)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = cls.skip_ws(data, pos, **options)

        entries: XRefTable = {}
        max_object_number = -1
        while pos < len(data):
            line, next_pos = cls.read_line(data, pos)
            b_line = line
            if b_line.startswith(b"trailer"):
                trailer_pos = pos + b_line.find(b"trailer") + len(b"trailer")
                pos = trailer_pos
                break
            if b_line.lstrip(rules.whitespace).startswith(b"<<"):
                raise PdfParseError("expected trailer keyword")
            if 11 in b_line:
                raise PdfParseError("invalid xref table subsection")
            normalized = b_line.translate(
                bytes.maketrans(rules.whitespace, b" " * len(rules.whitespace))
            )
            parts = [part for part in normalized.split(b" ") if part]
            if not parts:
                pos = next_pos
                continue
            if len(parts) == 2:
                try:
                    start_obj = cls.parse_subsection_integer(parts[0])
                    num_objs = cls.parse_subsection_integer(parts[1])
                except ValueError as error:
                    raise PdfParseError("invalid xref table subsection") from error
                if start_obj < 0 or num_objs < 0:
                    raise PdfParseError("invalid xref table subsection")
                pos = next_pos
                if num_objs > 0:
                    max_object_number = max(max_object_number, start_obj + num_objs - 1)
                subsection, pos, maximum = cls.read_subsection(data, pos, start_obj, num_objs)
                entries.update(subsection)
                max_object_number = max(max_object_number, maximum)

            else:
                raise PdfParseError("invalid xref table subsection")

        lexer = PdfLexer(data, semantic_context=semantic_context) if lexer is None else lexer
        if semantic_context is not None:
            lexer.semantic_context = semantic_context
        lexer.pos = cls.skip_ws(data, pos, **options)
        try:
            trailer_dict = lexer.parse_dictionary()
        finally:
            lexer.close()
        trailer_dict = cls.validate_trailer_size(trailer_dict, max_object_number)
        prev = trailer_dict.get("Prev")
        xrefstm = trailer_dict.get("XRefStm")
        if prev is not None and type(prev) is not int:
            raise PdfParseError("invalid xref table trailer /Prev")
        if xrefstm is not None and type(xrefstm) is not int:
            raise PdfParseError("invalid xref table trailer /XRefStm")
        return (
            entries,
            trailer_dict,
            prev,
            xrefstm,
        )

    @classmethod
    def read_subsection(
        cls, data: PdfByteBuffer, pos: int, start_obj: int, num_objs: int
    ) -> tuple[XRefTable, int, int]:
        entries: XRefTable = {}
        for i in range(num_objs):
            offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
            entries[key_for(start_obj + i, generation)] = PdfXRefEntry(offset, generation, in_use)
        return entries, pos, start_obj + num_objs - 1

    @classmethod
    def validate_trailer_size(cls, trailer: PdfDict, maximum: int) -> PdfDict:
        size = trailer.get("Size")
        if type(size) is not int or size <= maximum or size <= 0:
            raise PdfParseError("invalid xref trailer Size")
        return trailer

    @classmethod
    def parse_section_at(
        cls,
        data: PdfByteBuffer,
        start: int,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        if data[start : start + 4] == b"xref":
            if semantic_context is None:
                return cls.parse_table_section(data, start)
            return cls.parse_table_section(data, start, semantic_context=semantic_context)
        lexer = PdfLexer(data, semantic_context=semantic_context)
        try:
            lexer.rewind(start)
            obj = lexer.parse_indirect_object()
        finally:
            lexer.close()
        if not isinstance(obj, PdfStream):
            raise PdfParseError("expected xref stream")
        entries, trailer = cls.parse_stream(obj)
        prev = trailer.get("Prev")
        if prev is not None and type(prev) is not int:
            raise PdfParseError("invalid xref stream trailer /Prev")
        return entries, trailer, prev, None

    @classmethod
    def load_section_chain(
        cls,
        data: PdfByteBuffer,
        start: int,
        seen: set[int],
        *,
        semantic_context: SemanticContext | None = None,
    ) -> tuple[XRefTable, PdfDict]:
        options = {"semantic_context": semantic_context} if semantic_context is not None else {}
        section_start = start
        sections: list[XRefTable] = []
        trailer: PdfDict | None = None

        while True:
            if section_start in seen:
                raise PdfParseError("xref section loop detected")
            seen.add(section_start)

            entries, current_trailer, prev, xrefstm = cls.parse_section_at(
                data, section_start, **options
            )
            if trailer is None:
                trailer = current_trailer
            if prev is not None and prev < 0:
                raise PdfParseError("invalid xref section")
            if xrefstm is not None and xrefstm < 0:
                raise PdfParseError("invalid xref section")

            if xrefstm is not None:
                s_entries, ignored = cls.load_section_chain(
                    data,
                    xrefstm,
                    seen,
                    **options,
                )
                # ISO 32000-1 7.5.8.4: "if an entry is not found in any given
                # standard cross-reference section, the search shall proceed to
                # a cross-reference stream specified by the XRefStm entry before
                # looking in the previous cross-reference section". The stream
                # is the fallback, so the classic section overlays it.
                combined = dict(s_entries)
                combined.update(entries)
                entries = combined
            sections.append(entries)

            if prev is None:
                break
            section_start = prev

        return merge_xref_sections(sections), trailer if trailer is not None else {}

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDict]:
        dictionary = stream.dictionary
        if decoded_name(dictionary.get("Type")) != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = dictionary.get("Size")
        widths = dictionary.get("W")
        if type(size) is not int or size <= 0:
            raise PdfParseError("invalid xref stream size")
        if not isinstance(widths, list):
            raise PdfParseError("invalid xref stream W")
        w = cast(list[int], widths)
        row_size = internal_validate_xref_widths(w)
        index = dictionary.get("Index", [0, size])
        if not isinstance(index, list):
            raise PdfParseError("invalid xref stream Index")
        indices = cast(list[int], index)
        row_count = internal_validate_xref_index(indices, size)
        return (
            internal_decode_xref_rows(stream.data, w, indices, row_size, row_count),
            cast(PdfDict, dictionary),
        )


def internal_validate_xref_widths(widths: list[int]) -> int:
    if len(widths) != 3 or any(type(width) is not int or width < 0 for width in widths):
        raise PdfParseError("invalid xref stream W")
    row_size = sum(widths)
    if row_size <= 0:
        raise PdfParseError("invalid xref stream W")
    return row_size


def internal_validate_xref_index(index: list[int], size: int) -> int:
    if type(size) is not int or size <= 0 or len(index) % 2:
        raise PdfParseError("invalid xref stream Index")
    if any(type(value) is not int or value < 0 for value in index):
        raise PdfParseError("invalid xref stream Index")
    if any(index[i] + index[i + 1] > size for i in range(0, len(index), 2)):
        raise PdfParseError("invalid xref stream Index")
    return sum(index[1::2])


def decode_xref_row(
    data: bytes, pos: int, widths: list[int], object_number: int
) -> tuple[int, PdfXRefEntry, int]:
    """Decode one xref-stream row and return its key, entry, and next offset."""
    row_size = internal_validate_xref_widths(widths)
    if object_number < 0:
        raise PdfParseError("invalid xref stream Index")
    end = pos + row_size
    if pos < 0 or end > len(data):
        raise PdfParseError("xref stream length mismatch")
    return internal_decode_xref_row(data, pos, widths, object_number, row_size)


def internal_decode_xref_row(
    data: bytes, pos: int, widths: list[int], object_number: int, row_size: int
) -> tuple[int, PdfXRefEntry, int]:
    """Decode a bounded row using an already validated field layout."""
    end = pos + row_size
    type_end = pos + widths[0]
    offset_end = type_end + widths[1]
    kind = int.from_bytes(data[pos:type_end], "big") if widths[0] else 1
    value = int.from_bytes(data[type_end:offset_end], "big") if widths[1] else 0
    generation = int.from_bytes(data[offset_end:end], "big") if widths[2] else 0
    if kind < 2:
        if generation > 65535:
            raise PdfParseError("invalid xref generation number")
        return key_for(object_number, generation), PdfXRefEntry(value, generation, kind == 1), end
    if kind == 2:
        return (
            key_for(object_number),
            PdfXRefEntry(0, 0, True, object_stream=value, index_in_stream=generation),
            end,
        )
    # ISO 32000-1 7.5.8.3 prescribes null for future entry types.
    return key_for(object_number), PdfXRefEntry(0, 0, False), end


def decode_xref_rows(data: bytes, w: list[int], index: list[int], size: int) -> XRefTable:
    """Validate an entire xref-stream layout and decode its rows in order."""
    row_count = internal_validate_xref_index(index, size)
    row_size = internal_validate_xref_widths(w)
    return internal_decode_xref_rows(data, w, index, row_size, row_count)


def internal_decode_xref_rows(
    data: bytes, widths: list[int], index: list[int], row_size: int, row_count: int
) -> XRefTable:
    if len(data) != row_count * row_size:
        raise PdfParseError("xref stream length mismatch")
    entries: XRefTable = {}
    pos = 0
    for i in range(0, len(index), 2):
        for object_number in range(index[i], index[i] + index[i + 1]):
            key, entry, pos = internal_decode_xref_row(data, pos, widths, object_number, row_size)
            entries[key] = entry
    return entries


def find_eof_marker(data: PdfByteBuffer, *, semantic_context: SemanticContext | None = None) -> int:
    ws = lexical_rules(semantic_context).whitespace_table
    end = len(data)
    while end > 0 and ws[data[end - 1]]:
        end -= 1
    marker = end - 5
    if (
        marker < 0
        or data[marker:end] != b"%%EOF"
        or (marker > 0 and data[marker - 1] not in (10, 13))
    ):
        return -1
    return marker


def parse_object_marker_prefix(
    data: PdfByteBuffer | memoryview,
    marker: int,
    *,
    semantic_context: SemanticContext | None = None,
) -> tuple[int, int, int] | None:
    """Return ``(offset, object number, generation)`` for the header at ``marker``."""
    ws = lexical_rules(semantic_context).whitespace_table
    if marker + 3 < len(data) and not ws[data[marker + 3]]:
        return None
    pos = marker - 1
    while pos >= 0 and ws[data[pos]]:
        pos -= 1
    gen_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    gen_start = pos + 1
    if gen_start == gen_end:
        return None
    while pos >= 0 and ws[data[pos]]:
        pos -= 1
    obj_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    obj_start = pos + 1
    if obj_start == obj_end:
        return None
    if pos >= 0 and not ws[data[pos]]:
        return None
    try:
        object_number = int(data[obj_start:obj_end])
        generation_number = int(data[gen_start:gen_end])
    except ValueError:
        return None
    if generation_number > 65535:
        return None
    return obj_start, object_number, generation_number


def merge_xref_sections(sections: Iterable[XRefTable]) -> XRefTable:
    """Overlay oldest-to-newest revisions supplied in newest-first order."""
    merged: XRefTable = {}
    for section in reversed(list(sections)):
        # ISO 32000-1 7.5.4: a free entry's generation is "the generation
        # number to be used the next time an object with that object number
        # is created" -- not the generation being freed. Keying the entry by
        # it files the deletion under a key no lookup consults, so the body
        # left in the file by 7.5.6 ("deleted objects shall be left
        # unchanged in the file") stayed reachable. Drop every older entry
        # for an object this revision frees, so it resolves to null.
        freed = {key >> 16 for key, entry in section.items() if not entry.in_use}
        if freed:
            for key in [key for key in merged if (key >> 16) in freed]:
                del merged[key]
        merged.update(section)
    return merged


__all__ = (
    "PdfXRefEntry",
    "XRefScanner",
    "XRefTable",
    "decode_xref_row",
    "decode_xref_rows",
    "find_eof_marker",
    "key_for",
    "merge_xref_sections",
    "parse_object_marker_prefix",
    "parse_xref_entry_at",
    "parse_xref_entry_line",
)
