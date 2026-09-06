# SPDX-License-Identifier: AGPL-3.0-only
"""PDF cross-reference entries, streams, and revision precedence."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import WS_TABLE
from core_pdf.impl.types import PdfByteBuffer


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
    def find_startxref(data: PdfByteBuffer) -> int | None:
        eof = find_eof_marker(data)
        if eof < 0:
            return None
        marker = data.rfind(b"startxref", 0, eof)
        if marker < 0 or (marker > 0 and not WS_TABLE[data[marker - 1]]):
            return None
        raw = bytes(data[marker + 9 : eof]).strip(bytes((0, 9, 10, 12, 13, 32)))
        if not raw.isdigit():
            raise PdfParseError("invalid startxref offset")
        return int(raw)

    @staticmethod
    def skip_ws(data: PdfByteBuffer, pos: int) -> int:
        ws = WS_TABLE
        n = len(data)
        while pos < n and ws[data[pos]]:
            pos += 1
        return pos

    @staticmethod
    def skip_ignored(data: PdfByteBuffer, pos: int, stop: int | None = None) -> int:
        n = len(data)
        while pos < n:
            while pos < n and WS_TABLE[data[pos]]:
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
    def internal_subsection_integer(token: bytes) -> int:
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
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        pos = cls.skip_ws(data, start_pos)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = cls.skip_ws(data, pos)

        entries: XRefTable = {}
        max_object_number = -1
        while pos < len(data):
            line, next_pos = cls.read_line(data, pos)
            b_line = line
            if b_line.startswith(b"trailer"):
                trailer_pos = pos + b_line.find(b"trailer") + len(b"trailer")
                pos = trailer_pos
                break
            if b_line.lstrip().startswith(b"<<"):
                cls.internal_missing_trailer_keyword()
                break
            if 11 in b_line:
                raise PdfParseError("invalid xref table subsection")
            parts = b_line.strip().split()
            if not parts:
                pos = next_pos
                continue
            if len(parts) == 2:
                try:
                    start_obj = cls.internal_subsection_integer(parts[0])
                    num_objs = cls.internal_subsection_integer(parts[1])
                except ValueError as error:
                    raise PdfParseError("invalid xref table subsection") from error
                if start_obj < 0 or num_objs < 0:
                    raise PdfParseError("invalid xref table subsection")
                pos = next_pos
                if num_objs > 0:
                    max_object_number = max(max_object_number, start_obj + num_objs - 1)
                subsection, pos, maximum = cls.internal_read_subsection(
                    data, pos, start_obj, num_objs
                )
                entries.update(subsection)
                max_object_number = max(max_object_number, maximum)

            else:
                raise PdfParseError("invalid xref table subsection")

        lexer = PdfLexer(data) if lexer is None else lexer
        lexer.pos = cls.skip_ws(data, pos)
        try:
            trailer_dict = lexer.parse_dictionary()
        finally:
            lexer.close()
        trailer_dict = cls.internal_trailer_size(trailer_dict, max_object_number)
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
    def internal_read_subsection(
        cls, data: PdfByteBuffer, pos: int, start_obj: int, num_objs: int
    ) -> tuple[XRefTable, int, int]:
        entries: XRefTable = {}
        for i in range(num_objs):
            offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
            entries[key_for(start_obj + i, generation)] = PdfXRefEntry(offset, generation, in_use)
        return entries, pos, start_obj + num_objs - 1

    @classmethod
    def internal_trailer_size(cls, trailer: PdfDict, maximum: int) -> PdfDict:
        size = trailer.get("Size")
        if type(size) is not int or size <= maximum or size <= 0:
            raise PdfParseError("invalid xref trailer Size")
        return trailer

    @classmethod
    def internal_missing_trailer_keyword(cls) -> None:
        raise PdfParseError("expected trailer keyword")

    @classmethod
    def parse_section_at(
        cls, data: PdfByteBuffer, start: int
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        if data[start : start + 4] == b"xref":
            return cls.parse_table_section(data, start)
        lexer = PdfLexer(data)
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
        read_section: Callable[[int], tuple[XRefTable, PdfDict, int | None, int | None]]
        | None = None,
    ) -> tuple[XRefTable, PdfDict]:
        reader = (
            (lambda offset: cls.parse_section_at(data, offset))
            if read_section is None
            else read_section
        )
        section_start = start
        sections: list[XRefTable] = []
        trailer: PdfDict | None = None

        while True:
            if section_start in seen:
                raise PdfParseError("xref section loop detected")
            seen.add(section_start)

            entries, current_trailer, prev, xrefstm = reader(section_start)
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
                    read_section=reader,
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

        merged: XRefTable = {}
        for section in reversed(sections):
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
        return merged, trailer if trailer is not None else {}

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDict]:
        dictionary = stream.dictionary
        if normalize_pdf_name(dictionary.get("Type")) != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = dictionary.get("Size")
        widths = dictionary.get("W")
        if type(size) is not int or size <= 0:
            raise PdfParseError("invalid xref stream size")
        if (
            not isinstance(widths, list)
            or len(widths) != 3
            or not all(type(w) is int and w >= 0 for w in widths)
        ):
            raise PdfParseError("invalid xref stream W")
        index = dictionary.get("Index", [0, size])
        if (
            not isinstance(index, list)
            or len(index) % 2
            or not all(type(v) is int and v >= 0 for v in index)
        ):
            raise PdfParseError("invalid xref stream Index")
        w = cast(list[int], widths)
        indices = cast(list[int], index)
        if any(indices[i] + indices[i + 1] > size for i in range(0, len(indices), 2)):
            raise PdfParseError("invalid xref stream Index")
        data = stream.data
        if sum(w) <= 0 or len(data) != sum(indices[1::2]) * sum(w):
            raise PdfParseError("xref stream length mismatch")
        return decode_xref_rows(data, w, indices, size), cast(PdfDict, dictionary)


def decode_xref_rows(
    data: bytes,
    w: list[int],
    index: list[int],
    size: int,
    *,
    on_invalid_generation: Callable[[int], None] | None = None,
) -> XRefTable:
    entries: XRefTable = {}
    pos = 0
    row_size = sum(w)
    if row_size <= 0:
        raise PdfParseError("invalid xref stream W")

    for i in range(0, len(index), 2):
        start_obj = index[i]
        num_objs = index[i + 1]
        if start_obj < 0 or num_objs < 0:
            raise PdfParseError("invalid xref stream Index")
        rows_to_read = num_objs
        # Everything the row decode needs is loop-invariant: the field
        # widths, their running offsets, and the buffer length. Slicing the
        # three fields straight out of `data` also avoids materialising the
        # whole row first -- four bytes objects per row rather than one.
        # key_for is spelled out below for the same reason: this is the one
        # loop where a call per row is measurable. Elsewhere, call it.
        type_width = w[0]
        offset_width = w[1]
        gen_width = w[2]
        data_len = len(data)
        from_bytes = int.from_bytes
        for j in range(rows_to_read):
            row_end = pos + row_size
            if row_end > data_len:
                raise PdfParseError("xref stream length mismatch")
            type_end = pos + type_width
            offset_end = type_end + offset_width
            entry_type = from_bytes(data[pos:type_end], "big") if type_width else 1
            val1 = from_bytes(data[type_end:offset_end], "big") if offset_width else 0
            val2 = from_bytes(data[offset_end : offset_end + gen_width], "big") if gen_width else 0
            pos = row_end

            obj_num = start_obj + j
            if obj_num >= size:
                continue
            if entry_type < 2:
                if val2 > 65535:
                    if on_invalid_generation is None:
                        raise PdfParseError("invalid xref generation number")
                    on_invalid_generation(val2)
                entries[(obj_num << 16) | val2] = PdfXRefEntry(val1, val2, entry_type == 1)
            elif entry_type == 2:
                entries[obj_num << 16] = PdfXRefEntry(
                    0, 0, True, object_stream=val1, index_in_stream=val2
                )
            else:
                # ISO 32000-1 7.5.8.3: "Any other value shall be interpreted
                # as a reference to the null object, thus permitting new
                # entry types to be defined in the future." One forward-
                # compatible row must not reject the whole section.
                entries[obj_num << 16] = PdfXRefEntry(0, 0, False)

    return entries


def find_eof_marker(data: PdfByteBuffer) -> int:
    end = len(data)
    while end > 0 and WS_TABLE[data[end - 1]]:
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
    data: PdfByteBuffer | memoryview, marker: int
) -> tuple[int, int, int] | None:
    """Return ``(offset, object number, generation)`` for the header at ``marker``."""
    if marker + 3 < len(data) and not WS_TABLE[data[marker + 3]]:
        return None
    pos = marker - 1
    while pos >= 0 and WS_TABLE[data[pos]]:
        pos -= 1
    gen_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    gen_start = pos + 1
    if gen_start == gen_end:
        return None
    while pos >= 0 and WS_TABLE[data[pos]]:
        pos -= 1
    obj_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    obj_start = pos + 1
    if obj_start == obj_end:
        return None
    if pos >= 0 and not WS_TABLE[data[pos]]:
        return None
    try:
        return obj_start, int(data[obj_start:obj_end]), int(data[gen_start:gen_end])
    except ValueError:
        return None
