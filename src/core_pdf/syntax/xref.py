# SPDX-License-Identifier: AGPL-3.0-only
"""PDF cross-reference handling."""

from __future__ import annotations

import re
import typing

if typing.TYPE_CHECKING:
    from typing import Any

from core_pdf.syntax.errors import PdfParseError
from core_pdf.syntax.lexer import WS_TABLE, PdfLexer
from core_pdf.syntax.primitives import PdfName, PdfStream, parse_int_strict

STARTXREF_RE = re.compile(b"startxref")
OBJ_MARKER_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj")


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
    """Integer key for xref and object caches."""
    return (obj_num << 16) | gen_num


# Pre-computed lookup for newline detection
NEWLINE_TABLE = bytes([1 if i in (10, 13) else 0 for i in range(256)])


class XRefScanner:
    """Consolidates cross-reference scanning and parsing logic."""

    @staticmethod
    def lookup_entry(
        xref: XRefTable, object_number: int, generation_number: int
    ) -> PdfXRefEntry | None:
        if object_number < 0 or generation_number < 0:
            raise ValueError("invalid xref lookup")
        return xref.get((object_number << 16) | generation_number)

    @staticmethod
    def find_startxref(data: bytes) -> int | None:
        # Use re.finditer for rfind compatibility with memoryview/mmap
        matches = list(STARTXREF_RE.finditer(data))
        if not matches:
            return None
        marker = matches[-1].start()
        pos = marker + 9  # len(b"startxref")
        pos = XRefScanner.skip_ws(data, pos)
        number, _ = XRefScanner.read_line(data, pos)
        return parse_int_strict(bytes(number).strip())

    @staticmethod
    def skip_ws(data: bytes, pos: int) -> int:
        ws = WS_TABLE
        n = len(data)
        while pos < n and ws[data[pos]]:
            pos += 1
        return pos

    @staticmethod
    def read_line(data: bytes, pos: int) -> tuple[memoryview, int]:
        n = len(data)
        start = pos
        nt = NEWLINE_TABLE
        while pos < n and not nt[data[pos]]:
            pos += 1
        end = pos
        # Skip \r\n or \n\r or single \n / \r
        if pos < n and data[pos] == 13:
            pos += 1
            if pos < n and data[pos] == 10:
                pos += 1
        elif pos < n and data[pos] == 10:
            pos += 1
            if pos < n and data[pos] == 13:
                pos += 1
        return memoryview(data[start:end]), pos

    @staticmethod
    def parse_table_section(
        data: bytes, start_pos: int
    ) -> tuple[XRefTable, dict[Any, Any], int | None]:
        pos = XRefScanner.skip_ws(data, start_pos)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = XRefScanner.skip_ws(data, pos)

        entries: XRefTable = {}
        while pos < len(data):
            line, next_pos = XRefScanner.read_line(data, pos)
            if line == b"trailer":
                pos = next_pos
                break
            b_line = bytes(line)
            parts = b_line.strip().split()
            if not parts:
                pos = next_pos
                continue
            if len(parts) == 2:
                try:
                    start_obj = int(parts[0])
                    num_objs = int(parts[1])
                except ValueError:
                    raise PdfParseError("invalid xref table subsection")
                if start_obj < 0 or num_objs < 0:
                    raise PdfParseError("invalid xref table subsection")
                pos = next_pos
                for i in range(num_objs):
                    entry_line, pos = XRefScanner.read_line(data, pos)
                    parts = bytes(entry_line).strip().split()
                    if len(parts) != 3:
                        raise PdfParseError("invalid xref table entry")
                    try:
                        offset = int(parts[0])
                        generation = int(parts[1])
                    except ValueError as exc:
                        raise PdfParseError("invalid xref table entry") from exc
                    if parts[2] == b"n":
                        in_use = True
                    elif parts[2] == b"f":
                        in_use = False
                    else:
                        raise PdfParseError("invalid xref table entry")
                    entries[key_for(start_obj + i, generation)] = PdfXRefEntry(
                        offset, generation, in_use
                    )
            else:
                raise PdfParseError("invalid xref table subsection")

        # Parse trailer dictionary
        lexer = PdfLexer(data)
        lexer.pos = XRefScanner.skip_ws(data, pos)
        trailer_dict = lexer.parse_dictionary()
        prev = trailer_dict.get(PdfName.of(b"Prev"))
        return entries, trailer_dict, int(prev) if isinstance(prev, int) else None

    @staticmethod
    def load_section_chain(
        data: bytes, start: int, seen: set[int]
    ) -> tuple[XRefTable, dict[Any, Any]]:
        if start in seen:
            raise PdfParseError("xref section loop detected")
        seen.add(start)

        # 1. Try XRef Stream first
        lexer = PdfLexer(data)
        lexer.pos = start
        obj = lexer.parse_indirect_object()
        if isinstance(obj, PdfStream):
            entries, trailer = XRefScanner.parse_stream(obj)
            prev = trailer.get(PdfName.of(b"Prev"))
            if isinstance(prev, int):
                p_entries, p_trailer = XRefScanner.load_section_chain(data, prev, seen)
                p_entries.update(entries)
                return p_entries, trailer
            return entries, trailer

        # 2. Fallback to classic XRef table
        entries, trailer, prev = XRefScanner.parse_table_section(data, start)
        if prev is not None:
            p_entries, p_trailer = XRefScanner.load_section_chain(data, prev, seen)
            p_entries.update(entries)
            return p_entries, trailer
        return entries, trailer

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, dict[Any, Any]]:
        dict_obj = stream.dictionary
        type_value = dict_obj.get(PdfName.of(b"Type"))
        type_name = PdfName.of(type_value) if type_value is not None else None
        if type_name is None or type_name.value != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = dict_obj.get(PdfName.of(b"Size"))
        if not isinstance(size, int):
            raise PdfParseError("invalid xref stream size")

        w_raw = dict_obj.get(PdfName.of(b"W"))
        if not isinstance(w_raw, (list, tuple)) or len(w_raw) != 3:
            raise PdfParseError("invalid xref stream W")
        if not all(isinstance(x, int) for x in w_raw):
            raise PdfParseError("invalid xref stream W")
        w = [int(x) for x in w_raw]
        if any(width < 0 for width in w):
            raise PdfParseError("invalid xref stream W")

        index_raw = dict_obj.get(PdfName.of(b"Index"))
        if index_raw is None:
            raise PdfParseError("invalid xref stream Index")
        if not isinstance(index_raw, (list, tuple)):
            raise PdfParseError("invalid xref stream Index")
        if not all(isinstance(x, int) for x in index_raw):
            raise PdfParseError("invalid xref stream Index")
        index = [int(x) for x in index_raw]
        if len(index) % 2 != 0:
            raise PdfParseError("invalid xref stream Index")

        data = stream.data
        entries: XRefTable = {}
        pos = 0
        row_size = sum(w)
        if row_size <= 0:
            raise PdfParseError("invalid xref stream W")

        for i in range(0, len(index), 2):
            start_obj = index[i]
            num_objs = index[i + 1]
            if start_obj < 0 or num_objs < 0 or start_obj + num_objs > size:
                raise PdfParseError("invalid xref stream Index")
            for j in range(num_objs):
                if pos + row_size > len(data):
                    raise PdfParseError("xref stream length mismatch")
                row = data[pos : pos + row_size]
                pos += row_size

                t_bytes = row[: w[0]]
                o_bytes = row[w[0] : w[0] + w[1]]
                g_bytes = row[w[0] + w[1] : w[0] + w[1] + w[2]]

                entry_type = int.from_bytes(t_bytes, "big") if w[0] else 1
                val1 = int.from_bytes(o_bytes, "big") if w[1] else 0
                val2 = int.from_bytes(g_bytes, "big") if w[2] else 0

                obj_num = start_obj + j
                if entry_type == 0:  # Free
                    entries[key_for(obj_num, val2)] = PdfXRefEntry(val1, val2, False)
                elif entry_type == 1:  # Normal
                    entries[key_for(obj_num, val2)] = PdfXRefEntry(val1, val2, True)
                elif entry_type == 2:  # Compressed
                    entries[key_for(obj_num, 0)] = PdfXRefEntry(
                        0, 0, True, object_stream=val1, index_in_stream=val2
                    )
                else:
                    raise PdfParseError("invalid xref stream entry type")

        if pos != len(data):
            raise PdfParseError("xref stream length mismatch")

        return entries, dict_obj

    @staticmethod
    def brute_force_scan(data: bytes, max_entries: int = 100000) -> XRefTable:
        """Fallback: scan for 'obj' markers when xref is missing or corrupt."""
        entries: XRefTable = {}
        for match in OBJ_MARKER_RE.finditer(data):
            if len(entries) >= max_entries:
                break
            try:
                obj_num = int(match.group(1))
                gen_num = int(match.group(2))
                offset = match.start()
                if obj_num < 10000000:
                    entries[key_for(obj_num, gen_num)] = PdfXRefEntry(offset, gen_num, True)
            except ValueError, IndexError:
                continue
        return entries
