# SPDX-License-Identifier: AGPL-3.0-only
"""PDF cross-reference handling."""

from __future__ import annotations

import mmap
import re

from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfParseError
from core_pdf.impl.engine.spec.s_07_syntax.lexer import WS_TABLE, PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfDictLike,
    PdfName,
    PdfStream,
    parse_int_strict,
)

STARTXREF_RE = re.compile(b"startxref")
TRAILER_RE = re.compile(b"trailer")
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
ByteSource = bytes | bytearray | memoryview | mmap.mmap


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
    def find_startxref(data: ByteSource) -> int | None:
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
    def find_trailer_dictionary(data: ByteSource) -> PdfDictLike | None:
        matches = list(TRAILER_RE.finditer(data))
        for match in reversed(matches):
            lexer = PdfLexer(data)
            lexer.pos = XRefScanner.skip_ws(data, match.end())
            try:
                trailer = lexer.parse_dictionary()
            except PdfParseError:
                continue
            return trailer
        return None

    @staticmethod
    def skip_ws(data: ByteSource, pos: int) -> int:
        ws = WS_TABLE
        n = len(data)
        while pos < n and ws[data[pos]]:
            pos += 1
        return pos

    @staticmethod
    def read_line(data: ByteSource, pos: int) -> tuple[memoryview, int]:
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
        data: ByteSource, start_pos: int
    ) -> tuple[XRefTable, PdfDictLike, int | None]:
        pos = XRefScanner.skip_ws(data, start_pos)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = XRefScanner.skip_ws(data, pos)

        entries: XRefTable = {}
        while pos < len(data):
            line, next_pos = XRefScanner.read_line(data, pos)
            b_line = bytes(line)
            stripped_line = b_line.strip()
            if stripped_line.startswith(b"trailer"):
                after_trailer = stripped_line[len(b"trailer") :]
                if after_trailer and not WS_TABLE[after_trailer[0]]:
                    raise PdfParseError("invalid xref table subsection")
                leading_ws = len(b_line) - len(b_line.lstrip())
                pos += leading_ws + len(b"trailer")
                break
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
        data: ByteSource, start: int, seen: set[int]
    ) -> tuple[XRefTable, PdfDictLike]:
        sections: list[tuple[XRefTable, PdfDictLike]] = []
        current = start
        while True:
            if current in seen:
                raise PdfParseError("xref section loop detected")
            seen.add(current)

            pos = XRefScanner.skip_ws(data, current)
            if data[pos : pos + 4] == b"xref":
                entries, trailer, prev = XRefScanner.parse_table_section(data, pos)
                sections.append((entries, trailer))
                if prev is None:
                    break
                current = prev
                continue

            # Otherwise the startxref offset must point at an xref stream object.
            lexer = PdfLexer(data)
            lexer.pos = pos
            obj = lexer.parse_indirect_object()
            if not isinstance(obj, PdfStream):
                raise PdfParseError("expected xref stream")

            entries, trailer = XRefScanner.parse_stream(obj)
            sections.append((entries, trailer))
            stream_prev = trailer.get(PdfName.of(b"Prev"))
            if not isinstance(stream_prev, int):
                break
            current = stream_prev

        if not sections:
            raise PdfParseError("missing xref section")
        merged: XRefTable = {}
        for entries, _trailer in reversed(sections):
            merged.update(entries)
        return merged, sections[0][1]

    @staticmethod
    def offset_matches_object(
        data: ByteSource, offset: int, object_number: int, generation_number: int
    ) -> bool:
        if offset < 0 or offset >= len(data):
            return False
        match = OBJ_MARKER_RE.match(data, XRefScanner.skip_ws(data, offset))
        if match is None:
            return False
        try:
            return int(match.group(1)) == object_number and int(match.group(2)) == generation_number
        except ValueError:
            return False

    @staticmethod
    def repair_misaligned_entries(data: ByteSource, entries: XRefTable) -> XRefTable:
        scanned: XRefTable | None = None
        repaired: XRefTable | None = None
        for key, entry in entries.items():
            if not entry.in_use or entry.object_stream is not None:
                continue
            obj_num = key >> 16
            gen_num = key & 0xFFFF
            if XRefScanner.offset_matches_object(data, entry.offset, obj_num, gen_num):
                continue
            if scanned is None:
                scanned = XRefScanner.brute_force_scan(data)
            replacement = scanned.get(key)
            if replacement is None:
                continue
            if repaired is None:
                repaired = entries.copy()
            repaired[key] = replacement

        if repaired is None:
            return entries
        assert scanned is not None
        for key, replacement in scanned.items():
            current = repaired.get(key)
            if current is None or not current.in_use:
                repaired[key] = replacement
        return repaired

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDictLike]:
        dict_obj = stream.dictionary
        type_value = dict_obj.get(PdfName.of(b"Type"))
        type_name = (
            PdfName.of(type_value)
            if isinstance(type_value, (str, bytes, memoryview, PdfName))
            else None
        )
        if type_name is None or type_name.value != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = dict_obj.get(PdfName.of(b"Size"))
        if not isinstance(size, int):
            raise PdfParseError("invalid xref stream size")

        w_raw = dict_obj.get(PdfName.of(b"W"))
        if not isinstance(w_raw, (list, tuple)) or len(w_raw) != 3:
            raise PdfParseError("invalid xref stream W")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in w_raw):
            raise PdfParseError("invalid xref stream W")
        w = [x for x in w_raw if isinstance(x, int) and not isinstance(x, bool)]
        if any(width < 0 for width in w):
            raise PdfParseError("invalid xref stream W")

        index_raw = dict_obj.get(PdfName.of(b"Index"))
        if index_raw is None:
            index = [0, size]
        elif not isinstance(index_raw, (list, tuple)):
            raise PdfParseError("invalid xref stream Index")
        else:
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in index_raw):
                raise PdfParseError("invalid xref stream Index")
            index = [x for x in index_raw if isinstance(x, int) and not isinstance(x, bool)]
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
    def brute_force_scan(data: ByteSource, max_entries: int = 100000) -> XRefTable:
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
            except (ValueError, IndexError):
                continue
        return entries

    @staticmethod
    def recover_missing_xref(data: ByteSource) -> tuple[XRefTable, PdfDictLike]:
        entries = XRefScanner.brute_force_scan(data)
        trailer = XRefScanner.find_trailer_dictionary(data)
        if not entries or trailer is None:
            raise PdfParseError("missing startxref")
        return entries, trailer
