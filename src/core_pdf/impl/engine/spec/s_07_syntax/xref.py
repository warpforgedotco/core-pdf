# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import typing
import zlib
from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
    parse_int_strict,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import WS_TABLE, PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.objects import PdfObjectStream
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfByteBuffer, PdfDict


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


def parse_xref_entry_line(line: bytes) -> tuple[int, int, bool]:
    n = len(line)
    if n >= 18:
        marker = line[17]
        if line[10] in (9, 32) and line[16] in (9, 32) and marker in (102, 110):
            try:
                offset = int(line[:10])
                generation = int(line[11:16])
            except ValueError:
                pass
            else:
                if offset < 0 or generation < 0:
                    raise PdfParseError("invalid xref table entry")
                if n > 18 and not WS_TABLE[line[18]]:
                    raise PdfParseError("invalid xref table entry")
                return offset, generation, marker == 110

    parts = line.strip().split()
    if len(parts) == 2:
        try:
            offset = int(parts[0])
            generation = int(parts[1])
        except ValueError as exc:
            raise PdfParseError("invalid xref table entry") from exc
        if offset < 0 or generation < 0:
            raise PdfParseError("invalid xref table entry")
        return offset, generation, offset != 0
    if len(parts) != 3:
        raise PdfParseError("invalid xref table entry")
    try:
        offset = int(parts[0])
        generation = int(parts[1])
    except ValueError as exc:
        raise PdfParseError("invalid xref table entry") from exc
    if offset < 0 or generation < 0:
        raise PdfParseError("invalid xref table entry")
    if parts[2] == b"n":
        in_use = True
    elif parts[2] == b"f":
        in_use = False
    else:
        raise PdfParseError("invalid xref table entry")
    return offset, generation, in_use


def parse_xref_entry_at(data: PdfByteBuffer, pos: int) -> tuple[int, int, bool, int]:
    n = len(data)
    if pos + 18 <= n:
        marker = data[pos + 17]
        if data[pos + 10] in (9, 32) and data[pos + 16] in (9, 32) and marker in (102, 110):
            try:
                offset = int(data[pos : pos + 10])
                generation = int(data[pos + 11 : pos + 16])
            except ValueError:
                pass
            else:
                next_pos = pos + 18
                if next_pos < n:
                    while next_pos < n and data[next_pos] in (9, 32):
                        next_pos += 1
                if next_pos < n:
                    byte = data[next_pos]
                    if byte == 13:
                        next_pos += 1
                        if next_pos < n and data[next_pos] == 10:
                            next_pos += 1
                    elif byte == 10:
                        next_pos += 1
                        if next_pos < n and data[next_pos] == 13:
                            next_pos += 1
                    elif not WS_TABLE[byte]:
                        raise PdfParseError("invalid xref table entry")
                return offset, generation, marker == 110, next_pos

    entry_line, next_pos = XRefScanner.read_line(data, pos)
    offset, generation, in_use = parse_xref_entry_line(entry_line)
    return offset, generation, in_use, next_pos


class XRefScanner:
    @staticmethod
    def lookup_entry(
        xref: XRefTable, object_number: int, generation_number: int
    ) -> PdfXRefEntry | None:
        if object_number < 0 or generation_number < 0:
            raise ValueError("invalid xref lookup")
        return xref.get(key_for(object_number, generation_number))

    @staticmethod
    def find_startxref(data: PdfByteBuffer) -> int | None:
        eof_pos = find_eof_marker(data)
        has_eof = eof_pos >= 0
        if not has_eof:
            eof_pos = len(data)

        search_end = eof_pos + 1 if has_eof else len(data)
        while True:
            marker = data.rfind(b"startxref", 0, search_end)
            if marker < 0:
                break
            search_end = marker
            if marker < 0 or marker + 9 > len(data):
                continue

            if marker > 0 and not WS_TABLE[data[marker - 1]]:
                continue
            if marker + 9 >= len(data) or not WS_TABLE[data[marker + 9]]:
                continue

            if marker > eof_pos:
                continue

            pos = XRefScanner.skip_ws(data, marker + 9)
            startxref_number_bytes, ignored = XRefScanner.read_line(data, pos)
            if not startxref_number_bytes:
                continue

            number_end = pos + len(startxref_number_bytes)
            number_parts = startxref_number_bytes.strip().split(None, 1)
            if not number_parts:
                continue
            number_bytes = number_parts[0]
            if b"%" in number_bytes:
                number_bytes = number_bytes.split(b"%", 1)[0]
            number_end = pos + startxref_number_bytes.find(number_bytes) + len(number_bytes)
            next_pos = XRefScanner.skip_ignored(
                data,
                number_end,
                stop=eof_pos if has_eof else None,
            )
            if has_eof:
                if next_pos != eof_pos:
                    continue
            elif next_pos != len(data):
                continue

            try:
                return parse_int_strict(number_bytes)
            except ValueError:
                continue

        for candidate in XRefScanner.find_nearby_sections(data, eof_pos, window=len(data)):
            if candidate >= eof_pos:
                continue
            try:
                XRefScanner.parse_section_at(data, candidate)
            except PdfParseError:
                continue
            return candidate

        return None

    @staticmethod
    def find_nearby_section(data: PdfByteBuffer, start: int, window: int = 1024) -> int | None:
        candidates = XRefScanner.find_nearby_sections(data, start, window)
        return candidates[0] if candidates else None

    @staticmethod
    def find_nearby_sections(data: PdfByteBuffer, start: int, window: int = 1024) -> list[int]:
        n = len(data)
        if start < 0:
            return []
        search_start = max(0, start - window)
        search_end = min(n, start + window)

        candidates: set[int] = set()
        pos = data.find(b"xref", search_start, search_end)
        while pos >= 0:
            if pos > 0 and not WS_TABLE[data[pos - 1]]:
                pos = data.find(b"xref", pos + 1, search_end)
                continue
            after = pos + 4
            if after >= n or WS_TABLE[data[after]]:
                candidates.add(pos)
            pos = data.find(b"xref", pos + 1, search_end)

        type_pos = data.find(b"/Type", search_start, search_end)
        while type_pos >= 0:
            xref_pos = data.find(b"/XRef", type_pos, min(search_end, type_pos + 64))
            if xref_pos >= 0:
                object_marker = find_previous_object_marker(data, type_pos)
                if object_marker is not None:
                    candidates.add(object_marker)
            type_pos = data.find(b"/Type", type_pos + 5, search_end)

        return sorted(candidates, key=lambda candidate: (abs(candidate - start), candidate))

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
            if pos < n and data[pos] == 13:
                pos += 1
        return bytes(data[start:end]), pos

    @staticmethod
    def parse_table_section(
        data: PdfByteBuffer, start_pos: int
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        pos = XRefScanner.skip_ws(data, start_pos)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = XRefScanner.skip_ws(data, pos)

        entries: XRefTable = {}
        max_object_number = -1
        while pos < len(data):
            line, next_pos = XRefScanner.read_line(data, pos)
            b_line = line
            if b_line.startswith(b"trailer"):
                trailer_pos = pos + b_line.find(b"trailer") + len(b"trailer")
                pos = trailer_pos
                break
            if b_line.lstrip().startswith(b"<<"):
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
                if num_objs > 0:
                    max_object_number = max(max_object_number, start_obj + num_objs - 1)
                actual_count = 0
                for i in range(num_objs):
                    entry_pos = XRefScanner.skip_ws(data, pos)
                    if data[entry_pos : entry_pos + 7] == b"trailer":
                        pos = entry_pos
                        break
                    offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
                    entries[((start_obj + i) << 16) | generation] = PdfXRefEntry(
                        offset, generation, in_use
                    )
                    actual_count += 1
                while True:
                    entry_pos = XRefScanner.skip_ws(data, pos)
                    if data[entry_pos : entry_pos + 7] == b"trailer":
                        pos = entry_pos
                        break
                    line, ignored = XRefScanner.read_line(data, entry_pos)
                    parts = line.strip().split()
                    if len(parts) != 3 or parts[2] not in (b"f", b"n"):
                        break
                    try:
                        offset, generation, in_use, pos = parse_xref_entry_at(data, entry_pos)
                    except PdfParseError:
                        break
                    obj_num = start_obj + actual_count
                    entries[(obj_num << 16) | generation] = PdfXRefEntry(offset, generation, in_use)
                    max_object_number = max(max_object_number, obj_num)
                    actual_count += 1
            else:
                raise PdfParseError("invalid xref table subsection")

        lexer = PdfLexer(data)
        lexer.pos = XRefScanner.skip_ws(data, pos)
        trailer_dict = lexer.parse_dictionary()
        trailer_size = lookup_dict_key(trailer_dict, "Size")
        if type(trailer_size) is not int or trailer_size <= max_object_number or trailer_size <= 0:
            trailer_dict = dict(trailer_dict)
            trailer_dict["Size"] = max(max_object_number + 1, 1)
        prev = lookup_dict_key(trailer_dict, "Prev")
        xrefstm = lookup_dict_key(trailer_dict, "XRefStm")
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

    @staticmethod
    def parse_section_at(
        data: PdfByteBuffer, start: int
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        pos = XRefScanner.skip_ws(data, start)

        if data[pos : pos + 4] == b"xref":
            return XRefScanner.parse_table_section(data, pos)

        lexer = PdfLexer(data)
        lexer.pos = pos
        try:
            obj = lexer.parse_indirect_object()
        except PdfParseError:
            obj = XRefScanner.parse_xref_stream_salvage(data, pos)
            if obj is None:
                raise
        if not isinstance(obj, PdfStream):
            raise PdfParseError("expected xref stream")
        try:
            entries, trailer = XRefScanner.parse_stream(obj)
        except PdfParseError:
            obj = XRefScanner.parse_xref_stream_salvage(data, pos)
            if obj is None:
                raise
            entries, trailer = XRefScanner.parse_stream(obj)
        prev = lookup_dict_key(trailer, "Prev")
        if prev is not None and type(prev) is not int:
            raise PdfParseError("invalid xref stream trailer /Prev")
        return entries, trailer, prev, None

    @staticmethod
    def load_section_chain(
        data: PdfByteBuffer, start: int, seen: set[int]
    ) -> tuple[XRefTable, PdfDict]:
        section_start = start
        sections: list[XRefTable] = []
        trailer: PdfDict | None = None

        while True:
            if section_start in seen:
                raise PdfParseError("xref section loop detected")
            seen.add(section_start)

            try:
                entries, current_trailer, prev, xrefstm = XRefScanner.parse_section_at(
                    data, section_start
                )
            except PdfParseError as original_error:
                recovered = None
                for nearby in XRefScanner.find_nearby_sections(data, section_start):
                    if nearby in seen:
                        continue
                    try:
                        recovered = XRefScanner.parse_section_at(data, nearby)
                    except PdfParseError:
                        continue
                    seen.add(nearby)
                    break
                if recovered is None:
                    raise original_error
                entries, current_trailer, prev, xrefstm = recovered
            if trailer is None:
                trailer = current_trailer
            if prev is not None and prev < 0:
                raise PdfParseError("invalid xref section")
            if xrefstm is not None and xrefstm < 0:
                raise PdfParseError("invalid xref section")

            if xrefstm is not None:
                s_entries, ignored = XRefScanner.load_section_chain(data, xrefstm, seen)
                entries.update(s_entries)
            sections.append(entries)

            if prev is None:
                break
            section_start = prev

        merged: XRefTable = {}
        for section in reversed(sections):
            merged.update(section)
        return merged, trailer if trailer is not None else {}

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDict]:
        dict_obj = stream.dictionary
        type_value = lookup_dict_key(dict_obj, "Type")
        type_name = normalize_pdf_name(type_value)
        if type_name is not None and type_name != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = lookup_dict_key(dict_obj, "Size")
        if type(size) is not int or size <= 0:
            raise PdfParseError("invalid xref stream size")

        w_raw = lookup_dict_key(dict_obj, "W")
        if not isinstance(w_raw, (list, tuple)) or len(w_raw) < 3:
            raise PdfParseError("invalid xref stream W")
        if not all(type(x) is int for x in w_raw):
            raise PdfParseError("invalid xref stream W")
        w = [int(cast(typing.Any, x)) for x in w_raw[:3]]
        if any(width < 0 for width in w):
            raise PdfParseError("invalid xref stream W")

        index_raw = lookup_dict_key(dict_obj, "Index")
        if index_raw is None:
            index = [0, size]
        elif not isinstance(index_raw, (list, tuple)) or not all(type(x) is int for x in index_raw):
            raise PdfParseError("invalid xref stream Index")
        else:
            index = [int(cast(typing.Any, x)) for x in index_raw]
            if len(index) % 2 != 0:
                index = index[:-1]
        effective_size = size
        for i in range(0, len(index), 2):
            start_obj = index[i]
            num_objs = index[i + 1]
            if start_obj >= 0 and num_objs >= 0 and start_obj <= size:
                effective_size = max(effective_size, start_obj + num_objs)

        data = stream.data
        entries: XRefTable = {}
        pos = 0
        row_size = sum(w)
        if row_size <= 0:
            raise PdfParseError("invalid xref stream W")
        max_rows = len(data) // row_size

        for i in range(0, len(index), 2):
            start_obj = index[i]
            num_objs = index[i + 1]
            if start_obj < 0 or num_objs < 0:
                raise PdfParseError("invalid xref stream Index")
            remaining_rows = max_rows - (pos // row_size)
            if remaining_rows <= 0:
                break
            rows_to_read = min(num_objs, remaining_rows)
            for j in range(rows_to_read):
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
                if obj_num >= effective_size:
                    continue
                if entry_type == 0:
                    entries[key_for(obj_num, val2)] = PdfXRefEntry(val1, val2, False)
                elif entry_type == 1:
                    entries[key_for(obj_num, val2)] = PdfXRefEntry(val1, val2, True)
                elif entry_type == 2:
                    entries[key_for(obj_num, 0)] = PdfXRefEntry(
                        0, 0, True, object_stream=val1, index_in_stream=val2
                    )
                else:
                    raise PdfParseError("invalid xref stream entry type")

        return entries, typing.cast(PdfDict, dict_obj)

    @staticmethod
    def parse_xref_stream_salvage(data: PdfByteBuffer, pos: int) -> PdfStream | None:
        lexer = PdfLexer(data)
        dict_start = data.find(b"<<", pos)
        if dict_start < 0:
            return None
        lexer.pos = dict_start
        try:
            dict_obj = typing.cast(PdfDict, lexer.parse_dictionary())
        except PdfParseError:
            return None
        if normalize_pdf_name(lookup_dict_key(dict_obj, "Type")) != "XRef":
            return None

        lexer.skip_ignored()
        stream_pos = lexer.pos
        if data[stream_pos : stream_pos + 6] != b"stream":
            return None
        after_stream = stream_pos + 6
        if after_stream >= len(data) or not WS_TABLE[data[after_stream]]:
            return None
        if after_stream < len(data) and data[after_stream] not in (10, 13):
            while after_stream < len(data) and data[after_stream] in (0, 9, 12, 32):
                after_stream += 1
        lexer.pos = after_stream
        lexer.skip_eol()
        data_start = lexer.pos

        length = lookup_dict_key(dict_obj, "Length")
        if type(length) is int and length >= 0 and data_start + length <= len(data):
            data_end = data_start + length
            if lexer.find_object_end(data_end) < 0:
                return None
            raw_data = data[data_start:data_end]
        else:
            endstream = lexer.find_stream_end(data_start)
            if endstream < data_start or lexer.find_object_end(endstream + 9) < 0:
                return None
            raw_data = data[data_start:endstream]
        decoded_data = None
        filter_name = normalize_pdf_name(lookup_dict_key(dict_obj, "Filter"))
        if filter_name == "FlateDecode":
            try:
                decoded_data = zlib.decompress(raw_data)
            except zlib.error:
                try:
                    decoder = zlib.decompressobj()
                    decoded_data = decoder.decompress(raw_data) + decoder.flush()
                except zlib.error as exc:
                    raise PdfParseError("invalid xref stream") from exc
            w = lookup_dict_key(dict_obj, "W")
            index = lookup_dict_key(dict_obj, "Index")
            if isinstance(w, list) and isinstance(index, list):
                row_size = sum(item for item in w if type(item) is int)
                row_count = sum(
                    index[i + 1] for i in range(0, len(index) - 1, 2) if type(index[i + 1]) is int
                )
                if len(decoded_data) != row_size * row_count:
                    decoded_data = None
        return PdfStream(dict_obj, raw_data, None, decoded_data=decoded_data)

    @staticmethod
    def brute_force_scan(data: PdfByteBuffer, max_entries: int = 100000) -> XRefTable:
        entries: XRefTable = {}
        for offset, obj_num, gen_num in iter_object_markers(data):
            if len(entries) >= max_entries:
                break
            try:
                if obj_num < 10000000:
                    entries[key_for(obj_num, gen_num)] = PdfXRefEntry(offset, gen_num, True)
            except (ValueError, IndexError):
                continue
        XRefScanner.recover_object_stream_entries(data, entries, max_entries)
        return entries

    @staticmethod
    def recover_object_stream_entries(
        data: PdfByteBuffer, entries: XRefTable, max_entries: int = 100000
    ) -> None:
        lexer = PdfLexer(data)
        for key, entry in list(entries.items()):
            if len(entries) >= max_entries:
                return
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            obj_num = key >> 16
            gen_num = key & 0xFFFF
            lexer.rewind(entry.offset)
            try:
                obj = lexer.parse_indirect_object()
            except Exception:
                continue
            if not isinstance(obj, PdfStream):
                continue
            dictionary = obj.dictionary
            type_name = normalize_pdf_name(lookup_dict_key(dictionary, "Type"))
            if type_name != "ObjStm" and (
                lookup_dict_key(dictionary, "N") is None
                or lookup_dict_key(dictionary, "First") is None
            ):
                continue
            try:
                container = PdfObjectStream(obj)
            except Exception:
                continue
            for embedded_num in container.index:
                if len(entries) >= max_entries:
                    return
                if embedded_num < 0 or embedded_num >= 10000000:
                    continue
                embedded_key = key_for(embedded_num, 0)
                if embedded_key in entries:
                    continue
                entries[embedded_key] = PdfXRefEntry(
                    0,
                    0,
                    True,
                    object_stream=obj_num,
                    index_in_stream=gen_num,
                )


def find_all_bytes(data: PdfByteBuffer, needle: bytes) -> list[int]:
    positions: list[int] = []
    pos = 0
    while True:
        pos = data.find(needle, pos)
        if pos < 0:
            return positions
        positions.append(pos)
        pos += 1


def find_eof_marker(data: PdfByteBuffer) -> int:
    def is_delimited(marker: int) -> bool:
        before_ok = marker == 0 or data[marker - 1] in (10, 13)
        after = marker + 5
        after_ok = after >= len(data) or bool(WS_TABLE[data[after]])
        return before_ok and after_ok

    raw_exact = data.rfind(b"%%EOF")
    exact = raw_exact
    while exact >= 0:
        if is_delimited(exact):
            return exact
        exact = data.rfind(b"%%EOF", 0, exact)

    search_end = len(data)
    raw_recovered = -1
    while True:
        marker = data.rfind(b"%", 0, search_end)
        if marker < 0:
            return raw_exact if raw_exact >= 0 else raw_recovered
        search_end = marker
        if marker + 5 > len(data):
            continue
        if data[marker : marker + 2] != b"%%":
            continue
        token = data[marker + 2 : marker + 5]
        mismatches = sum(1 for actual, expected in zip(token, b"EOF") if actual != expected)
        if mismatches == 1:
            if raw_recovered < 0:
                raw_recovered = marker
            if is_delimited(marker):
                return marker


def iter_object_markers(
    data: PdfByteBuffer, start: int = 0, stop: int | None = None
) -> typing.Iterator[tuple[int, int, int]]:
    if stop is None:
        stop = len(data)
    pos = start
    while pos < stop:
        marker = data.find(b"obj", pos, stop)
        if marker < 0:
            return
        parsed = parse_object_marker_prefix(data, marker)
        if parsed is not None:
            offset, obj_num, gen_num = parsed
            yield offset, obj_num, gen_num
        pos = marker + 3


def find_previous_object_marker(data: PdfByteBuffer, before: int) -> int | None:
    search_end = min(before, len(data))
    while True:
        marker = data.rfind(b"obj", 0, search_end)
        if marker < 0:
            return None
        parsed = parse_object_marker_prefix(data, marker)
        if parsed is not None:
            return parsed[0]
        search_end = marker


def parse_object_marker_prefix(data: PdfByteBuffer, marker: int) -> tuple[int, int, int] | None:
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
