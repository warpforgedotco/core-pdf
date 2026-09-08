# SPDX-License-Identifier: AGPL-3.0-only
"""Read cross-reference tables and streams, with recovery scanning for damaged files."""

from __future__ import annotations

import typing
import zlib
from collections.abc import Callable, Iterator
from typing import cast

from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.document.recovery.objects import PdfObjectStream
from core_pdf.impl._impl.document.recovery.scanning import matches_keyword_with_one_substitution
from core_pdf.impl._impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax.xref import (
    PdfXRefEntry,
    XRefTable,
    decode_xref_rows,
    key_for,
    parse_object_marker_prefix,
)
from core_pdf.impl.spec.s_07_syntax.xref import XRefScanner as SyntaxXRefScanner
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_int_strict,
)
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    FindableSizedBuffer,
    full_source_buffer,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import WS_TABLE
from core_pdf.impl.types import PdfByteBuffer


def parse_xref_entry_line(line: bytes) -> tuple[int, int, bool]:
    """Parse a loosely formatted xref entry.

    Only reached once parse_xref_entry_at's fixed-width form has failed on
    these same bytes, so this does not retry it -- the entry is malformed in
    some way and the whitespace-split form is what is left.
    """
    if 11 in line:
        raise PdfParseError("invalid xref table entry")
    parts = line.strip().split()
    if len(parts) not in (2, 3):
        raise PdfParseError("invalid xref table entry")
    try:
        offset = int(parts[0])
        generation = int(parts[1])
    except ValueError as exc:
        raise PdfParseError("invalid xref table entry") from exc
    if offset < 0:
        raise PdfParseError("invalid xref table entry")
    if not 0 <= generation <= 65535:
        raise PdfParseError("invalid xref generation number")
    if len(parts) == 2:
        # No f/n marker: a zero offset is the free-list head, anything else
        # is an in-use object.
        return offset, generation, offset != 0
    if parts[2] == b"n":
        return offset, generation, True
    if parts[2] == b"f":
        return offset, generation, False
    raise PdfParseError("invalid xref table entry")


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
                if not 0 <= generation <= 65535:
                    raise PdfParseError("invalid xref generation number")
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


class XRefScanner(SyntaxXRefScanner):
    @staticmethod
    def read_line(data: PdfByteBuffer, pos: int) -> tuple[bytes, int]:
        line, next_pos = SyntaxXRefScanner.read_line(data, pos)
        line_end = pos + len(line)
        if (
            line_end < len(data)
            and data[line_end] == 10
            and next_pos < len(data)
            and data[next_pos] == 13
        ):
            next_pos += 1
        return line, next_pos

    @staticmethod
    def internal_subsection_integer(token: bytes) -> int:
        return int(token)

    @staticmethod
    def internal_invalid_generation(generation: int) -> None:
        pass

    @staticmethod
    def recover_object_stream_entries(
        entries: XRefTable,
        parsed_streams: dict[int, tuple[int, PdfStream]],
        max_entries: int = 100000,
    ) -> None:
        for key, entry in list(entries.items()):
            if len(entries) >= max_entries:
                return
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            obj_num = key >> 16
            gen_num = key & 0xFFFF
            parsed = parsed_streams.get(key)
            if parsed is None or parsed[0] != entry.offset:
                continue
            obj = parsed[1]
            if not isinstance(obj, PdfStream):
                continue
            dictionary = obj.dictionary
            type_name = normalize_pdf_name(dictionary.get("Type"))
            if type_name != "ObjStm" and (
                dictionary.get("N") is None or dictionary.get("First") is None
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
            if not startxref_number_bytes or 11 in startxref_number_bytes:
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

    @classmethod
    def parse_section_at(
        cls,
        data: PdfByteBuffer,
        start: int,
        *,
        recover_malformed_objects: bool = True,
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        pos = XRefScanner.skip_ws(data, start)

        if data[pos : pos + 4] == b"xref":
            return XRefScanner.parse_table_section(
                data,
                pos,
                recover_malformed_objects=recover_malformed_objects,
            )

        lexer = PdfLexer(
            data,
            recover_malformed_objects=recover_malformed_objects,
        )
        lexer.pos = pos
        try:
            obj = lexer.parse_indirect_object()
        except PdfParseError:
            obj = XRefScanner.parse_xref_stream_salvage(
                data,
                pos,
                recover_malformed_objects=recover_malformed_objects,
            )
            if obj is None:
                raise
        if not isinstance(obj, PdfStream):
            raise PdfParseError("expected xref stream")
        try:
            entries, trailer = XRefScanner.parse_stream(obj)
        except PdfParseError:
            obj = XRefScanner.parse_xref_stream_salvage(
                data,
                pos,
                recover_malformed_objects=recover_malformed_objects,
            )
            if obj is None:
                raise
            entries, trailer = XRefScanner.parse_stream(obj)
        prev = trailer.get("Prev")
        if prev is not None and type(prev) is not int:
            raise PdfParseError("invalid xref stream trailer /Prev")
        return entries, trailer, prev, None

    @staticmethod
    def parse_xref_stream_salvage(
        data: PdfByteBuffer,
        pos: int,
        *,
        recover_malformed_objects: bool = True,
    ) -> PdfStream | None:
        lexer = PdfLexer(
            data,
            recover_malformed_objects=recover_malformed_objects,
        )
        header_marker = data.find(b"obj", pos, min(len(data), pos + 64))
        if header_marker < 0:
            return None
        parsed_header = parse_object_marker_prefix(data, header_marker)
        if parsed_header is None or parsed_header[0] != pos:
            return None
        lexer.pos = header_marker + 3
        lexer.skip_ignored()
        dict_start = lexer.pos
        if data[dict_start : dict_start + 2] != b"<<":
            return None
        lexer.pos = dict_start
        try:
            dict_obj = lexer.parse_dictionary()
        except PdfParseError:
            return None
        if normalize_pdf_name(dict_obj.get("Type")) != "XRef":
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

        length = dict_obj.get("Length")
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
        filter_name = normalize_pdf_name(dict_obj.get("Filter"))
        if filter_name == "FlateDecode":
            try:
                decoded_data = zlib.decompress(raw_data)
            except zlib.error:
                try:
                    decoder = zlib.decompressobj()
                    decoded_data = decoder.decompress(raw_data) + decoder.flush()
                except zlib.error as exc:
                    raise PdfParseError("invalid xref stream") from exc
            w = dict_obj.get("W")
            index = dict_obj.get("Index")
            if isinstance(w, list) and isinstance(index, list):
                row_size = sum(cast(int, item) for item in w if type(item) is int)
                row_count = sum(
                    cast(int, index[i + 1])
                    for i in range(0, len(index) - 1, 2)
                    if type(index[i + 1]) is int
                )
                if len(decoded_data) != row_size * row_count:
                    decoded_data = None
        return PdfStream(
            dict_obj, raw_data, None, decoded_data=decoded_data, decoder=decode_stream_data
        )

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
    def brute_force_scan(
        data: PdfByteBuffer,
        max_entries: int = 100000,
        *,
        stop_at_first_trailer: bool = False,
    ) -> XRefTable:
        entries: XRefTable = {}
        parsed_streams: dict[int, tuple[int, PdfStream]] = {}
        lexer = PdfLexer(data)
        search_pos = 0
        scan_end = len(data)
        if stop_at_first_trailer:
            trailer = data.find(b"trailer")
            while trailer >= 0:
                after = trailer + 7
                if (trailer == 0 or data[trailer - 1] in (10, 13)) and (
                    after >= len(data) or WS_TABLE[data[after]]
                ):
                    scan_end = trailer
                    break
                trailer = data.find(b"trailer", trailer + 7)
        while search_pos < scan_end:
            if len(entries) >= max_entries:
                break
            marker = data.find(b"obj", search_pos, scan_end)
            if marker < 0:
                break
            search_pos = marker + 3
            parsed_header = parse_object_marker_prefix(data, marker)
            if parsed_header is None:
                continue
            offset, obj_num, gen_num = parsed_header
            lexer.rewind(offset)
            try:
                obj = lexer.parse_indirect_object()
            except Exception:
                if (
                    stop_at_first_trailer
                    and internal_bare_stream_start(data, marker, offset, scan_end) is not None
                ):
                    break
                continue
            if stop_at_first_trailer and not isinstance(obj, PdfStream):
                bare_stream = internal_bare_stream_start(data, marker, offset, scan_end)
                # An unterminated one is what pdfminer's fallback parser treats
                # as running to end of file, taking the rest of the objects with
                # it; a terminated one leaves the following objects readable.
                if (
                    bare_stream is not None
                    and data.find(b"endstream", bare_stream + 6, scan_end) < 0
                ):
                    break
            # A damaged stream /Length can carry the lexer past an earlier
            # endstream/endobj pair and over later, valid indirect objects. In
            # that case keep scanning from the current marker. Otherwise skip
            # the stream payload so object-like binary data cannot replace a
            # genuine xref entry.
            early_stream_end = (
                isinstance(obj, PdfStream)
                and data.find(b"endstream", offset, max(offset, lexer.pos - 9)) >= 0
            )
            if (
                not early_stream_end
                and lexer.pos >= 6
                and data[lexer.pos - 6 : lexer.pos] == b"endobj"
            ):
                search_pos = max(search_pos, lexer.pos)
            if obj_num >= 10000000:
                continue
            key = key_for(obj_num, gen_num)
            entries[key] = PdfXRefEntry(offset, gen_num, True)
            if isinstance(obj, PdfStream):
                parsed_streams[key] = (offset, obj)
        XRefScanner.recover_object_stream_entries(entries, parsed_streams, max_entries)
        return entries

    @classmethod
    def internal_read_subsection(
        cls, data: PdfByteBuffer, pos: int, start_obj: int, num_objs: int
    ) -> tuple[XRefTable, int, int]:
        entries: XRefTable = {}
        max_object_number = start_obj + num_objs - 1
        actual_count = 0
        for i in range(num_objs):
            entry_pos = cls.skip_ws(data, pos)
            if data[entry_pos : entry_pos + 7] == b"trailer":
                pos = entry_pos
                break
            offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
            entries[((start_obj + i) << 16) | generation] = PdfXRefEntry(offset, generation, in_use)
            actual_count += 1
        while True:
            entry_pos = cls.skip_ws(data, pos)
            if data[entry_pos : entry_pos + 7] == b"trailer":
                pos = entry_pos
                break
            line, ignored = cls.read_line(data, entry_pos)
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
        return entries, pos, max_object_number

    @classmethod
    def internal_trailer_size(cls, trailer: PdfDict, maximum: int) -> PdfDict:
        size = trailer.get("Size")
        if type(size) is not int or size <= maximum or size <= 0:
            trailer = dict(trailer)
            trailer["Size"] = max(maximum + 1, 1)
        return trailer

    @classmethod
    def internal_missing_trailer_keyword(cls) -> None:
        pass

    @classmethod
    def parse_table_section(
        cls,
        data: PdfByteBuffer,
        start_pos: int,
        *,
        recover_malformed_objects: bool = True,
        lexer: SyntaxLexer | None = None,
    ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
        return super().parse_table_section(
            data,
            start_pos,
            lexer=PdfLexer(data, recover_malformed_objects=recover_malformed_objects)
            if lexer is None
            else lexer,
        )

    @classmethod
    def load_section_chain(
        cls,
        data: PdfByteBuffer,
        start: int,
        seen: set[int],
        *,
        recover_malformed_objects: bool = True,
        read_section: Callable[[int], tuple[XRefTable, PdfDict, int | None, int | None]]
        | None = None,
    ) -> tuple[XRefTable, PdfDict]:
        if read_section is not None:
            return SyntaxXRefScanner.load_section_chain(
                data, start, seen, read_section=read_section
            )

        def internal_read_section(
            section_start: int,
        ) -> tuple[XRefTable, PdfDict, int | None, int | None]:
            try:
                return cls.parse_section_at(
                    data, section_start, recover_malformed_objects=recover_malformed_objects
                )
            except PdfParseError as original_error:
                for nearby in cls.find_nearby_sections(data, section_start):
                    if nearby in seen:
                        continue
                    try:
                        result = cls.parse_section_at(
                            data, nearby, recover_malformed_objects=recover_malformed_objects
                        )
                    except PdfParseError:
                        continue
                    seen.add(nearby)
                    return result
                raise original_error

        return SyntaxXRefScanner.load_section_chain(
            data, start, seen, read_section=internal_read_section
        )

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDict]:
        dict_obj = stream.dictionary
        type_value = dict_obj.get("Type")
        type_name = normalize_pdf_name(type_value)
        if type_name is not None and type_name != "XRef":
            raise PdfParseError("invalid xref stream type")
        size = dict_obj.get("Size")
        if type(size) is not int or size <= 0:
            raise PdfParseError("invalid xref stream size")

        w_raw = dict_obj.get("W")
        if not isinstance(w_raw, (list, tuple)) or len(w_raw) < 3:
            raise PdfParseError("invalid xref stream W")
        if not all(type(x) is int for x in w_raw):
            raise PdfParseError("invalid xref stream W")
        w = [int(cast(typing.Any, x)) for x in w_raw[:3]]
        if any(width < 0 for width in w):
            raise PdfParseError("invalid xref stream W")

        index_raw = dict_obj.get("Index")
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
        row_size = sum(w)
        if row_size <= 0:
            raise PdfParseError("invalid xref stream W")
        remaining = len(data) // row_size
        available_index: list[int] = []
        for i in range(0, len(index), 2):
            start_obj, count = index[i : i + 2]
            if start_obj < 0 or count < 0:
                raise PdfParseError("invalid xref stream Index")
            if remaining <= 0:
                break
            count = min(count, remaining)
            available_index.extend((start_obj, count))
            remaining -= count
        entries = decode_xref_rows(
            data,
            w,
            available_index,
            effective_size,
            on_invalid_generation=XRefScanner.internal_invalid_generation,
        )
        return entries, typing.cast(PdfDict, dict_obj)


def internal_bare_stream_start(
    data: PdfByteBuffer, marker: int, offset: int, scan_end: int
) -> int | None:
    """Offset of a ``stream`` keyword this object reaches with no dictionary between.

    Returns None when another ``obj`` header comes first, or when the bytes
    between the header and the keyword hold anything but comments. Both callers
    use it to recognise the dictionary-less stream that pdfminer's fallback
    parser swallows to end of file, so object-looking bytes after it are not
    xref entries.
    """
    stream_marker = data.find(b"stream", offset, scan_end)
    next_object_marker = data.find(b"obj", marker + 3, scan_end)
    if stream_marker < 0 or not (next_object_marker < 0 or stream_marker < next_object_marker):
        return None
    prefix = data[marker + 3 : stream_marker]
    uncommented = b"\n".join(line.split(b"%", 1)[0] for line in prefix.splitlines())
    return None if uncommented.strip() else stream_marker


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
        if matches_keyword_with_one_substitution(data, marker + 2, b"EOF"):
            if raw_recovered < 0:
                raw_recovered = marker
            if is_delimited(marker):
                return marker


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


def iter_indirect_object_headers(
    data: PdfByteBuffer | memoryview,
    search_start: int,
    search_end: int,
    *,
    source_buffer: FindableSizedBuffer | None = None,
    allow_prefix_before_start: bool = False,
) -> Iterator[tuple[int, int, int]]:
    """Yield validated headers in marker order within a bounded search window.

    Approximate document offsets may admit a prefix before the marker window;
    object-level recovery requires the entire header to start inside it. The
    keyword is validated against the full data, not a truncated search slice.
    """
    search_start = max(0, search_start)
    search_end = min(len(data), search_end)
    source = source_buffer
    if source is None:
        source = (
            full_source_buffer(data, len(data))
            if isinstance(data, memoryview)
            else cast(FindableSizedBuffer, data)
        )
    copied_region = bytes(data[search_start:search_end]) if source is None else None
    pos = search_start
    while pos < search_end:
        if source is not None:
            marker = source.find(b"obj", pos, search_end)
        else:
            assert copied_region is not None
            marker = copied_region.find(b"obj", pos - search_start)
        if marker < 0:
            return
        if source is None:
            marker += search_start
        parsed = parse_object_marker_prefix(data, marker)
        if parsed is not None and (allow_prefix_before_start or parsed[0] >= search_start):
            yield parsed
        pos = marker + 3
