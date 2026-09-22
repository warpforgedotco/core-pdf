# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
import zlib
from collections.abc import Iterator
from typing import cast

from core_pdf.impl.document.recovery.lexer import PdfLexer, matches_keyword_with_one_substitution
from core_pdf.impl.document.recovery.objects import PdfObjectStream
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import PdfByteBuffer
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax.xref import (
    ParsedXRefSection,
    PdfXRefEntry,
    XRefTable,
    decode_xref_row,
    key_for,
)
from core_pdf_spec.s_07_syntax.xref import XRefScanner as SyntaxXRefScanner
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_int_strict,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    FindableSizedBuffer,
    full_source_buffer,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import WS_TABLE
from core_pdf_spec.standards import SemanticContext


def parse_object_marker_prefix(
    data: PdfByteBuffer | memoryview,
    marker: int,
    *,
    semantic_context: SemanticContext | None = None,
) -> tuple[int, int, int] | None:
    if marker < 0 or data[marker : marker + 3] != b"obj":
        return None
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
    if obj_start == obj_end or (pos >= 0 and not WS_TABLE[data[pos]]):
        return None
    try:
        object_number = int(data[obj_start:obj_end])
        generation = int(data[gen_start:gen_end])
    except ValueError:
        return None
    if generation > 65535:
        return None
    return obj_start, object_number, generation


def validate_xref_numbers(offset: int, generation: int) -> None:
    if offset < 0:
        raise PdfParseError("invalid xref table entry")
    if not 0 <= generation <= 65535:
        raise PdfParseError("invalid xref generation number")


def parse_xref_entry_line(line: bytes) -> tuple[int, int, bool]:
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
    validate_xref_numbers(offset, generation)
    if len(parts) == 2:
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
                validate_xref_numbers(offset, generation)
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
    def skip_ws(
        data: PdfByteBuffer, pos: int, *, semantic_context: SemanticContext | None = None
    ) -> int:
        return SyntaxXRefScanner.skip_ws(data, pos)

    @staticmethod
    def skip_ignored(
        data: PdfByteBuffer,
        pos: int,
        stop: int | None = None,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> int:
        return SyntaxXRefScanner.skip_ignored(data, pos, stop)

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
    def parse_subsection_integer(token: bytes) -> int:
        return int(token)

    @staticmethod
    def recover_object_stream_entries(
        entries: XRefTable,
        parsed_streams: dict[int, tuple[int, PdfStream]],
        max_entries: int = 100000,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        for key, entry in list(entries.items()):
            if len(entries) >= max_entries:
                return
            if not entry.in_use or entry.object_stream is not None or entry.offset < 0:
                continue
            obj_num = key >> 16
            parsed = parsed_streams.get(key)
            if parsed is None or parsed[0] != entry.offset:
                continue
            obj = parsed[1]
            if not isinstance(obj, PdfStream):
                continue
            dictionary = obj.dictionary
            type_name = recover_pdf_name(dictionary.get("Type"))
            if type_name != "ObjStm" and (
                dictionary.get("N") is None or dictionary.get("First") is None
            ):
                continue
            try:
                container = PdfObjectStream(obj, semantic_context=semantic_context)
            except Exception:
                continue
            for embedded_index, embedded_num in enumerate(container.index):
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
                    index_in_stream=embedded_index,
                )

    @staticmethod
    def find_startxref(
        data: PdfByteBuffer, *, semantic_context: SemanticContext | None = None
    ) -> int | None:
        eof_pos = find_eof_marker(data, semantic_context=semantic_context)
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

            pos = XRefScanner.skip_ws(data, marker + 9, semantic_context=semantic_context)
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
                semantic_context=semantic_context,
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

        for candidate in XRefScanner.find_nearby_sections(
            data, eof_pos, window=len(data), semantic_context=semantic_context
        ):
            if candidate >= eof_pos:
                continue
            try:
                XRefScanner.parse_section_at(data, candidate, semantic_context=semantic_context)
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
        stream_only: bool = False,
        recover_malformed_objects: bool = True,
        semantic_context: SemanticContext | None = None,
    ) -> ParsedXRefSection:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        pos = XRefScanner.skip_ws(data, start, semantic_context=semantic_context)

        if data[pos : pos + 4] == b"xref":
            if stream_only:
                raise PdfParseError("expected xref stream")
            entries, trailer = cls.parse_table_section(
                data,
                pos,
                recover_malformed_objects=recover_malformed_objects,
                semantic_context=semantic_context,
            )
            return ParsedXRefSection(pos, "table", entries, trailer)

        lexer = PdfLexer(
            data,
            recover_malformed_objects=recover_malformed_objects,
            semantic_context=semantic_context,
        )
        lexer.pos = pos
        try:
            try:
                obj = lexer.parse_indirect_object()
            except PdfParseError:
                obj = cls.parse_xref_stream_salvage(
                    data,
                    pos,
                    recover_malformed_objects=recover_malformed_objects,
                    semantic_context=semantic_context,
                )
                if obj is None:
                    raise
        finally:
            lexer.close()
        if not isinstance(obj, PdfStream):
            raise PdfParseError("expected xref stream")
        try:
            entries, trailer = cls.parse_stream(obj)
        except PdfParseError:
            obj = cls.parse_xref_stream_salvage(
                data,
                pos,
                recover_malformed_objects=recover_malformed_objects,
                semantic_context=semantic_context,
            )
            if obj is None:
                raise
            entries, trailer = cls.parse_stream(obj)
        return ParsedXRefSection(pos, "stream", entries, trailer)

    @staticmethod
    def parse_xref_stream_salvage(
        data: PdfByteBuffer,
        pos: int,
        *,
        recover_malformed_objects: bool = True,
        semantic_context: SemanticContext | None = None,
    ) -> PdfStream | None:
        lexer = PdfLexer(
            data,
            recover_malformed_objects=recover_malformed_objects,
            semantic_context=semantic_context,
        )
        header_marker = data.find(b"obj", pos, min(len(data), pos + 64))
        if header_marker < 0:
            return None
        parsed_header = parse_object_marker_prefix(
            data, header_marker, semantic_context=semantic_context
        )
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
        if recover_pdf_name(dict_obj.get("Type")) != "XRef":
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
        filter_name = recover_pdf_name(dict_obj.get("Filter"))
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
                row_size = sum(item for item in w if type(item) is int)
                row_count = sum(
                    cast(int, index[i + 1])
                    for i in range(0, len(index) - 1, 2)
                    if type(index[i + 1]) is int
                )
                if len(decoded_data) != row_size * row_count:
                    decoded_data = None
        return PdfStream(
            dict_obj,
            raw_data if decoded_data is None else decoded_data,
            None,
            decoder=decode_stream_data,
        )

    @staticmethod
    def find_nearby_sections(
        data: PdfByteBuffer,
        start: int,
        window: int = 1024,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> list[int]:
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
                object_marker = find_previous_object_marker(
                    data, type_pos, semantic_context=semantic_context
                )
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
        semantic_context: SemanticContext | None = None,
    ) -> XRefTable:
        entries: XRefTable = {}
        parsed_streams: dict[int, tuple[int, PdfStream]] = {}
        lexer = PdfLexer(data, semantic_context=semantic_context)
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
            parsed_header = parse_object_marker_prefix(
                data, marker, semantic_context=semantic_context
            )
            if parsed_header is None:
                continue
            offset, obj_num, gen_num = parsed_header
            lexer.rewind(offset)
            try:
                obj = lexer.parse_indirect_object()
            except Exception:
                if stop_at_first_trailer:
                    stream_marker = data.find(b"stream", offset, scan_end)
                    next_object_marker = data.find(b"obj", marker + 3, scan_end)
                    if stream_marker >= 0 and (
                        next_object_marker < 0 or stream_marker < next_object_marker
                    ):
                        prefix = data[marker + 3 : stream_marker]
                        uncommented = b"\n".join(
                            line.split(b"%", 1)[0] for line in prefix.splitlines()
                        )
                        if not uncommented.strip():
                            break
                continue
            if stop_at_first_trailer and not isinstance(obj, PdfStream):
                stream_marker = data.find(b"stream", offset, scan_end)
                next_object_marker = data.find(b"obj", marker + 3, scan_end)
                if stream_marker >= 0 and (
                    next_object_marker < 0 or stream_marker < next_object_marker
                ):
                    prefix = data[marker + 3 : stream_marker]
                    uncommented = b"\n".join(line.split(b"%", 1)[0] for line in prefix.splitlines())
                    endstream = data.find(b"endstream", stream_marker + 6, scan_end)
                    if not uncommented.strip() and endstream < 0:
                        break
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
        XRefScanner.recover_object_stream_entries(
            entries, parsed_streams, max_entries, semantic_context=semantic_context
        )
        return entries

    @classmethod
    def read_subsection(
        cls, data: PdfByteBuffer, pos: int, start_obj: int, num_objs: int
    ) -> tuple[XRefTable, int, int]:
        entries: XRefTable = {}
        max_object_number = start_obj + num_objs - 1
        actual_count = 0
        for i in range(num_objs):
            entry_pos = cls.skip_ws(data, pos)
            if data[entry_pos : entry_pos + 7].startswith((b"trailer", b"<<")):
                pos = entry_pos
                break
            offset, generation, in_use, pos = parse_xref_entry_at(data, pos)
            entries[((start_obj + i) << 16) | generation] = PdfXRefEntry(offset, generation, in_use)
            actual_count += 1
        while True:
            entry_pos = cls.skip_ws(data, pos)
            if data[entry_pos : entry_pos + 7].startswith((b"trailer", b"<<")):
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
    def validate_trailer_size(cls, trailer: PdfDict, maximum: int) -> PdfDict:
        size = trailer.get("Size")
        if type(size) is not int or size <= maximum or size <= 0:
            trailer = dict(trailer)
            trailer["Size"] = max(maximum + 1, 1)
        return trailer

    @classmethod
    def parse_table_section(
        cls,
        data: PdfByteBuffer,
        start_pos: int,
        *,
        recover_malformed_objects: bool = True,
        lexer: SyntaxLexer | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> tuple[XRefTable, PdfDict]:
        pos = cls.skip_ws(data, start_pos)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = cls.skip_ws(data, pos)

        entries: XRefTable = {}
        max_object_number = -1
        while pos < len(data):
            line, next_pos = cls.read_line(data, pos)
            if line.startswith(b"trailer"):
                pos += len(b"trailer")
                break
            if line.lstrip().startswith(b"<<"):
                break
            if 11 in line:
                raise PdfParseError("invalid xref table subsection")
            parts = line.replace(b"\x00", b" ").strip().split()
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

        lexer = (
            PdfLexer(
                data,
                recover_malformed_objects=recover_malformed_objects,
                semantic_context=semantic_context,
            )
            if lexer is None
            else lexer
        )
        if semantic_context is not None:
            lexer.semantic_context = semantic_context
        lexer.pos = cls.skip_ws(data, pos)
        try:
            trailer_dict = lexer.parse_dictionary()
        finally:
            lexer.close()
        trailer_dict = cls.validate_trailer_size(trailer_dict, max_object_number)
        return entries, trailer_dict

    @classmethod
    def recover_section_at(
        cls,
        data: PdfByteBuffer,
        start: int,
        *,
        stream_only: bool = False,
        recover_malformed_objects: bool = True,
        semantic_context: SemanticContext | None = None,
    ) -> ParsedXRefSection:
        try:
            return cls.parse_section_at(
                data,
                start,
                stream_only=stream_only,
                recover_malformed_objects=recover_malformed_objects,
                semantic_context=semantic_context,
            )
        except PdfParseError as original_error:
            for nearby in cls.find_nearby_sections(data, start, semantic_context=semantic_context):
                if nearby == start:
                    continue
                try:
                    return cls.parse_section_at(
                        data,
                        nearby,
                        stream_only=stream_only,
                        recover_malformed_objects=recover_malformed_objects,
                        semantic_context=semantic_context,
                    )
                except PdfParseError:
                    continue
            raise original_error

    @staticmethod
    def parse_stream(stream: PdfStream) -> tuple[XRefTable, PdfDict]:
        dict_obj = stream.dictionary
        type_value = dict_obj.get("Type")
        type_name = recover_pdf_name(type_value)
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
        entries: XRefTable = {}
        pos = 0
        for i in range(0, len(available_index), 2):
            start, count = available_index[i : i + 2]
            for object_number in range(start, start + count):
                row_pos = pos
                pos += row_size
                if object_number >= effective_size:
                    continue
                try:
                    if w[1] == 0:
                        split = row_pos + w[0]
                        row = data[row_pos:split] + b"\x00" + data[split:pos]
                        key, entry, _ = decode_xref_row(row, 0, [w[0], 1, w[2]], object_number)
                    else:
                        key, entry, _ = decode_xref_row(data, row_pos, w, object_number)
                except PdfParseError:
                    continue
                entries[key] = entry
        return entries, dict_obj


def find_eof_marker(data: PdfByteBuffer, *, semantic_context: SemanticContext | None = None) -> int:
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


def find_previous_object_marker(
    data: PdfByteBuffer, before: int, *, semantic_context: SemanticContext | None = None
) -> int | None:
    search_end = min(before, len(data))
    while True:
        marker = data.rfind(b"obj", 0, search_end)
        if marker < 0:
            return None
        parsed = parse_object_marker_prefix(data, marker, semantic_context=semantic_context)
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
    semantic_context: SemanticContext | None = None,
) -> Iterator[tuple[int, int, int]]:
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
        parsed = parse_object_marker_prefix(data, marker, semantic_context=semantic_context)
        if parsed is not None and (allow_prefix_before_start or parsed[0] >= search_start):
            yield parsed
        pos = marker + 3
