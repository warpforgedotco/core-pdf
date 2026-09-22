# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import batched
from typing import Any, ClassVar, Literal, NoReturn, Protocol, Self, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    decoded_name,
    parse_int,
)
from core_pdf_spec.s_07_syntax_primitives.numbers import parse_identifier_tokens
from core_pdf_spec.s_07_syntax_primitives.tokens import lexical_rules
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfByteBuffer

frozen_setattr = object.__setattr__


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


class ParsedXRefSection:
    __slots__ = ("offset", "kind", "entries", "trailer")

    offset: int
    kind: Literal["table", "stream"]
    entries: XRefTable
    trailer: PdfDict

    __fields__: ClassVar[tuple[str, ...]] = ("offset", "kind", "entries", "trailer")
    __match_args__ = ("offset", "kind", "entries", "trailer")

    def __init__(
        self,
        offset: int,
        kind: Literal["table", "stream"],
        entries: XRefTable,
        trailer: PdfDict,
    ) -> None:
        frozen_setattr(self, "offset", offset)
        frozen_setattr(self, "kind", kind)
        frozen_setattr(self, "entries", entries)
        frozen_setattr(self, "trailer", trailer)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"offset={self.offset!r}, "
            f"kind={self.kind!r}, "
            f"entries={self.entries!r}, "
            f"trailer={self.trailer!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.offset == other.offset
            and self.kind == other.kind
            and self.entries == other.entries
            and self.trailer == other.trailer
        )

    def __hash__(self) -> int:
        return hash((self.offset, self.kind, self.entries, self.trailer))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        offset = changes.pop("offset", self.offset)
        kind = changes.pop("kind", self.kind)
        entries = changes.pop("entries", self.entries)
        trailer = changes.pop("trailer", self.trailer)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(offset, kind, entries, trailer)


class XRefRevision:
    __slots__ = ("offset", "entries", "trailer")

    offset: int
    entries: XRefTable
    trailer: PdfDict

    __fields__: ClassVar[tuple[str, ...]] = ("offset", "entries", "trailer")
    __match_args__ = ("offset", "entries", "trailer")

    def __init__(self, offset: int, entries: XRefTable, trailer: PdfDict) -> None:
        frozen_setattr(self, "offset", offset)
        frozen_setattr(self, "entries", entries)
        frozen_setattr(self, "trailer", trailer)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"offset={self.offset!r}, "
            f"entries={self.entries!r}, "
            f"trailer={self.trailer!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.offset == other.offset
            and self.entries == other.entries
            and self.trailer == other.trailer
        )

    def __hash__(self) -> int:
        return hash((self.offset, self.entries, self.trailer))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        offset = changes.pop("offset", self.offset)
        entries = changes.pop("entries", self.entries)
        trailer = changes.pop("trailer", self.trailer)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(offset, entries, trailer)


class XRefSectionReader(Protocol):
    def __call__(self, offset: int, /, *, stream_only: bool = False) -> ParsedXRefSection: ...


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
    ) -> tuple[XRefTable, PdfDict]:
        rules = lexical_rules(semantic_context)
        translation = bytes.maketrans(rules.whitespace, b" " * len(rules.whitespace))
        pos = cls.skip_ws(data, start_pos, semantic_context=semantic_context)
        if data[pos : pos + 4] != b"xref":
            raise PdfParseError("expected xref table")
        pos += 4
        pos = cls.skip_ws(data, pos, semantic_context=semantic_context)

        entries: XRefTable = {}
        object_numbers: set[int] = set()
        max_object_number = -1
        while pos < len(data):
            line, next_pos = cls.read_line(data, pos)
            if line.startswith(b"trailer"):
                pos += len(b"trailer")
                break
            if line.lstrip(rules.whitespace).startswith(b"<<"):
                raise PdfParseError("expected trailer keyword")
            if 11 in line:
                raise PdfParseError("invalid xref table subsection")
            normalized = line.translate(translation)
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
                numbers = {key >> 16 for key in subsection}
                if not object_numbers.isdisjoint(numbers):
                    raise PdfParseError("overlapping xref table subsections")
                object_numbers.update(numbers)
                entries.update(subsection)
                max_object_number = max(max_object_number, maximum)

            else:
                raise PdfParseError("invalid xref table subsection")

        lexer = PdfLexer(data, semantic_context=semantic_context) if lexer is None else lexer
        if semantic_context is not None:
            lexer.semantic_context = semantic_context
        lexer.pos = cls.skip_ws(data, pos, semantic_context=semantic_context)
        try:
            trailer_dict = lexer.parse_dictionary()
        finally:
            lexer.close()
        trailer_dict = cls.validate_trailer_size(trailer_dict, max_object_number)
        return entries, trailer_dict

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
        stream_only: bool = False,
        semantic_context: SemanticContext | None = None,
    ) -> ParsedXRefSection:
        if start < 0 or start >= len(data):
            raise PdfParseError("invalid xref section")
        if data[start : start + 4] == b"xref":
            if stream_only:
                raise PdfParseError("expected xref stream")
            entries, trailer = cls.parse_table_section(
                data, start, semantic_context=semantic_context
            )
            return ParsedXRefSection(start, "table", entries, trailer)
        lexer = PdfLexer(data, semantic_context=semantic_context)
        try:
            lexer.rewind(start)
            obj = lexer.parse_indirect_object()
        finally:
            lexer.close()
        if not isinstance(obj, PdfStream):
            raise PdfParseError("expected xref stream")
        entries, trailer = cls.parse_stream(obj)
        return ParsedXRefSection(start, "stream", entries, trailer)

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
        row_size = validate_xref_widths(w)
        index = dictionary.get("Index")
        if index is None:
            index = [0, size]
        if not isinstance(index, list):
            raise PdfParseError("invalid xref stream Index")
        indices = cast(list[int], index)
        row_count = validate_xref_index(indices, size)
        return (
            decode_xref_row_table(stream.data, w, indices, row_size, row_count),
            cast(PdfDict, dictionary),
        )


def validate_xref_widths(widths: list[int]) -> int:
    if len(widths) != 3 or any(type(width) is not int or width < 0 for width in widths):
        raise PdfParseError("invalid xref stream W")
    if widths[1] == 0:
        raise PdfParseError("invalid xref stream W")
    return sum(widths)


def validate_xref_index(index: list[int], size: int) -> int:
    if type(size) is not int or size <= 0 or len(index) % 2:
        raise PdfParseError("invalid xref stream Index")
    if any(type(value) is not int or value < 0 for value in index):
        raise PdfParseError("invalid xref stream Index")
    previous_start = -1
    previous_end = 0
    row_count = 0
    for start, count in batched(index, 2, strict=True):
        end = start + count
        if start < previous_start or start < previous_end or end > size:
            raise PdfParseError("invalid xref stream Index")
        previous_start, previous_end = start, end
        row_count += count
    return row_count


def decode_xref_row(
    data: bytes, pos: int, widths: list[int], object_number: int
) -> tuple[int, PdfXRefEntry, int]:
    row_size = validate_xref_widths(widths)
    if object_number < 0:
        raise PdfParseError("invalid xref stream Index")
    end = pos + row_size
    if pos < 0 or end > len(data):
        raise PdfParseError("xref stream length mismatch")
    return decode_xref_row_at(data, pos, widths, object_number, row_size)


def decode_xref_row_at(
    data: bytes, pos: int, widths: list[int], object_number: int, row_size: int
) -> tuple[int, PdfXRefEntry, int]:
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
    return key_for(object_number), PdfXRefEntry(0, 0, False), end


def decode_xref_rows(data: bytes, w: list[int], index: list[int], size: int) -> XRefTable:
    row_count = validate_xref_index(index, size)
    row_size = validate_xref_widths(w)
    return decode_xref_row_table(data, w, index, row_size, row_count)


def decode_xref_row_table(
    data: bytes, widths: list[int], index: list[int], row_size: int, row_count: int
) -> XRefTable:
    if len(data) != row_count * row_size:
        raise PdfParseError("xref stream length mismatch")
    entries: XRefTable = {}
    pos = 0
    for start, count in batched(index, 2, strict=True):
        for object_number in range(start, start + count):
            key, entry, pos = decode_xref_row_at(data, pos, widths, object_number, row_size)
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
    rules = lexical_rules(semantic_context)
    ws = rules.whitespace_table
    if marker < 0 or data[marker : marker + 3] != b"obj":
        return None
    if marker + 3 < len(data) and not rules.separator_table[data[marker + 3]]:
        return None
    if marker == 0 or not ws[data[marker - 1]]:
        return None
    pos = marker - 1
    while pos >= 0 and ws[data[pos]]:
        pos -= 1
    gen_end = pos + 1
    while pos >= 0 and not rules.separator_table[data[pos]]:
        pos -= 1
    gen_start = pos + 1
    if pos < 0 or not ws[data[pos]]:
        return None
    while pos >= 0 and ws[data[pos]]:
        pos -= 1
    obj_end = pos + 1
    while pos >= 0 and not rules.separator_table[data[pos]]:
        pos -= 1
    obj_start = pos + 1
    try:
        object_number, generation_number = parse_identifier_tokens(
            bytes(data[obj_start:obj_end]),
            bytes(data[gen_start:gen_end]),
            canonical=rules.canonical_identifiers,
        )
    except PdfParseError:
        return None
    return obj_start, object_number, generation_number


def xref_pointer(trailer: PdfDict, name: str) -> int | None:
    value = trailer.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise PdfParseError(f"invalid xref section /{name}")
    return value


def iter_xref_revisions(start: int, read_section: XRefSectionReader) -> Iterator[XRefRevision]:
    seen: set[int] = set()
    position: int | None = start
    while position is not None:
        if type(position) is not int or position < 0:
            raise PdfParseError("invalid xref section")
        if position in seen:
            raise PdfParseError("xref section loop detected")
        section = read_section(position, stream_only=False)
        if type(section.offset) is not int or section.offset < 0:
            raise PdfParseError("invalid xref section")
        if section.offset in seen:
            raise PdfParseError("xref section loop detected")
        seen.update((position, section.offset))
        previous = xref_pointer(section.trailer, "Prev")
        entries = section.entries
        if section.kind == "table":
            supplemental_offset = xref_pointer(section.trailer, "XRefStm")
            if supplemental_offset is not None:
                supplemental = read_section(supplemental_offset, stream_only=True)
                if supplemental.kind != "stream":
                    raise PdfParseError("expected xref stream")
                entries = dict(supplemental.entries)
                overlay_xref_entries(entries, section.entries)
        yield XRefRevision(section.offset, entries, section.trailer)
        position = previous


def overlay_xref_entries(destination: XRefTable, newer: XRefTable) -> None:
    if not newer or destination is newer:
        return
    replaced = {key >> 16 for key in newer}
    for key in [key for key in destination if key >> 16 in replaced]:
        del destination[key]
    destination.update(newer)


def merge_xref_sections(sections: Iterable[XRefTable]) -> XRefTable:
    merged: XRefTable = {}
    claimed: set[int] = set()
    for section in sections:
        for key, entry in section.items():
            if key >> 16 not in claimed:
                merged[key] = entry
        claimed.update(key >> 16 for key in section)
    return merged


__all__ = (
    "PdfXRefEntry",
    "ParsedXRefSection",
    "XRefRevision",
    "XRefScanner",
    "XRefSectionReader",
    "XRefTable",
    "decode_xref_row",
    "decode_xref_rows",
    "find_eof_marker",
    "key_for",
    "iter_xref_revisions",
    "merge_xref_sections",
    "overlay_xref_entries",
    "parse_object_marker_prefix",
    "parse_xref_entry_at",
    "parse_xref_entry_line",
)
