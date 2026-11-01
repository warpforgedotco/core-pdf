# Follows pypdf._reader.PdfReader.read, Copyright (c) 2006, Mathieu Fenniak, under its
# BSD-style license.
"""pypdf's walk over a file's cross-reference data.

core-pdf resolves the objects. This walk gives a caller what pypdf's reader promises while
opening a file: the same strict errors, the same warnings for what it tolerates, and the
``xref`` tables it exposes. Nothing here changes how core-pdf reads the file.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from collections.abc import Iterator
from io import BytesIO
from mmap import mmap
from operator import itemgetter
from typing import Any

from core_pdf_compat.pypdf.errors import (
    STREAM_TRUNCATED_PREMATURELY,
    EmptyFileError,
    LimitReachedError,
    PdfReadError,
    PdfStreamError,
)
from core_pdf_compat.pypdf.generic import DictionaryObject, NameObject, to_generic
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.stream import PdfStream

logger = logging.getLogger(__name__)

HEADER_SIZE = 8
STARTXREF_RECOVERY_LINES = 1000
LINE_SEARCH_WINDOW = 4096
TRUNCATED_EOF_MARKERS = (b"%%EO", b"%%E", b"%%", b"%")
WHITESPACE = b"\x00\t\n\f\r "
XREF_STREAM_TRAILER_KEYS = ("/Root", "/Encrypt", "/Info", "/ID", "/Size")
XREF_ENTRY_TYPE_COMPRESSED = 2
FREE_GENERATION = 65535


def previous_line(data: bytes | mmap, position: int) -> tuple[bytes, int]:
    """The line ending at ``position``, and the position past the line breaks above it."""
    if position == 0:
        raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
    # Search back in windows, so a file with no line breaks is not quadratic.
    line_break = -1
    start = end = position
    while start > 0:
        start = max(0, start - LINE_SEARCH_WINDOW)
        line_break = max(data.rfind(b"\n", start, end), data.rfind(b"\r", start, end))
        if line_break >= 0:
            break
        end = start
    line = data[line_break + 1 : position]
    while line_break >= 0 and data[line_break] in b"\r\n":
        line_break -= 1
    return line, line_break + 1


def find_objects(data: bytes) -> Iterator[tuple[int, int, int]]:
    """Each object number, generation and start offset, found by scanning for `` obj``."""
    index = data.find(b" obj")
    while index != -1:
        match = OBJECT_PREFIX.search(data, max(0, index - 32), index)
        if match is not None:
            yield int(match[1]), int(match[2]), match.start(1)
        index = data.find(b" obj", index + 4)


OBJECT_PREFIX = re.compile(rb"(\d+)[\x00\t\n\f\r ]+(\d+)[\x00\t\n\f\r ]*\Z")
OBJECT_HEADER = re.compile(rb"(?:%[^\r\n]*[\r\n]+)?[\x00\t\n\f\r ]*(\d+)[\x00\t\n\f\r ]+\d+")


def _skip_whitespace(stream: BytesIO) -> None:
    while (byte := stream.read(1)) and byte in WHITESPACE:
        pass
    if byte:
        stream.seek(-1, 1)


class XrefScan:
    def __init__(self, data: bytes, strict: bool, pdf: Any) -> None:
        self.data = data
        self.strict = strict
        self.pdf = pdf
        self.stream = BytesIO(data)
        self.lexer = PdfLexer(data)
        self.xref: dict[int, dict[int, int]] = {}
        self.xref_free_entry: dict[int, dict[int, bool]] = {}
        self.xref_objStm: dict[int, tuple[int, int]] = {}  # noqa: N815
        self.xref_index = 0
        self.trailer = DictionaryObject()

    def read(self) -> None:
        self._check_header()
        position = self._find_eof_marker()
        startxref = self._find_startxref(position)
        issue = self._xref_issue(startxref)
        if issue:
            if self.strict:
                raise PdfReadError("Broken xref table")
            logger.warning("incorrect startxref pointer(%d)", issue)
        self._read_tables(startxref, issue)
        if not self.strict:
            if self.xref_index:
                self._correct_shifted_table()
            self._drop_wrong_entries()

    def _object_number(self, offset: int) -> int:
        match = OBJECT_HEADER.match(self.data, offset)
        if match is None:
            raise ValueError(f"no object header at {offset}")
        return int(match[1])

    def _correct_shifted_table(self) -> None:
        for generation, table in self.xref.items():
            if generation == FREE_GENERATION:
                continue
            for number in sorted(table):
                try:
                    found = self._object_number(table[number])
                except ValueError:
                    self._rebuild()
                    return
                # A table that does not start at zero may be off by its first number.
                if found == number - self.xref_index:
                    table[found] = table.pop(number)

    def _drop_wrong_entries(self) -> None:
        for generation, table in self.xref.items():
            if generation == FREE_GENERATION:
                continue
            for number, offset in list(table.items()):
                try:
                    self._object_number(offset)
                except ValueError:
                    logger.warning(
                        "Ignoring wrong pointing object %d %d (offset %d)",
                        number,
                        generation,
                        offset,
                    )
                    del table[number]

    def _check_header(self) -> None:
        header = self.data[:5]
        if header == b"":
            raise EmptyFileError("Cannot read an empty file")
        if header == b"%PDF-":
            return
        if self.strict:
            raise PdfReadError(f"PDF starts with '{header.decode('utf8')}', but '%PDF-' expected")
        logger.warning("invalid pdf header: %r", header)

    def _find_eof_marker(self) -> int:
        position = len(self.data)
        line = b""
        first = True
        while not line.startswith(b"%%EOF"):
            if line and first:
                if line.strip().endswith(TRUNCATED_EOF_MARKERS):
                    logger.warning("EOF marker seems truncated")
                    break
                first = False
            if b"startxref" in line:
                logger.warning(
                    "CAUTION: startxref found while searching for %%EOF. "
                    "The file might be truncated and some data might not be read."
                )
            if position < HEADER_SIZE:
                if self.strict:
                    raise PdfReadError("EOF marker not found")
                logger.warning("EOF marker not found")
            line, position = previous_line(self.data, position)
        return position

    def _find_startxref(self, position: int) -> int:
        line, position = previous_line(self.data, position)
        try:
            return self._integer_then_keyword(int(line), position)
        except ValueError:
            pass
        if not line.startswith(b"startxref"):
            raise PdfReadError("startxref not found")
        startxref = int(line[9:].strip())
        logger.warning("startxref on same line as offset")
        return startxref

    def _integer_then_keyword(self, startxref: int, position: int) -> int:
        line, position = previous_line(self.data, position)
        if line.startswith(b"startxref"):
            return startxref
        # A broken update can leave a corrupt pointer after an intact one. Recovering
        # breaks the standard, so strict mode does not.
        if self.strict:
            raise PdfReadError("startxref not found")
        offset: int | None = None
        for _ in range(STARTXREF_RECOVERY_LINES):
            if position <= 0:
                break
            line, position = previous_line(self.data, position)
            if not line.startswith(b"startxref"):
                try:
                    offset = int(line)
                except ValueError:
                    offset = None
                continue
            if len(line) > 9:
                return int(line[9:].strip())
            if offset is not None:
                logger.warning(
                    "found startxref pointing to a previous revision after a corrupt one"
                )
                return offset
            break
        raise PdfReadError("startxref not found")

    def _xref_issue(self, startxref: int) -> int:
        if startxref == 0:
            return 4
        stream = self.stream
        stream.seek(startxref - 1, 0)
        byte = stream.read(1)
        if byte == b"j":
            byte = stream.read(1)
        if byte not in b"\r\n \t":
            return 1
        if stream.read(4) == b"xref":
            return 0
        byte = b""
        while byte in b"0123456789 \t":
            byte = stream.read(1)
            if byte == b"":
                return 2
        return 0 if (byte + stream.read(2)).lower() == b"obj" else 3

    def _read_tables(self, startxref: int | None, issue: int) -> None:
        stream = self.stream
        visited: set[int] = set()
        while startxref is not None:
            if startxref in visited:
                logger.warning("Circular xref chain detected at offset %d, stopping", startxref)
                break
            visited.add(startxref)
            stream.seek(startxref, 0)
            byte = stream.read(1)
            if byte in b"\r\n":
                byte = stream.read(1)
            if byte == b"x":
                startxref = self._read_table_and_trailer()
            elif issue:
                try:
                    self._rebuild()
                    break
                except PdfReadError, PdfParseError, ValueError:
                    issue = 0
            elif byte.isdigit():
                try:
                    dictionary = self._read_xref_stream()
                except (PdfReadError, LimitReachedError, PdfParseError, ValueError) as error:
                    if "/Root" in self.trailer:
                        logger.warning("Previous trailer cannot be read: %s", error.args)
                        break
                    raise PdfReadError(f"Trailer cannot be read: {error!s}") from error
                self._merge_xref_stream(dictionary)
                if "/Prev" not in dictionary:
                    break
                startxref = dictionary["/Prev"]
            else:
                startxref = self._after_bad_xref_character(startxref)

    def _read_object(self) -> Any:
        self.lexer.rewind(self.stream.tell())
        try:
            value = self.lexer.parse_object()
        except PdfParseError as error:
            raise PdfReadError(str(error)) from error
        self.stream.seek(self.lexer.pos)
        return to_generic(value, self.pdf)

    def _read_table(self) -> None:
        stream = self.stream
        if stream.read(3) != b"ref":
            raise PdfReadError("xref table read error")
        _skip_whitespace(stream)
        first_time = True
        recovered: dict[int, tuple[int, int]] | None = None
        while True:
            number = self._read_object()
            if first_time and number != 0:
                self.xref_index = number
                if self.strict:
                    logger.warning(
                        "Xref table not zero-indexed. ID numbers for objects will be corrected."
                    )
            first_time = False
            _skip_whitespace(stream)
            size = self._read_object()
            if not isinstance(size, int):
                logger.warning("Invalid/Truncated xref table. Rebuilding it.")
                self._rebuild()
                stream.read()
                return
            _skip_whitespace(stream)
            for _ in range(size):
                recovered = self._read_entry(number, recovered)
                number += 1
            _skip_whitespace(stream)
            while stream.read(1) == b"%":
                stream.seek(-1, 1)
                stream.readline()
                _skip_whitespace(stream)
            stream.seek(-1, 1)
            if stream.read(7) == b"trailer":
                return
            stream.seek(-7, 1)

    def _read_entry(
        self, number: int, recovered: dict[int, tuple[int, int]] | None
    ) -> dict[int, tuple[int, int]] | None:
        stream = self.stream
        line = stream.read(20)
        if not line:
            raise PdfReadError("Unexpected empty line in Xref table.")
        # Entries are 20 bytes, but CRLF line ends make some 21.
        while line[0] in b"\r\n":
            stream.seek(-19, 1)
            line = stream.read(20)
        # A single line end with no space before it bleeds into the next entry.
        if line[-1] in b"0123456789t":
            stream.seek(-1, 1)
        try:
            offset_text, generation_text = line[:16].split(b" ")
            kind = line[17:18]
            offset, generation = int(offset_text), int(generation_text)
        except ValueError:
            if recovered is None:
                recovered = {}
                for found, found_generation, start in find_objects(self.data):
                    recovered.setdefault(found, (start, found_generation))
            if number in recovered:
                logger.warning("entry %d in Xref table invalid but object found", number)
                generation, offset = recovered[number]
                kind = b"n"
            else:
                logger.warning("entry %d in Xref table invalid; object not found", number)
                generation, offset, kind = 65535, -1, b"f"
        table = self.xref.setdefault(generation, {})
        free = self.xref_free_entry.setdefault(generation, {})
        if number not in table:  # Read backwards, so the newest table comes first.
            if kind == b"n":
                table[number] = offset
            free[number] = kind == b"f"
        return recovered

    def _read_table_and_trailer(self) -> int | None:
        stream = self.stream
        self._read_table()
        if stream.read(1) == b"":
            return None
        stream.seek(-1, 1)
        _skip_whitespace(stream)
        trailer = self._read_object()
        for key, value in trailer.items():
            if key not in self.trailer:
                self.trailer[key] = value
        if "/XRefStm" in trailer:
            position = stream.tell()
            stream.seek(trailer["/XRefStm"] + 1, 0)
            try:
                self._read_xref_stream()
            except PdfReadError, LimitReachedError, PdfParseError, ValueError:
                logger.warning(
                    "XRef object at %d can not be read, some object may be missing",
                    trailer["/XRefStm"],
                )
            stream.seek(position, 0)
        return trailer.get("/Prev")

    def _after_bad_xref_character(self, startxref: int) -> int | None:
        if startxref == 0:
            if self.strict:
                raise PdfReadError("/Prev=0 in the trailer (try opening with strict=False)")
            logger.warning("/Prev=0 in the trailer - assuming there is no previous xref table")
            return None
        # An off-by-one pointer is common, so look nearby.
        stream = self.stream
        stream.seek(-11, 1)
        nearby = stream.read(20).find(b"xref")
        if nearby != -1:
            return startxref - (10 - nearby)
        stream.seek(startxref, 0)
        for look in range(25):
            if stream.read(1).isdigit():
                return startxref + look
        if "/Root" in self.trailer and not self.strict:
            logger.warning("Invalid parent xref., rebuild xref")
            try:
                self._rebuild()
            except (PdfParseError, ValueError) as error:
                raise PdfReadError("Cannot rebuild xref") from error
            return None
        raise PdfReadError("Could not find xref table at specified location")

    def _read_indirect(self, position: int) -> Any:
        lexer = self.lexer
        lexer.rewind(position)
        lexer.read_indirect_header()
        return lexer.parse_object()

    def _read_xref_stream(self) -> DictionaryObject:
        value = self._read_indirect(self.stream.tell() - 1)
        attributes = value.dictionary if isinstance(value, PdfStream) else value
        dictionary = to_generic(attributes, self.pdf)
        if dictionary["/Type"] != "/XRef":
            raise PdfReadError(f"Unexpected type {dictionary['/Type']!r}")
        if "/Size" not in dictionary:
            raise PdfReadError(f"Size missing from XRef stream {dictionary!r}!")
        widths = dictionary.get("/W")
        if self.strict and len(widths) > 3:
            raise PdfReadError(f"Too many entry sizes: {widths}")
        data = to_generic(value, self.pdf).get_data()
        pairs = self._sanitized_pairs(
            dictionary.get("/Index", [0, dictionary["/Size"]]), widths, len(data)
        )
        self._read_subsections(pairs, widths, data)
        return dictionary

    def _sanitized_pairs(self, pairs: list[int], widths: list[int], length: int) -> list[int]:
        entry_bytes = sum(int(width) for width in widths[:3])
        if entry_bytes == 0:
            message = "Cross-reference stream encodes no entry data."
            if self.strict:
                raise PdfStreamError(message)
            logger.warning(message)
            return []
        maximum = length // entry_bytes + 1
        result: list[int] = []
        total = 0
        for index, value in enumerate(pairs):
            count = int(value)
            if index % 2:
                if total + count > maximum:
                    if self.strict:
                        raise LimitReachedError(
                            f"Total XRef entries {total + count} exceed maximum "
                            f"allowed value {maximum}."
                        )
                    clamped = max(0, maximum - total)
                    logger.warning(
                        "Clamping XRef count from %d to %d to fit stream size.", count, clamped
                    )
                    count = clamped
                total += count
            result.append(count)
        return result

    def _read_subsections(self, pairs: list[int], widths: list[int], data: bytes) -> None:
        position = 0

        def entry(index: int, default: int) -> int:
            nonlocal position
            width = widths[index]
            if not width:
                return default
            value = int.from_bytes(data[position : position + width])
            position += width
            return value

        for index in range(0, len(pairs) - 1, 2):
            for number in range(pairs[index], pairs[index] + pairs[index + 1]):
                kind = entry(0, 1)
                first = entry(1, 0)
                second = entry(2, 0)
                if kind == 1:
                    table = self.xref.setdefault(second, {})
                    if number not in table and number not in self.xref_objStm:
                        table[number] = first
                elif kind == XREF_ENTRY_TYPE_COMPRESSED:
                    if number not in self.xref.get(0, ()) and number not in self.xref_objStm:
                        self.xref_objStm[number] = (first, second)
                elif kind != 0 and self.strict:
                    raise PdfReadError(f"Unknown xref type: {kind}")

    def _merge_xref_stream(self, dictionary: DictionaryObject) -> None:
        for key in XREF_STREAM_TRAILER_KEYS:
            if key in dictionary and key not in self.trailer:
                self.trailer[NameObject(key)] = dictionary.raw_get(key)
        if "/XRefStm" in dictionary:
            position = self.stream.tell()
            self.stream.seek(dictionary["/XRefStm"] + 1, 0)
            self._read_xref_stream()
            self.stream.seek(position, 0)

    def _rebuild(self) -> None:
        self.xref = {}
        objects = sorted(find_objects(self.data), key=itemgetter(2))
        for number, generation, start in objects:
            self.xref.setdefault(generation, {})[number] = start
        logger.warning("parsing for Object Streams")
        trailers = [*self._xref_stream_trailers(objects), *self._literal_trailers()]
        for _, trailer in sorted(trailers, key=itemgetter(0)):
            self.trailer.update(trailer)

    def _xref_stream_trailers(
        self, objects: list[tuple[int, int, int]]
    ) -> Iterator[tuple[int, DictionaryObject]]:
        starts = [start for _, _, start in objects]
        seen: set[int] = set()
        for match in re.finditer(rb"/XRef\b", self.data):
            index = bisect_right(starts, match.start()) - 1
            if index < 0 or starts[index] in seen:
                continue
            seen.add(starts[index])
            try:
                value = to_generic(self._read_indirect(starts[index]), self.pdf)
            except PdfParseError, ValueError:
                continue
            if isinstance(value, DictionaryObject) and value.get("/Type") == "/XRef":
                trailer = DictionaryObject()
                for key in XREF_STREAM_TRAILER_KEYS:
                    if key in value:
                        trailer[NameObject(key)] = value.raw_get(key)
                yield starts[index], trailer

    def _literal_trailers(self) -> Iterator[tuple[int, DictionaryObject]]:
        for match in re.finditer(rb"trailer[\x00\t\n\f\r ]*<<", self.data):
            position = match.end() - 2
            self.lexer.rewind(position)
            yield position, to_generic(self.lexer.parse_object(), self.pdf)
