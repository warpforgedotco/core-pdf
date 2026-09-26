# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import binascii
import mmap
import re
from collections.abc import Callable
from copy import replace
from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl.types import PdfByteBuffer, PdfName, PdfReference, PdfString
from core_pdf_cythonized import ObjectScanner
from core_pdf_spec.s_07_filters.decode_spec import StreamDecoder
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf_spec.s_07_syntax_primitives.numbers import is_integer_token, parse_integer_token
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    HEX_VALUE,
    FindableSizedBuffer,
    looks_like_indirect_object_header,
    read_literal_string,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import (
    WHITESPACE,
    LexicalRules,
    lexical_rules,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext, recognized_version

RECOVERABLE_DICTIONARY_KEY_NAMES = {
    b"Type",
    b"Subtype",
    b"Pages",
    b"Kids",
    b"Count",
    b"Parent",
    b"MediaBox",
    b"CropBox",
    b"Resources",
    b"Contents",
    b"Length",
    b"Filter",
    b"DecodeParms",
    b"Root",
    b"Size",
    b"Prev",
    b"XRefStm",
    b"Info",
    b"Encrypt",
}


def reader_lexical_rules(rules: LexicalRules) -> LexicalRules:
    return replace(rules, whitespace=WHITESPACE, canonical_identifiers=False)


INDIRECT_HEADER_PATTERNS: dict[LexicalRules, re.Pattern[bytes]] = {}


def indirect_header_pattern(rules: LexicalRules) -> re.Pattern[bytes]:
    """Two digit runs and obj, each ending where scan_word would end it."""
    pattern = INDIRECT_HEADER_PATTERNS.get(rules)
    if pattern is None:
        space = b"[" + re.escape(rules.whitespace) + b"]"
        separator = b"[" + re.escape(rules.whitespace + rules.delimiters) + b"]"
        pattern = INDIRECT_HEADER_PATTERNS[rules] = re.compile(
            b"([0-9]+)" + space + b"+([0-9]+)" + space + b"+obj(?=" + separator + b"|\\Z)"
        )
    return pattern


READER_RULES: dict[LexicalRules, LexicalRules] = {
    rules: reader_lexical_rules(rules)
    for rules in {
        lexical_rules(None),
        *(
            lexical_rules(SemanticContext(PdfVersion(major, minor)))
            for major, minor in ((1, 1), (1, 2), (1, 7), (2, 0))
        ),
    }
}


# The names the object scanner has produced, so a repeated one is a dict hit in
# C rather than a call into PdfName.of. It is filled only through PdfName.of,
# so every entry is the instance that returned, and bounded like spec's own
# table.
SCANNED_NAMES: dict[bytes, PdfName] = {}
SCANNED_NAME_LIMIT = 1 << 16


def scanned_name(raw: bytes) -> PdfName:
    name = PdfName.of(raw)
    if len(SCANNED_NAMES) >= SCANNED_NAME_LIMIT:
        SCANNED_NAMES.clear()
    SCANNED_NAMES[raw] = name
    return name


def drop_unknown_escape(_byte: int) -> bytes:
    return b""


def reader_eol_pair(first: int, second: int) -> bool:
    return (first == 13 and second == 10) or (first == 10 and second == 13)


class PdfLexer(SyntaxLexer):
    __slots__ = (
        "recover_malformed_objects",
        "recover_dictionary_structure",
        "scanner",
        "scanner_rules",
        "copied_data",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        *,
        reference_resolver: Callable[[PdfReference], object] | None = None,
        decipher: Decipher | None = None,
        recover_malformed_objects: bool = True,
        recover_dictionary_structure: bool = True,
        stream_decoder: StreamDecoder | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.scanner: ObjectScanner | None = None
        self.scanner_rules: LexicalRules | None = None
        self.copied_data: bytes | None = None
        super().__init__(
            data,
            reference_resolver=reference_resolver,
            decipher=decipher,
            stream_decoder=decode_stream_data if stream_decoder is None else stream_decoder,
            semantic_context=semantic_context,
        )
        self.recover_malformed_objects = recover_malformed_objects
        self.recover_dictionary_structure = recover_dictionary_structure

    def close(self) -> None:
        # The scanner holds an export of raw_data, and a memoryview with a
        # live export refuses to release.
        if self.scanner is not None:
            self.scanner.release()
            self.scanner = None
        self.copied_data = None
        super().close()

    def search_buffer(self) -> bytes | FindableSizedBuffer:
        """The data as a buffer with find and rfind: the source buffer, or a
        copy of the data made once for a lexer that has none."""
        source_buffer = self.source_buffer
        if source_buffer is not None:
            return source_buffer
        copied = self.copied_data
        if copied is None:
            copied = self.copied_data = self.raw_data.tobytes()
        return copied

    def object_scanner(self) -> ObjectScanner:
        """The compiled scanner for this lexer's data and rules."""
        rules = self.lexical_rules
        scanner = self.scanner
        if scanner is None or self.scanner_rules is not rules:
            scanner = self.scanner = ObjectScanner(
                self.raw_data,
                rules.whitespace_table,
                rules.separator_table,
                rules.name_escapes,
                rules.split_whitespace_compatible,
                SCANNED_NAMES,
                scanned_name,
                PdfString,
                PdfReference,
            )
            self.scanner_rules = rules
        return scanner

    def scanner_args(self) -> tuple[int] | tuple[int, Decipher, int, int]:
        """The scanner's arguments at this position: with the decipher and
        object identity when strings are to be decrypted."""
        if self.decipher is not None and self.current_obj_num is not None:
            return (self.pos, self.decipher, self.current_obj_num, self.current_gen_num or 0)
        return (self.pos,)

    def parse_dictionary(self) -> PdfDict:
        # The scanner owns well-formed syntax and declines the rest, which the
        # Python below then parses from the same position -- including every
        # recovery hook this class overrides.
        parsed = self.object_scanner().parse_dictionary(*self.scanner_args())
        if parsed is None:
            return super().parse_dictionary()
        dictionary, self.pos = parsed
        return dictionary

    def parse_array(self) -> list[Any]:
        parsed = self.object_scanner().parse_array(*self.scanner_args())
        if parsed is None:
            return super().parse_array()
        values, self.pos = parsed
        return values

    def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
        if recognized_version(context) is None:
            context = None
        rules = lexical_rules(context)
        reader_rules = READER_RULES.get(rules)
        return reader_lexical_rules(rules) if reader_rules is None else reader_rules

    def read_string(
        self,
        *,
        drop_unknown_escapes: bool = False,
        unknown_escape: Callable[[int], bytes] | None = None,
        eol_pair: Callable[[int, int], bool] | None = None,
    ) -> bytes:
        source = self.source_buffer
        value, self.pos = read_literal_string(
            source if type(source) is bytes else self.raw_data,
            self.pos,
            self.data_len,
            unknown_escape=drop_unknown_escape if drop_unknown_escapes else unknown_escape,
            eol_pair=reader_eol_pair if eol_pair is None else eol_pair,
        )
        if value is None:
            raise PdfParseError("unterminated string")
        return value

    def handle_invalid_hex_string(self, filtered: bytes) -> bytes:
        if not self.recover_malformed_objects:
            raise PdfParseError("invalid hex string") from None
        recovered = bytes(byte for byte in filtered if HEX_VALUE[byte] != 255)
        if not recovered:
            return b""
        if len(recovered) & 1:
            recovered += b"0"
        return binascii.unhexlify(recovered)

    def scan_value_word(self, position: int) -> tuple[bytes, int] | None:
        scanned = super().scan_value_word(position)
        if scanned is None:
            return None
        raw, end = scanned
        if len(raw) > 6 and raw[-6:] == b"endobj":
            return raw[:-6], end - 6
        return raw, end

    def is_numeric_array_word(self, raw: bytes | memoryview) -> bool:
        return True

    def parse_integer_token(self, token: bytes | memoryview) -> int:
        try:
            return int(token)
        except ValueError:
            if is_integer_token(token):
                return parse_integer_token(token)
            raise

    def parse_real_token(self, token: bytes | memoryview) -> float:
        return float(token)

    def handle_missing_endobj(self, keyword: tuple[bytes, int] | None) -> bool:
        if (
            keyword is not None
            and len(keyword[0]) == 6
            and matches_keyword_with_one_substitution(self.raw_data, keyword[1] - 6, b"endobj")
        ):
            return True
        if keyword is not None and keyword[0] in {b"xref", b"trailer", b"startxref"}:
            return True
        return keyword is not None and looks_like_indirect_object_header(
            self.raw_data,
            keyword[1] - len(keyword[0]),
            self.data_len,
            rules=self.lexical_rules,
            parse_identifier=self.parse_identifier,
        )

    def handle_empty_indirect_object(self) -> object:
        self.advance(6)
        return None

    def handle_invalid_name_escape(
        self, data: bytes, index: int, decoded: int | None, out: bytearray
    ) -> int:
        if decoded is None:
            out.append(35)
            return index + 1
        out.append(decoded)
        return index + 3

    def handle_duplicate_dictionary_key(self, key: PdfName) -> None:
        pass

    def dictionary_end_length(self, pos: int) -> int:
        length = super().dictionary_end_length(pos)
        if length or not (self.recover_malformed_objects and self.recover_dictionary_structure):
            return length
        return 1 if self.raw_data[pos] == 62 and pos + 1 >= self.data_len else 0

    def handle_dictionary_key_error(self) -> bool:
        if not (self.recover_malformed_objects and self.recover_dictionary_structure):
            return False
        data = self.raw_data
        pos = self.pos
        end = min(self.data_len, pos + 256)
        while pos < end:
            byte = data[pos]
            if byte == 47:
                self.pos = pos
                return True
            if byte == 62 and self.dictionary_end_length(pos):
                self.pos = pos
                return True
            if data[pos : pos + 6] == b"endobj":
                return False
            pos += 1
        return False

    def handle_dictionary_entry_error(self, value_start: int) -> bool:
        self.pos = value_start
        if not (self.recover_malformed_objects and self.recover_dictionary_structure):
            return False
        data = self.raw_data
        pos = value_start
        end = min(self.data_len, pos + 512)
        while pos < end:
            byte = data[pos]
            if byte == 62 and self.dictionary_end_length(pos):
                self.pos = pos
                return True
            if data[pos : pos + 6] == b"endobj":
                return False
            if byte == 47:
                match = self.lexical_rules.separator_re.search(data, pos + 1)
                name_end = self.data_len if match is None else match.start()
                name = bytes(data[pos + 1 : name_end])
                if name not in RECOVERABLE_DICTIONARY_KEY_NAMES:
                    pos += 1
                    continue
                value_pos = self.skip_ignored_at(name_end)
                if value_pos < self.data_len:
                    next_byte = data[value_pos]
                    if (
                        next_byte in (40, 47, 60, 91)
                        or next_byte in {45, 43, 46}
                        or 48 <= next_byte <= 57
                        or 65 <= next_byte <= 90
                        or 97 <= next_byte <= 122
                    ):
                        self.pos = pos
                        return True
            pos += 1
        return self.handle_dictionary_key_error()

    def read_stream_eol(self) -> None:
        self.skip_eol()

    def read_stream_data(self, length: object) -> bytes | memoryview:
        data_start = self.pos
        raw_data: bytes | memoryview
        if type(length) is not int or length < 0:
            recovered = self.recover_stream_data(data_start)
            if recovered is None:
                raise PdfParseError("invalid stream length")
            raw_data = recovered
        else:
            raw_data = self.read_bytes(length)
            if len(raw_data) != length:
                recovered = self.recover_stream_data(data_start)
                if recovered is None:
                    raw_data = bytes(raw_data)
                    self.rewind(self.data_len)
                else:
                    raw_data = recovered
            else:
                self.pos = self.skip_ignored_at(self.pos)
                if not self.at_endstream():
                    recovered = self.recover_stream_data(data_start, preferred=self.pos)
                    if recovered is None:
                        self.rewind(self.data_len)
                    else:
                        raw_data = recovered
        if self.at_endstream():
            self.advance(9)

        return raw_data

    def at_endstream(self) -> bool:
        return self.raw_data[
            self.pos : self.pos + 9
        ] == b"endstream" or matches_keyword_with_one_substitution(
            self.raw_data, self.pos, b"endstream"
        )

    def recover_stream_data(
        self, data_start: int, *, preferred: int | None = None
    ) -> bytes | memoryview | None:
        endstream_pos = self.find_stream_end(data_start, preferred=preferred)
        if endstream_pos >= 0:
            self.rewind(endstream_pos)
            return self.raw_data[data_start:endstream_pos]
        endobj_pos = self.find_object_end(data_start)
        if endobj_pos >= 0:
            self.rewind(endobj_pos)
            return self.raw_data[data_start:endobj_pos].tobytes().rstrip(WHITESPACE)
        return None

    def find_keyword_candidate(
        self,
        keyword: bytes,
        start: int,
        end: int,
        *,
        reverse: bool,
        require_eol_before: bool,
        buffer: bytes | FindableSizedBuffer | None = None,
    ) -> tuple[int, int]:
        if buffer is None:
            buffer = self.search_buffer()

        find = buffer.rfind if reverse else buffer.find
        raw_candidate = find(keyword, start, end)
        candidate = raw_candidate
        while candidate >= 0:
            before = candidate - 1
            after = candidate + len(keyword)
            before_ok = (
                before >= 0 and buffer[before] in (10, 13)
                if require_eol_before
                else before < 0 or bool(self.lexical_rules.separator_table[buffer[before]])
            )
            after_ok = after >= self.data_len or bool(
                self.lexical_rules.separator_table[buffer[after]]
            )
            if before_ok and after_ok:
                return candidate, raw_candidate
            if reverse:
                end = candidate
            else:
                start = candidate + 1
            candidate = find(keyword, start, end)
        return -1, raw_candidate

    def find_stream_end(self, data_start: int, preferred: int | None = None) -> int:
        search_buffer = self.search_buffer()
        search_start = data_start if preferred is None else preferred
        candidate, raw_candidate = self.find_keyword_candidate(
            b"endstream",
            search_start,
            self.data_len,
            reverse=False,
            require_eol_before=True,
            buffer=search_buffer,
        )
        if preferred is None:
            return candidate if candidate >= 0 else raw_candidate

        previous, previous_raw = self.find_keyword_candidate(
            b"endstream",
            data_start,
            preferred,
            reverse=True,
            require_eol_before=True,
            buffer=search_buffer,
        )
        if candidate >= 0 and previous >= 0:
            forward_distance = candidate - preferred
            reverse_distance = preferred - previous
            return previous if reverse_distance <= forward_distance else candidate
        if candidate >= 0:
            return candidate
        if previous >= 0:
            return previous
        return raw_candidate if raw_candidate >= 0 else previous_raw

    def find_object_end(self, data_start: int) -> int:
        candidate, raw_candidate = self.find_keyword_candidate(
            b"endobj",
            data_start,
            self.data_len,
            reverse=False,
            require_eol_before=False,
        )
        return candidate if candidate >= 0 else raw_candidate

    def parse_keyword(self, value: memoryview | bytes) -> Any:
        key: bytes = value.tobytes() if type(value) is memoryview else value
        if key == b"true":
            return True
        if key == b"false":
            return False
        if key == b"null":
            return None
        if key == b"R":
            raise PdfParseError("unexpected indirect reference marker")
        return key.decode("latin-1")

    def read_indirect_header(self) -> tuple[int, int]:
        # The usual header -- two digit runs and obj, whitespace between them
        # -- is the three words scan_word would find, so one match reads it;
        # anything else (a comment in between, a sign, obj run into a word)
        # goes through the words as before.
        start = self.skip_ignored_at(self.pos)
        source = self.source_buffer
        header = indirect_header_pattern(self.lexical_rules).match(
            source if type(source) is bytes else self.raw_data, start, self.data_len
        )
        if header is None:
            return super().read_indirect_header()
        identifier = self.parse_identifier(header[1], header[2])
        self.pos = header.end()
        return identifier

    def parse_identifier(self, object_token: bytes, generation_token: bytes) -> tuple[int, int]:
        obj_num = parse_integer_token(object_token)
        gen_num = parse_integer_token(generation_token)
        if obj_num < 0 or not 0 <= gen_num <= 65535:
            raise PdfParseError("invalid indirect object identifier")
        return obj_num, gen_num

    def stream_data_start(self) -> int | None:
        # The keyword may carry one wrong byte. After it, a run of spaces is
        # skipped up to the end of line it leads to; without one, only the
        # first space.
        data = self.raw_data
        pos = self.pos
        if data[pos : pos + 6] != b"stream" and not (
            pos + 6 <= self.data_len and matches_keyword_with_one_substitution(data, pos, b"stream")
        ):
            return None
        start = pos + 6
        if start < self.data_len and data[start] in (0, 9, 12, 32):
            end = start + 1
            while end < self.data_len and data[end] in (0, 9, 12, 32):
                end += 1
            return end if end < self.data_len and data[end] in (10, 13) else start + 1
        return start


def matches_keyword_with_one_substitution(
    data: PdfByteBuffer | memoryview, pos: int, keyword: bytes
) -> bool:
    end = pos + len(keyword)
    if end > len(data):
        return False
    mismatches = 0
    for index, expected in enumerate(keyword):
        if data[pos + index] != expected:
            mismatches += 1
            if mismatches > 1:
                return False
    return mismatches == 1
