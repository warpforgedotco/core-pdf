# SPDX-License-Identifier: AGPL-3.0-only
"""Reader recovery layered over the PDF lexical grammar."""

from __future__ import annotations

import binascii
import mmap
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from core_pdf.impl._impl.document.recovery.scanning import (
    matches_keyword_with_one_substitution,
    read_literal_string,
)
from core_pdf.impl._impl.graphics.stream_decoding import decode_stream_data
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_filters.decode_spec import StreamDecoder
from core_pdf_spec.s_07_syntax.lexer import PdfLexer as SyntaxLexer
from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf_spec.s_07_syntax_primitives.numbers import is_integer_token, parse_integer_token
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    EMPTY_TRANSLATE_TABLE,
    HEX_VALUE,
    FindableSizedBuffer,
    looks_like_indirect_object_header,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import (
    WHITESPACE,
    LexicalRules,
    lexical_rules,
)
from core_pdf_spec.standards import SemanticContext

PdfName_of = PdfName.of
HEX_STRING_END_RE = re.compile(b">")

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


def internal_drop_unknown_escape(byte: int) -> bytes:
    return b""


def internal_reader_eol_pair(first: int, second: int) -> bool:
    return (first == 13 and second == 10) or (first == 10 and second == 13)


class PdfLexer(SyntaxLexer):
    __slots__ = ("recover_malformed_objects", "recover_dictionary_structure")

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
        super().__init__(
            data,
            reference_resolver=reference_resolver,
            decipher=decipher,
            stream_decoder=decode_stream_data if stream_decoder is None else stream_decoder,
            semantic_context=semantic_context,
        )
        self.recover_malformed_objects = recover_malformed_objects
        self.recover_dictionary_structure = recover_dictionary_structure

    def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
        # Preserve reader acceptance of NUL whitespace and unknown versions.
        # Valid legacy names still retain a literal # instead of changing identity.
        if context is not None and (context.version is None or not context.version.recognized):
            context = None
        rules = lexical_rules(context)
        return replace(rules, whitespace=WHITESPACE, canonical_identifiers=False)

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
            unknown_escape=internal_drop_unknown_escape if drop_unknown_escapes else unknown_escape,
            eol_pair=internal_reader_eol_pair if eol_pair is None else eol_pair,
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
        # Retain reader acceptance of Python numeric spellings in array fast paths.
        return True

    def parse_integer_token(self, token: bytes | memoryview) -> int:
        # Retain Python spellings in numeric-array recovery, but report limits
        # on otherwise valid PDF integers consistently with ordinary parsing.
        try:
            return int(token)
        except ValueError:
            if is_integer_token(token):
                return parse_integer_token(token)
            raise

    def parse_real_token(self, token: bytes | memoryview) -> float:
        # Keep the reader's existing numeric conversion, including overflow to infinity.
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
        return None

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
            if byte == 62 and pos + 1 < self.data_len and data[pos + 1] == 62:
                self.pos = pos
                return True
            if data[pos : pos + 6] == b"endobj":
                return False
            pos += 1
        return False

    def handle_dictionary_entry_error(self) -> bool:
        if not (self.recover_malformed_objects and self.recover_dictionary_structure):
            return False
        data = self.raw_data
        pos = self.pos
        end = min(self.data_len, pos + 512)
        while pos < end:
            byte = data[pos]
            if byte == 62 and pos + 1 < self.data_len and data[pos + 1] == 62:
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
                        or next_byte == 45
                        or next_byte == 43
                        or next_byte == 46
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
            recovered = self.internal_recover_stream_data(data_start)
            if recovered is None:
                raise PdfParseError("invalid stream length")
            raw_data = recovered
        else:
            raw_data = self.read_bytes(length)
            if len(raw_data) != length:
                recovered = self.internal_recover_stream_data(data_start)
                if recovered is None:
                    raw_data = bytes(raw_data)
                    self.rewind(self.data_len)
                else:
                    raw_data = recovered
            else:
                self.pos = self.skip_ignored_at(self.pos)
                if not self.internal_at_endstream():
                    recovered = self.internal_recover_stream_data(data_start, preferred=self.pos)
                    if recovered is None:
                        self.rewind(self.data_len)
                    else:
                        raw_data = recovered
        if self.internal_at_endstream():
            self.advance(9)

        return raw_data

    def internal_at_endstream(self) -> bool:
        return self.raw_data[
            self.pos : self.pos + 9
        ] == b"endstream" or matches_keyword_with_one_substitution(
            self.raw_data, self.pos, b"endstream"
        )

    def internal_recover_stream_data(
        self, data_start: int, *, preferred: int | None = None
    ) -> bytes | memoryview | None:
        """Delimit stream data by ``endstream``, else ``endobj``; ``None`` if neither exists.

        On success the lexer is positioned at the keyword found; the caller
        consumes ``endstream`` itself, since a stream cut at ``endobj`` has
        no ``endstream`` to skip.
        """
        endstream_pos = self.find_stream_end(data_start, preferred=preferred)
        if endstream_pos >= 0:
            self.rewind(endstream_pos)
            return self.raw_data[data_start:endstream_pos]
        endobj_pos = self.find_object_end(data_start)
        if endobj_pos >= 0:
            self.rewind(endobj_pos)
            return self.raw_data[data_start:endobj_pos].tobytes().rstrip(WHITESPACE)
        return None

    def internal_find_keyword_candidate(
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
            source_buffer = self.source_buffer
            buffer = self.raw_data.tobytes() if source_buffer is None else source_buffer

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
        source_buffer = self.source_buffer
        search_buffer = self.raw_data.tobytes() if source_buffer is None else source_buffer
        search_start = data_start if preferred is None else preferred
        candidate, raw_candidate = self.internal_find_keyword_candidate(
            b"endstream",
            search_start,
            self.data_len,
            reverse=False,
            require_eol_before=True,
            buffer=search_buffer,
        )
        if preferred is None:
            return candidate if candidate >= 0 else raw_candidate

        previous, previous_raw = self.internal_find_keyword_candidate(
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
        candidate, raw_candidate = self.internal_find_keyword_candidate(
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

    def parse_identifier(self, object_token: bytes, generation_token: bytes) -> tuple[int, int]:
        # Reader recovery retains zero, signed and padded object identifiers.
        obj_num = parse_integer_token(object_token)
        gen_num = parse_integer_token(generation_token)
        if obj_num < 0 or not 0 <= gen_num <= 65535:
            raise PdfParseError("invalid indirect object identifier")
        return obj_num, gen_num

    def parse_dictionary_or_stream(self) -> Any:
        dictionary = self.parse_dictionary()
        self.pos = self.skip_ignored_at(self.pos)
        if self.raw_data[self.pos : self.pos + 6] == b"stream" or (
            self.pos + 6 <= self.data_len
            and matches_keyword_with_one_substitution(self.raw_data, self.pos, b"stream")
        ):
            next_pos = self.pos + 6
            if next_pos < self.data_len:
                next_byte = self.raw_data[next_pos]
                if next_byte not in (10, 13):
                    if next_byte in (0, 9, 12, 32):
                        separator_end = next_pos + 1
                        while separator_end < self.data_len and self.raw_data[separator_end] in (
                            0,
                            9,
                            12,
                            32,
                        ):
                            separator_end += 1
                        self.pos = (
                            separator_end
                            if separator_end < self.data_len
                            and self.raw_data[separator_end] in (10, 13)
                            else next_pos + 1
                        )
                        return self.parse_stream(dictionary)
                    if next_byte != 37:
                        self.pos = next_pos
                        return self.parse_stream(dictionary)
            self.pos = next_pos
            return self.parse_stream(dictionary)
        return dictionary

    def read_hex_string(self) -> bytes:
        self.advance(1)
        start = self.pos
        source_buffer = self.source_buffer
        marker = -1
        if source_buffer is not None:
            marker = source_buffer.find(b">", start)
        else:
            match = HEX_STRING_END_RE.search(self.raw_data, start)
            if match is not None:
                marker = match.start()
        if marker < 0:
            raise PdfParseError("unterminated hex string")

        raw = (
            source_buffer[start:marker]
            if source_buffer is not None
            else bytes(self.raw_data[start:marker])
        )
        self.pos = marker + 1

        if not (len(raw) & 1):
            try:
                return binascii.unhexlify(raw)
            except binascii.Error:
                pass

        filtered = raw.translate(EMPTY_TRANSLATE_TABLE, WHITESPACE)
        if len(filtered) & 1:
            filtered += b"0"
        try:
            return binascii.unhexlify(filtered)
        except binascii.Error:
            return self.handle_invalid_hex_string(filtered)

    def read_name(self) -> memoryview:
        self.advance(1)
        match = self.lexical_rules.separator_re.search(self.raw_data, self.pos)
        end = self.data_len if match is None else match.start()

        start = self.pos
        self.pos = end
        raw = self.raw_data[start:end]
        if not self.lexical_rules.name_escapes or 35 not in raw:
            return raw
        data = raw.tobytes()
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            byte = data[i]
            if byte != 35:
                out.append(byte)
                i += 1
                continue
            if i + 2 >= n:
                out.append(byte)
                i += 1
                continue
            hi = HEX_VALUE[data[i + 1]]
            lo = HEX_VALUE[data[i + 2]]
            if hi == 255 or lo == 255:
                out.append(byte)
                i += 1
                continue
            out.append((hi << 4) | lo)
            i += 3
        return memoryview(bytes(out))

    def parse_dictionary(self) -> PdfDict:
        values: PdfDict = {}
        contents_was_parsed_without_decipher = False
        # The signature-Contents carve-out below is the only reason to inspect a
        # value before parsing it, and it cannot apply when nothing is being
        # deciphered. Deciding that once keeps the probe -- a skip_ignored_at the
        # following parse_object immediately repeats, two buffer reads, and a
        # PdfName-to-str compare -- off every key of every dictionary.
        deciphering = self.decipher is not None and self.current_obj_num is not None
        self.advance(2)
        while True:
            self.pos = self.skip_ignored_at(self.pos)
            if self.pos >= self.data_len:
                raise PdfParseError("unterminated dictionary")
            if (
                self.raw_data[self.pos] == 62
                and self.pos + 1 < self.data_len
                and self.raw_data[self.pos + 1] == 62
            ):
                self.advance(2)
                break
            if self.raw_data[self.pos] != 47:
                if self.handle_dictionary_key_error():
                    if self.pos >= self.data_len:
                        raise PdfParseError("unterminated dictionary")
                    if (
                        self.raw_data[self.pos] == 62
                        and self.pos + 1 < self.data_len
                        and self.raw_data[self.pos + 1] == 62
                    ):
                        self.advance(2)
                        break
                if self.raw_data[self.pos] != 47:
                    raise PdfParseError("dictionary keys must be names")

            key_bytes = self.read_name()

            key = PdfName_of(key_bytes)
            value_start = self.pos
            try:
                if deciphering and key == "Contents":
                    value_pos = self.skip_ignored_at(value_start)
                    if (
                        value_pos < self.data_len
                        and self.raw_data[value_pos] == 60
                        and (value_pos + 1 >= self.data_len or self.raw_data[value_pos + 1] != 60)
                    ):
                        # ISO 32000-2:2020, 7.6.2 excludes a Signature
                        # dictionary's hexadecimal Contents value from
                        # encryption. Parse it raw until the whole dictionary
                        # identifies its type; Type may follow Contents and
                        # defaults to Sig under Table 255.
                        self.pos = value_pos
                        values[key] = PdfString(self.read_hex_string(), is_literal=False)
                        contents_was_parsed_without_decipher = True
                    else:
                        values[key] = self.parse_object()
                        contents_was_parsed_without_decipher = False
                else:
                    values[key] = self.parse_object()
            except PdfParseError:
                self.pos = value_start
                if not self.handle_dictionary_entry_error():
                    raise

        if contents_was_parsed_without_decipher:
            signature_type = values.get("Type")
            is_signature_dictionary = signature_type in ("Sig", "DocTimeStamp") or (
                signature_type is None and "ByteRange" in values
            )
            raw_contents = values.get("Contents")
            if not is_signature_dictionary and type(raw_contents) is PdfString:
                values["Contents"] = PdfString(
                    self.apply_decipher(raw_contents.data),
                    is_literal=raw_contents.is_literal,
                )
        return values

    def parse_indirect_object(self, *, expected_reference: PdfReference | None = None) -> Any:
        obj_num, gen_num = self.read_indirect_header()
        if expected_reference is not None and (obj_num, gen_num) != (
            expected_reference.object_number,
            expected_reference.generation_number,
        ):
            raise PdfParseError("indirect object header does not match expected reference")
        previous_obj = self.current_obj_num
        previous_gen = self.current_gen_num
        self.current_obj_num = obj_num
        self.current_gen_num = gen_num
        try:
            self.pos = self.skip_ignored_at(self.pos)
            if self.raw_data[self.pos : self.pos + 6] == b"endobj":
                self.advance(6)
                return self.handle_empty_indirect_object()
            obj = self.parse_object()
        finally:
            self.current_obj_num = previous_obj
            self.current_gen_num = previous_gen
        self.pos = self.skip_ignored_at(self.pos)
        keyword = self.scan_word(skip_ignored=True)
        if keyword is None or keyword[0] != b"endobj":
            if self.handle_missing_endobj(keyword):
                return obj
            raise PdfParseError("expected keyword 'endobj'")
        self.pos = keyword[1]
        return obj
