# SPDX-License-Identifier: AGPL-3.0-only
"""Lexer for PDF syntax: tokens, objects, strings, and numeric arrays."""

from __future__ import annotations

import binascii
import contextlib
import mmap
import re
from collections.abc import Callable
from typing import Any

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_filters.decode_spec import StreamDecoder
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float
from core_pdf_spec.s_07_syntax_primitives.scanning import (
    EMPTY_TRANSLATE_TABLE,
    HEX_VALUE,
    FindableSizedBuffer,
    full_source_buffer,
    is_integer_word,
    is_number_word_bytes,
    read_literal_string,
    skip_pdf_ignored,
)
from core_pdf_spec.s_07_syntax_primitives.tokens import (
    LexicalRules,
    lexical_rules,
)
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import (
    PdfName,
    PdfReference,
    PdfString,
)

PdfName_of = PdfName.of
HEX_STRING_END_RE = re.compile(b">")
ARRAY_END_RE = re.compile(b"]")


class PdfLexer:
    __slots__ = (
        "raw_data",
        "source_buffer",
        "data_len",
        "pos",
        "reference_resolver",
        "decipher",
        "current_obj_num",
        "current_gen_num",
        "stream_decoder",
        "internal_semantic_context",
        "lexical_rules",
    )

    raw_data: memoryview
    source_buffer: FindableSizedBuffer | None
    data_len: int
    pos: int
    reference_resolver: Callable[[PdfReference], object] | None
    decipher: Decipher | None
    current_obj_num: int | None
    current_gen_num: int | None
    stream_decoder: StreamDecoder | None

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        *,
        reference_resolver: Callable[[PdfReference], object] | None = None,
        decipher: Decipher | None = None,
        stream_decoder: StreamDecoder | None = None,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.semantic_context = semantic_context
        if type(data) is memoryview:
            self.raw_data = (
                memoryview(data)
                if data.ndim == 1 and data.format == "B" and data.c_contiguous
                else memoryview(data.tobytes())
            )
        else:
            self.raw_data = memoryview(data)
        self.data_len = len(self.raw_data)
        self.source_buffer = full_source_buffer(self.raw_data, self.data_len)
        self.pos = 0
        self.reference_resolver = reference_resolver
        self.decipher = decipher
        self.current_obj_num = None
        self.current_gen_num = None
        self.stream_decoder = stream_decoder

    @property
    def semantic_context(self) -> SemanticContext | None:
        return self.internal_semantic_context

    @semantic_context.setter
    def semantic_context(self, context: SemanticContext | None) -> None:
        self.lexical_rules = self.select_lexical_rules(context)
        self.internal_semantic_context = context

    def select_lexical_rules(self, context: SemanticContext | None) -> LexicalRules:
        """Select token rules; readers may override explicit recovery policy.

        Context changes affect subsequent parsing, including dictionary names,
        hexadecimal strings, and numeric-array fast paths. Existing values are
        not rewritten. Calls without context retain the modern grammar.
        """
        return lexical_rules(context)

    def close(self) -> None:
        self.source_buffer = None
        self.reference_resolver = None
        self.decipher = None
        with contextlib.suppress(ValueError):
            self.raw_data.release()
        self.raw_data = memoryview(b"")
        self.data_len = 0
        self.pos = 0

    def rewind(self, position: int = 0) -> None:
        self.pos = max(0, min(position, self.data_len))
        self.current_obj_num = None
        self.current_gen_num = None

    def advance(self, count: int) -> None:
        self.pos += count

    def read_bytes(self, count: int) -> bytes | memoryview:
        start = self.pos
        end = min(self.data_len, start + max(0, count))
        self.pos = end
        return self.raw_data[start:end]

    def skip_ignored(self) -> None:
        self.pos = self.skip_ignored_at(self.pos)

    def skip_ignored_at(self, position: int) -> int:
        data_len = self.data_len
        if position >= data_len:
            return position
        data = self.raw_data
        ws_table = self.lexical_rules.whitespace_table
        pos = position
        byte = data[pos]
        if not ws_table[byte] and byte != 37:
            return pos
        if ws_table[byte]:
            pos += 1
            if pos >= data_len:
                return pos
            byte = data[pos]
            if not ws_table[byte] and byte != 37:
                return pos
        short_end = min(data_len, pos + 8)
        while pos < short_end and ws_table[data[pos]]:
            pos += 1
        if pos >= data_len:
            return pos
        byte = data[pos]
        if byte != 37 and not ws_table[byte]:
            return pos
        # Only whitespace was consumed reaching `pos`, which skip_pdf_ignored
        # would skip again anyway -- resume there rather than making it redo
        # the peek above from `position`.
        return skip_pdf_ignored(data, pos, data_len, rules=self.lexical_rules)

    def scan_word_at(self, position: int, skip_ignored: bool = True) -> tuple[bytes, int] | None:
        data = self.raw_data
        pos = self.skip_ignored_at(position) if skip_ignored else position
        if pos >= self.data_len:
            return None

        byte = data[pos]
        source_buffer = self.source_buffer

        if self.lexical_rules.separator_table[byte]:
            token = (
                source_buffer[pos : pos + 1]
                if source_buffer is not None
                else bytes(data[pos : pos + 1])
            )
            return token, pos + 1

        start = pos
        match = self.lexical_rules.separator_re.search(data, start)
        pos = self.data_len if match is None else match.start()

        token = source_buffer[start:pos] if source_buffer is not None else bytes(data[start:pos])
        return token, pos

    def scan_word(self, skip_ignored: bool = True) -> tuple[bytes, int] | None:
        return self.scan_word_at(self.pos, skip_ignored=skip_ignored)

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
        raise PdfParseError("invalid PDF value keyword")

    def read_string(self) -> bytes:
        source = self.source_buffer
        value, self.pos = read_literal_string(
            source if type(source) is bytes else self.raw_data, self.pos, self.data_len
        )
        if value is None:
            raise PdfParseError("unterminated string")
        return value

    def read_hex_string(self) -> bytes:
        self.advance(1)
        start = self.pos
        source_buffer = self.source_buffer
        marker = -1
        if source_buffer is not None:
            marker = source_buffer.find(b">", start)
        elif self.raw_data.c_contiguous:
            match = HEX_STRING_END_RE.search(self.raw_data, start)
            if match is not None:
                marker = match.start()
        else:
            marker = start
            while marker < self.data_len and self.raw_data[marker] != 62:
                marker += 1
            if marker >= self.data_len:
                marker = -1
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

        filtered = raw.translate(EMPTY_TRANSLATE_TABLE, self.lexical_rules.whitespace)
        if len(filtered) & 1:
            filtered += b"0"
        try:
            return binascii.unhexlify(filtered)
        except binascii.Error:
            raise PdfParseError("invalid hex string") from None

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
                raise PdfParseError("invalid hexadecimal escape in name")
            hi = HEX_VALUE[data[i + 1]]
            lo = HEX_VALUE[data[i + 2]]
            if hi == 255 or lo == 255:
                raise PdfParseError("invalid hexadecimal escape in name")
            decoded = (hi << 4) | lo
            if decoded == 0:
                raise PdfParseError("null byte in PDF name")
            out.append(decoded)
            i += 3
        return memoryview(bytes(out))

    def apply_decipher(self, value: bytes | memoryview, dictionary: PdfDict | None = None) -> bytes:
        if self.decipher is None or self.current_obj_num is None:
            return value.tobytes() if type(value) is memoryview else value
        if type(value) is memoryview:
            value = value.tobytes()
        deciphered = self.decipher(
            self.current_obj_num, self.current_gen_num or 0, value, dictionary
        )
        return deciphered.tobytes() if type(deciphered) is memoryview else deciphered

    def parse_object(self) -> Any:
        self.pos = self.skip_ignored_at(self.pos)
        data = self.raw_data
        pos = self.pos
        if pos >= self.data_len:
            raise PdfParseError("unexpected end of PDF input")

        byte = data[pos]
        match byte:
            case 40:
                value = self.read_string()
                if self.decipher is not None and self.current_obj_num is not None:
                    value = self.apply_decipher(value)
                return PdfString(value, is_literal=True)
            case 60:
                if pos + 1 < self.data_len and data[pos + 1] == 60:
                    return self.parse_dictionary_or_stream()
                value = self.read_hex_string()
                if self.decipher is not None and self.current_obj_num is not None:
                    value = self.apply_decipher(value)
                return PdfString(value, is_literal=False)
            case 91:
                return self.parse_array()
            case 47:
                return PdfName_of(self.read_name())
            case 62 if pos + 1 < self.data_len and data[pos + 1] == 62:
                raise PdfParseError("unexpected dictionary end")
            case 93:
                raise PdfParseError("unexpected array end")

        scanned = self.scan_value_word(pos)
        if scanned is None:
            raise PdfParseError("unexpected end of PDF input")
        raw, end = scanned
        self.pos = end
        if is_number_word_bytes(raw):
            raw_is_integer = 46 not in raw
            if not raw_is_integer:
                return self.parse_real_token(raw)
            next_pos = self.skip_ignored_at(end)
            if next_pos < self.data_len and 48 <= data[next_pos] <= 57:
                next_token = self.scan_word_at(next_pos, skip_ignored=False)
                assert next_token is not None
                next_raw, next_end = next_token
                if is_integer_word(next_raw):
                    reference = self.parse_reference_suffix(raw, next_raw, next_end)
                    if reference is not None:
                        return reference
            return int(raw)
        return self.parse_keyword(raw)

    def parse_object_at(self, position: int) -> Any:
        self.rewind(position)
        return self.parse_object()

    def read_indirect_header(self) -> tuple[int, int]:
        """Consume an object header; leave the cursor after obj on success."""
        scanned = self.scan_word(skip_ignored=True)
        if scanned is None or not is_number_word_bytes(scanned[0]):
            raise PdfParseError("expected indirect object header")
        raw, end = scanned
        self.pos = end
        try:
            obj_num = int(raw)
        except ValueError as exc:
            raise PdfParseError("invalid indirect object header") from exc
        if obj_num <= 0:
            raise PdfParseError("invalid indirect object header")

        gen_num_raw = self.scan_word(skip_ignored=True)
        if gen_num_raw is None or not is_number_word_bytes(gen_num_raw[0]):
            raise PdfParseError("expected indirect object generation number")
        self.pos = gen_num_raw[1]
        try:
            gen_num = int(gen_num_raw[0])
        except ValueError as exc:
            raise PdfParseError("invalid indirect object generation number") from exc
        if not 0 <= gen_num <= 65535:
            raise PdfParseError("invalid indirect object generation number")
        keyword = self.scan_word(skip_ignored=True)
        if keyword is None or keyword[0] != b"obj":
            raise PdfParseError("expected keyword 'obj'")
        self.pos = keyword[1]
        return obj_num, gen_num

    def read_indirect_terminator(self) -> None:
        """Consume endobj, raising at its expected position when absent."""
        self.pos = self.skip_ignored_at(self.pos)
        keyword = self.scan_word(skip_ignored=False)
        if keyword is None or keyword[0] != b"endobj":
            raise PdfParseError("expected keyword 'endobj'")
        self.pos = keyword[1]

    def parse_indirect_object(self) -> Any:
        obj_num, gen_num = self.read_indirect_header()
        previous_obj, previous_gen = self.current_obj_num, self.current_gen_num
        self.current_obj_num, self.current_gen_num = obj_num, gen_num
        try:
            self.pos = self.skip_ignored_at(self.pos)
            if self.raw_data[self.pos : self.pos + 6] == b"endobj":
                raise PdfParseError("missing indirect object value")
            obj = self.parse_object()
        finally:
            self.current_obj_num, self.current_gen_num = previous_obj, previous_gen
        self.read_indirect_terminator()
        return obj

    def parse_numeric_array(self) -> list[int | float] | None:
        start_pos = self.pos
        data = self.raw_data
        source_buffer = self.source_buffer
        raw_data = source_buffer if source_buffer is not None else data
        data_len = self.data_len

        end_array = -1
        if source_buffer is not None:
            end_array = source_buffer.find(b"]", start_pos + 1)
        elif data.c_contiguous:
            match = ARRAY_END_RE.search(data, start_pos + 1)
            if match is not None:
                end_array = match.start()
        if end_array >= 0:
            if end_array == start_pos + 1:
                self.pos = end_array + 1
                return []
            payload = (
                source_buffer[start_pos + 1 : end_array]
                if source_buffer is not None
                else data[start_pos + 1 : end_array].tobytes()
            )
            if b"%" not in payload and b"[" not in payload and b"\v" not in payload:
                tokens = payload.split()
                if tokens and (
                    tokens[-1] == b"R"
                    or tokens[0][0] not in (43, 45, 46)
                    and not 48 <= tokens[0][0] <= 57
                ):
                    return None
                if not all(self.is_numeric_array_word(token) for token in tokens):
                    return None
                try:
                    values: list[int | float] = list(map(int, tokens))
                except ValueError:
                    try:
                        values = [
                            self.parse_real_token(token) if b"." in token else int(token)
                            for token in tokens
                        ]
                    except ValueError:
                        pass
                    else:
                        self.pos = end_array + 1
                        return values
                else:
                    self.pos = end_array + 1
                    return values

        values = []
        pos = start_pos + 1
        ws_table = self.lexical_rules.whitespace_table
        sep_table = self.lexical_rules.separator_table

        while True:
            while pos < data_len:
                byte = raw_data[pos]
                if ws_table[byte]:
                    pos += 1
                    continue
                if byte == 37:
                    pos += 1
                    while pos < data_len and raw_data[pos] not in (10, 13):
                        pos += 1
                    continue
                break
            if pos >= data_len:
                self.pos = start_pos
                return None

            byte = raw_data[pos]
            if byte == 93:
                self.pos = pos + 1
                return values
            if byte not in (43, 45, 46) and not 48 <= byte <= 57:
                self.pos = start_pos
                return None

            has_decimal = byte == 46
            end = pos + 1
            while end < data_len:
                end_byte = raw_data[end]
                if sep_table[end_byte]:
                    break
                if end_byte == 46:
                    has_decimal = True
                end += 1

            raw = raw_data[pos:end]
            if not self.is_numeric_array_word(raw):
                self.pos = start_pos
                return None
            try:
                value = self.parse_real_token(raw) if has_decimal else int(raw)
            except ValueError:
                self.pos = start_pos
                return None
            values.append(value)
            pos = end

    def is_numeric_array_word(self, raw: bytes | memoryview) -> bool:
        """Whether a complete token can enter the numeric-array fast path.

        The default accepts PDF number syntax. Overrides must not change parser
        state; returning False leaves the token to ordinary array parsing.
        """
        return is_number_word_bytes(raw.tobytes() if isinstance(raw, memoryview) else raw)

    def parse_real_token(self, token: bytes | memoryview) -> float:
        """Convert one complete PDF number token without changing parser state.

        Valid syntax outside the implementation's finite real-number range
        raises PdfParseError, as required by ISO 32000-2:2020, Annex C.
        Consumers may override this conversion while reusing lexical traversal.
        """
        value = parse_float(token, default=None)
        if value is not None:
            return value
        raw = token.tobytes() if isinstance(token, memoryview) else token
        if is_number_word_bytes(raw):
            raise PdfParseError("PDF real number exceeds implementation limits")
        raise PdfParseError("invalid PDF real number")

    def parse_reference_suffix(
        self, raw: bytes, next_raw: bytes, next_end: int
    ) -> PdfReference | None:
        """The ``R`` of an ``N G R`` reference, given the two integers already scanned.

        Returns None -- leaving self.pos alone -- when the third token is not
        ``R``, so the caller can fall back to treating ``raw`` as a number.
        """
        next_next = self.scan_word_at(next_end)
        if next_next is None or next_next[0] != b"R":
            return None
        self.pos = next_next[1]
        try:
            obj_num = int(raw)
            gen_num = int(next_raw)
        except ValueError as exc:
            raise PdfParseError("invalid reference") from exc
        if obj_num < 0 or not 0 <= gen_num <= 65535:
            raise PdfParseError("invalid reference")
        return PdfReference(obj_num, gen_num)

    def parse_array(self) -> list[Any]:
        numeric_values = self.parse_numeric_array()
        if numeric_values is not None:
            return numeric_values

        values: list[Any] = []
        self.advance(1)
        data = self.raw_data
        should_decipher = self.decipher is not None and self.current_obj_num is not None
        apply_decipher = self.apply_decipher
        while True:
            self.pos = self.skip_ignored_at(self.pos)
            if self.pos >= self.data_len:
                raise PdfParseError("unterminated array")
            pos = self.pos
            byte = data[pos]
            match byte:
                case 93:
                    self.advance(1)
                    return values
                case 40:
                    value = self.read_string()
                    if should_decipher:
                        value = apply_decipher(value)
                    values.append(PdfString(value, is_literal=True))
                    continue
                case 91:
                    values.append(self.parse_array())
                    continue
                case 60:
                    if pos + 1 < self.data_len and data[pos + 1] == 60:
                        values.append(self.parse_dictionary_or_stream())
                    else:
                        value = self.read_hex_string()
                        if should_decipher:
                            value = apply_decipher(value)
                        values.append(PdfString(value, is_literal=False))
                    continue
                case 47:
                    values.append(PdfName_of(self.read_name()))
                    continue

            scanned = self.scan_word_at(pos, skip_ignored=False)
            if scanned is None:
                raise PdfParseError("unexpected end of PDF input")
            raw, end = scanned
            self.pos = end
            if is_number_word_bytes(raw):
                if 46 in raw:
                    values.append(self.parse_real_token(raw))
                    continue
                next_pos = self.skip_ignored_at(end)
                if next_pos < self.data_len and 48 <= data[next_pos] <= 57:
                    next_token = self.scan_word_at(next_pos, skip_ignored=False)
                    if next_token is not None:
                        next_raw, next_end = next_token
                        if is_integer_word(next_raw):
                            reference = self.parse_reference_suffix(raw, next_raw, next_end)
                            if reference is not None:
                                values.append(reference)
                                continue
                values.append(int(raw))
                continue
            values.append(self.parse_keyword(raw))

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
                raise PdfParseError("dictionary keys must be names")

            key_bytes = self.read_name()

            key = PdfName_of(key_bytes)
            value_start = self.pos
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

    def parse_stream(self, dictionary: PdfDict) -> PdfStream:
        self.read_stream_eol()
        should_decipher = self.decipher is not None and self.current_obj_num is not None
        length: object = dictionary.get("Length")
        if type(length) is PdfReference:
            if self.reference_resolver is None:
                raise PdfParseError("stream length reference must be resolved by the caller")
            length = self.reference_resolver(length)
        raw_data = self.read_stream_data(length)
        if should_decipher:
            raw_data = self.apply_decipher(raw_data, dictionary)
        return PdfStream(dictionary, raw_data, dictionary, decoder=self.stream_decoder)

    def read_stream_eol(self) -> None:
        """Consume the LF or CRLF required after the stream keyword."""
        if self.pos >= self.data_len or self.raw_data[self.pos] not in (10, 13):
            raise PdfParseError("stream keyword must be followed by an end-of-line marker")
        if self.raw_data[self.pos] == 13 and (
            self.pos + 1 >= self.data_len or self.raw_data[self.pos + 1] != 10
        ):
            raise PdfParseError("stream keyword requires LF or CRLF")
        self.skip_eol()

    def read_stream_data(self, length: object) -> bytes | memoryview:
        if type(length) is not int or length < 0:
            raise PdfParseError("invalid stream length")
        raw_data = self.read_bytes(length)
        if len(raw_data) != length:
            raise PdfParseError("truncated stream data")
        self.skip_eol()
        keyword = self.scan_word(skip_ignored=False)
        if keyword is None or keyword[0] != b"endstream":
            raise PdfParseError("expected keyword 'endstream'")
        self.pos = keyword[1]
        return raw_data

    def parse_dictionary_or_stream(self) -> Any:
        dictionary = self.parse_dictionary()
        self.pos = self.skip_ignored_at(self.pos)
        keyword = self.scan_word(skip_ignored=False)
        if keyword is not None and keyword[0] == b"stream":
            self.pos = keyword[1]
            return self.parse_stream(dictionary)
        return dictionary

    def skip_eol(self) -> None:
        data = self.raw_data
        pos = self.pos
        if pos < self.data_len and data[pos] == 13:
            pos += 1
            if pos < self.data_len and data[pos] == 10:
                pos += 1
        elif pos < self.data_len and data[pos] == 10:
            pos += 1

        self.pos = pos

    def scan_value_word(self, position: int) -> tuple[bytes, int] | None:
        """Scan one scalar value token without changing the cursor."""
        return self.scan_word_at(position, skip_ignored=False)


__all__ = ("PdfLexer",)
