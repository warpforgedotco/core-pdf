# SPDX-License-Identifier: AGPL-3.0-only
"""Lexer for PDF syntax: tokens, objects, strings, and numeric arrays."""

from __future__ import annotations

import binascii
import contextlib
import mmap
import re
from collections.abc import Callable
from typing import Any

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import (
    PdfName,
    PdfReference,
    PdfString,
)
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import Decipher, PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.scanning import (
    EMPTY_TRANSLATE_TABLE,
    HEX_VALUE,
    FindableSizedBuffer,
    full_source_buffer,
    is_integer_word,
    is_number_word_bytes,
    looks_like_indirect_object_header,
    matches_keyword_with_one_substitution,
    read_literal_string,
    skip_pdf_ignored,
)
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import (
    DELIMITERS,
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
)

PdfName_of = PdfName.of
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

SEPARATOR_RE = re.compile(b"[" + re.escape(WHITESPACE + DELIMITERS) + b"]")
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
        "recover_malformed_objects",
        "recover_dictionary_structure",
    )

    raw_data: memoryview
    source_buffer: FindableSizedBuffer | None
    data_len: int
    pos: int
    reference_resolver: Callable[[PdfReference], object] | None
    decipher: Decipher | None
    current_obj_num: int | None
    current_gen_num: int | None
    recover_malformed_objects: bool
    recover_dictionary_structure: bool

    def __init__(
        self,
        data: bytes | bytearray | memoryview | mmap.mmap,
        *,
        reference_resolver: Callable[[PdfReference], object] | None = None,
        decipher: Decipher | None = None,
        recover_malformed_objects: bool = True,
        recover_dictionary_structure: bool = True,
    ) -> None:

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
        self.recover_malformed_objects = recover_malformed_objects
        self.recover_dictionary_structure = recover_dictionary_structure

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
        pos = position
        byte = data[pos]
        if not WS_TABLE[byte] and byte != 37:
            return pos
        if WS_TABLE[byte]:
            pos += 1
            if pos >= data_len:
                return pos
            byte = data[pos]
            if not WS_TABLE[byte] and byte != 37:
                return pos
        short_end = min(data_len, pos + 8)
        while pos < short_end and WS_TABLE[data[pos]]:
            pos += 1
        if pos >= data_len:
            return pos
        byte = data[pos]
        if byte != 37 and not WS_TABLE[byte]:
            return pos
        # Only whitespace was consumed reaching `pos`, which skip_pdf_ignored
        # would skip again anyway -- resume there rather than making it redo
        # the peek above from `position`.
        return skip_pdf_ignored(data, pos, data_len)

    def scan_word_at(self, position: int, skip_ignored: bool = True) -> tuple[bytes, int] | None:
        data = self.raw_data
        pos = self.skip_ignored_at(position) if skip_ignored else position
        if pos >= self.data_len:
            return None

        byte = data[pos]
        source_buffer = self.source_buffer

        if SEPARATOR_TABLE[byte]:
            token = (
                source_buffer[pos : pos + 1]
                if source_buffer is not None
                else bytes(data[pos : pos + 1])
            )
            return token, pos + 1

        start = pos
        match = SEPARATOR_RE.search(data, start)
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
        return key.decode("latin-1")

    def read_string(self, *, drop_unknown_escapes: bool = False) -> bytes:
        source_buffer = self.source_buffer
        value, self.pos = read_literal_string(
            source_buffer if type(source_buffer) is bytes else self.raw_data,
            self.pos,
            self.data_len,
            drop_unknown_escapes=drop_unknown_escapes,
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

        filtered = raw.translate(EMPTY_TRANSLATE_TABLE, WHITESPACE)
        if len(filtered) & 1:
            filtered += b"0"
        try:
            return binascii.unhexlify(filtered)
        except binascii.Error:
            if not self.recover_malformed_objects:
                raise PdfParseError("invalid hex string") from None
            recovered = bytes(byte for byte in filtered if HEX_VALUE[byte] != 255)
            if not recovered:
                return b""
            if len(recovered) & 1:
                recovered += b"0"
            return binascii.unhexlify(recovered)

    def read_name(self) -> memoryview:
        self.advance(1)
        match = SEPARATOR_RE.search(self.raw_data, self.pos)
        end = self.data_len if match is None else match.start()

        start = self.pos
        self.pos = end
        raw = self.raw_data[start:end]
        if 35 not in raw:
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

        scanned = self.scan_word_at(pos, skip_ignored=False)
        if scanned is None:
            raise PdfParseError("unexpected end of PDF input")
        raw, end = scanned
        self.pos = end
        if len(raw) > 6 and raw[-6:] == b"endobj":
            raw = raw[:-6]
            self.pos = end - 6
        if is_number_word_bytes(raw):
            raw_is_integer = 46 not in raw
            if not raw_is_integer:
                return float(raw)
            next_pos = self.skip_ignored_at(end)
            if next_pos < self.data_len and 48 <= data[next_pos] <= 57:
                next_token = self.scan_word_at(next_pos, skip_ignored=False)
                assert next_token is not None
                next_raw, next_end = next_token
                if is_integer_word(next_raw):
                    reference = self.internal_reference_at(raw, next_raw, next_end)
                    if reference is not None:
                        return reference
            return int(raw)
        return self.parse_keyword(raw)

    def parse_object_at(self, position: int) -> Any:
        self.rewind(position)
        return self.parse_object()

    def parse_indirect_object(self) -> Any:
        scanned = self.scan_word(skip_ignored=True)
        if scanned is None or not is_number_word_bytes(scanned[0]):
            raise PdfParseError("expected indirect object header")
        raw, end = scanned
        self.pos = end
        try:
            obj_num = int(raw)
        except ValueError as exc:
            raise PdfParseError("invalid indirect object header") from exc
        if obj_num < 0:
            raise PdfParseError("invalid indirect object header")

        gen_num_raw = self.scan_word(skip_ignored=True)
        if gen_num_raw is None or not is_number_word_bytes(gen_num_raw[0]):
            raise PdfParseError("expected indirect object generation number")
        self.pos = gen_num_raw[1]
        try:
            gen_num = int(gen_num_raw[0])
        except ValueError as exc:
            raise PdfParseError("invalid indirect object generation number") from exc
        if gen_num < 0:
            raise PdfParseError("invalid indirect object generation number")
        keyword = self.scan_word(skip_ignored=True)
        if keyword is None or keyword[0] != b"obj":
            raise PdfParseError("expected keyword 'obj'")
        self.pos = keyword[1]
        previous_obj = self.current_obj_num
        previous_gen = self.current_gen_num
        self.current_obj_num = obj_num
        self.current_gen_num = gen_num
        try:
            self.pos = self.skip_ignored_at(self.pos)
            if self.raw_data[self.pos : self.pos + 6] == b"endobj":
                self.advance(6)
                return None
            obj = self.parse_object()
        finally:
            self.current_obj_num = previous_obj
            self.current_gen_num = previous_gen
        self.pos = self.skip_ignored_at(self.pos)
        keyword = self.scan_word(skip_ignored=True)
        if keyword is None or keyword[0] != b"endobj":
            if (
                keyword is not None
                and len(keyword[0]) == 6
                and matches_keyword_with_one_substitution(self.raw_data, keyword[1] - 6, b"endobj")
            ):
                return obj
            if keyword is not None and keyword[0] in {
                b"xref",
                b"trailer",
                b"startxref",
            }:
                return obj
            if keyword is not None:
                marker = keyword[1] - len(keyword[0])
                if looks_like_indirect_object_header(self.raw_data, marker, self.data_len):
                    return obj
            raise PdfParseError("expected keyword 'endobj'")
        self.pos = keyword[1]
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
                try:
                    values: list[int | float] = list(map(int, tokens))
                except ValueError:
                    try:
                        values = [float(token) if b"." in token else int(token) for token in tokens]
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
        ws_table = WS_TABLE
        sep_table = SEPARATOR_TABLE

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
            try:
                value = float(raw) if has_decimal else int(raw)
            except ValueError:
                self.pos = start_pos
                return None
            values.append(value)
            pos = end

    def internal_reference_at(
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
        if obj_num < 0 or gen_num < 0:
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
                    values.append(float(raw))
                    continue
                next_pos = self.skip_ignored_at(end)
                if next_pos < self.data_len and 48 <= data[next_pos] <= 57:
                    next_token = self.scan_word_at(next_pos, skip_ignored=False)
                    if next_token is not None:
                        next_raw, next_end = next_token
                        if is_integer_word(next_raw):
                            reference = self.internal_reference_at(raw, next_raw, next_end)
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
                if (
                    self.recover_malformed_objects
                    and self.recover_dictionary_structure
                    and self.recover_dictionary_key_position()
                ):
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
                if (
                    not self.recover_malformed_objects
                    or not self.recover_dictionary_structure
                    or not self.recover_dictionary_entry_position()
                ):
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

    def recover_dictionary_key_position(self) -> bool:
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

    def recover_dictionary_entry_position(self) -> bool:
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
                match = SEPARATOR_RE.search(data, pos + 1)
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
        return self.recover_dictionary_key_position()

    def parse_stream(self, dictionary: PdfDict) -> PdfStream:
        self.skip_eol()
        should_decipher = self.decipher is not None and self.current_obj_num is not None
        length: object = dictionary.get("Length")
        if type(length) is PdfReference:
            if self.reference_resolver is None:
                raise PdfParseError("stream length reference must be resolved by the caller")
            length = self.reference_resolver(length)

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

        if should_decipher:
            raw_data = self.apply_decipher(raw_data, dictionary)
        return PdfStream(dictionary, raw_data, dictionary)

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
                else before < 0 or bool(SEPARATOR_TABLE[buffer[before]])
            )
            after_ok = after >= self.data_len or bool(SEPARATOR_TABLE[buffer[after]])
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
