from __future__ import annotations

import binascii
import re
import typing

if typing.TYPE_CHECKING:
    from typing import Any, Iterator

from core_pdf.syntax.errors import PdfParseError
from core_pdf.syntax.primitives import PdfName, PdfReference, PdfStream, PdfString

# Local bindings for hot-path optimization (Zero Cross-Module Call Overhead)
PdfParseError = PdfParseError
PdfName_of = PdfName.of
PdfReference = PdfReference
PdfString = PdfString

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"

SEPARATOR_TABLE = bytes([1 if i in WHITESPACE or i in DELIMITERS else 0 for i in range(256)])

# 256-entry lookup tables - O(1) membership tests, zero allocation per call.
WS_TABLE = bytes([1 if i in (0, 9, 10, 12, 13, 32) else 0 for i in range(256)])
IS_HEX = bytes([1 if chr(i) in "0123456789abcdefABCDEF" else 0 for i in range(256)])
IS_NUMBER_CHAR = bytes([1 if i in b"+-0123456789." else 0 for i in range(256)])

HEX_CHARS = b"0123456789abcdefABCDEF"
NON_HEX_CHARS = bytes(i for i in range(256) if i not in HEX_CHARS)

# Bytearray for delimiter lookup — replaces SEPARATOR_RE.search in dispatch_operations
# to avoid allocating 44M re.Match objects (24GB of allocation churn per run).
_WORD_BREAK = bytearray(256)
for _b in b"()<>[]{}%/":
    _WORD_BREAK[_b] = 1

WHITESPACE_OR_COMMENT_RE = re.compile(rb"([\x00\t\n\x0c\r ]+|%[^\r\n]*[\r\n]?)+")
SEPARATOR_RE = re.compile(rb"[()<>\[\]{}/%\x00\t\n\x0c\r ]")
STRING_SPECIAL_RE = re.compile(rb"[()\\\r]")
_STRING_SPECIAL_TABLE = bytes([1 if i in (40, 41, 92, 13) else 0 for i in range(256)])
NUMBER_RE = re.compile(rb"^[+-]?(?:\d+\.?\d*|\.\d+)$")
INTEGER_RE = re.compile(rb"^[+-]?\d+$")
HEX_DIGIT = bytes(
    [
        i - 48 if 48 <= i <= 57 else i - 55 if 65 <= i <= 70 else i - 87 if 97 <= i <= 102 else 0
        for i in range(256)
    ]
)


# Backslash escape table - allocated once, never rebuilt per call site.
STRING_ESCAPE: dict[int, bytes] = {
    110: b"\n",  # n
    114: b"\r",  # r
    116: b"\t",  # t
    98: b"\b",  # b
    102: b"\f",  # f
    40: b"(",  # (
    41: b")",  # )
    92: b"\\",  # \
}

HEX_END_RE = re.compile(rb">")
ENDSTREAM_RE = re.compile(rb"endstream")

R_SENTINEL = object()
IS_WORD_START = bytes(
    [1 if i > 32 and i not in (40, 41, 60, 62, 91, 93, 47, 123, 125, 37) else 0 for i in range(256)]
)
IS_DIGIT = bytes([1 if 48 <= i <= 57 else 0 for i in range(256)])

HEX_VALUE = bytes(
    [
        i - 48 if 48 <= i <= 57 else i - 55 if 65 <= i <= 70 else i - 87 if 97 <= i <= 102 else 255
        for i in range(256)
    ]
)


class InlineImage:
    """Best-effort inline image payload."""

    __slots__ = ("dictionary", "data")

    dictionary: dict[str, typing.Any]
    data: bytes

    def __init__(self, dictionary: dict[str, Any], data: bytes) -> None:
        object.__setattr__(self, "dictionary", dictionary)
        object.__setattr__(self, "data", data)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"cannot assign to field {name!r}")


class PdfLexer:
    """Byte-oriented PDF cursor and parser helpers."""

    __slots__ = (
        "raw_data",
        "data_len",
        "pos",
        "exhausted",
        "reference_resolver",
        "decipher",
        "current_obj_num",
        "current_gen_num",
        "kw_cache",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        reference_resolver: Any | None = None,
        decipher: Any | None = None,
        kw_cache: dict[bytes, Any] | None = None,
    ) -> None:
        # ZERO-COPY: Accept memoryview directly, never call bytes(data)
        if isinstance(data, memoryview):
            self.raw_data = data
        else:
            self.raw_data = memoryview(data)
        self.data_len = len(self.raw_data)
        self.pos = 0
        self.exhausted = False
        self.reference_resolver = reference_resolver
        self.decipher = decipher
        self.current_obj_num = None
        self.current_gen_num = None

        if kw_cache is not None:
            self.kw_cache = kw_cache
            if b"true" not in self.kw_cache:
                self.kw_cache[b"true"] = True
                self.kw_cache[b"false"] = False
                self.kw_cache[b"null"] = None
                self.kw_cache[b"R"] = R_SENTINEL
        else:
            self.kw_cache = {
                b"true": True,
                b"false": False,
                b"null": None,
                b"R": R_SENTINEL,
            }

    @property
    def data(self) -> memoryview:
        return self.raw_data

    @property
    def position(self) -> int:
        return self.pos

    def rewind(self, position: int = 0) -> None:
        self.pos = max(0, min(position, self.data_len))
        self.exhausted = False
        self.current_obj_num = None
        self.current_gen_num = None

    def advance(self, count: int) -> None:
        self.pos = max(0, min(self.pos + count, self.data_len))

    def read_bytes(self, count: int) -> bytes:
        start = self.pos
        end = min(self.data_len, start + max(0, count))
        self.pos = end
        data = self.raw_data[start:end]
        return bytes(data) if isinstance(data, memoryview) else data

    def skip_ignored(self) -> None:
        data = self.raw_data
        pos = self.pos
        n = self.data_len
        if pos < n:
            byte = data[pos]
            if byte == 32 or byte == 10 or byte == 13 or byte == 9:
                pos += 1
                while pos < n:
                    byte = data[pos]
                    if byte == 32 or byte == 10 or byte == 13 or byte == 9:
                        pos += 1
                    else:
                        break
                self.pos = pos
                if pos < n and byte == 37:  # %
                    match = WHITESPACE_OR_COMMENT_RE.match(data, pos)
                    if match:
                        self.pos = match.end()
            elif byte == 37:
                match = WHITESPACE_OR_COMMENT_RE.match(data, pos)
                if match:
                    self.pos = match.end()

    def skip_ignored_at(self, position: int) -> int:
        data = self.raw_data
        pos = position
        n = self.data_len
        if pos < n:
            byte = data[pos]
            if byte == 32 or byte == 10 or byte == 13 or byte == 9:
                pos += 1
                while pos < n:
                    byte = data[pos]
                    if byte == 32 or byte == 10 or byte == 13 or byte == 9:
                        pos += 1
                    else:
                        break
                if pos < n and byte == 37:
                    match = WHITESPACE_OR_COMMENT_RE.match(data, pos)
                    if match:
                        return pos + match.end()
                return pos
            elif byte == 37:
                match = WHITESPACE_OR_COMMENT_RE.match(data, pos)
                if match:
                    return pos + match.end()
        return pos

    def scan_word_at(self, position: int, skip_ignored: bool = True) -> tuple[memoryview, int] | None:
        data = self.raw_data
        pos = self.skip_ignored_at(position) if skip_ignored else position
        if pos >= self.data_len:
            return None

        byte = data[pos]
        # Fast-path: check if current char is a separator (delimiter or whitespace)
        if SEPARATOR_TABLE[byte]:
            return data[pos : pos + 1], pos + 1

        start = pos
        # Optimization: PDF tokens are typically small. Search only the next 1KB first.
        match = SEPARATOR_RE.search(data, start, start + 1024)
        if match:
            pos = match.start()
        else:
            # Fallback for very long tokens (rare)
            match = SEPARATOR_RE.search(data, start + 1024)
            if match:
                pos = match.start()
            else:
                pos = self.data_len

        return data[start:pos], pos

    def scan_word(self, skip_ignored: bool = True) -> tuple[memoryview, int] | None:
        return self.scan_word_at(self.pos, skip_ignored=skip_ignored)

    @staticmethod
    def is_number_word(value: memoryview | bytes) -> bool:
        if not value:
            return False
        first = value[0]
        if len(value) == 1:
            return 48 <= first <= 57

        if 48 <= first <= 57:
            if bytes(value).isdigit():
                return True
            return bool(NUMBER_RE.match(value))

        if not IS_NUMBER_CHAR[first]:
            return False

        return bool(NUMBER_RE.match(value))

    @staticmethod
    def is_integer_word(value: memoryview | bytes) -> bool:
        if not value:
            return False
        first = value[0]
        if len(value) == 1:
            return 48 <= first <= 57

        if 48 <= first <= 57:
            return bytes(value).isdigit()

        if first not in (43, 45):
            return False

        return bool(INTEGER_RE.match(value))

    @staticmethod
    def parse_number(value: memoryview | bytes) -> int | float:
        try:
            if 46 in value:  # .
                return float(value)
            return int(value)
        except ValueError as exc:
            raise PdfParseError(f"invalid number {bytes(value)!r}") from exc

    def parse_keyword(self, value: memoryview | bytes) -> Any:
        key: bytes = value.tobytes() if type(value) is memoryview else value

        cached = self.kw_cache.get(key)
        if cached is not None:
            if cached is R_SENTINEL:
                raise PdfParseError("unexpected indirect reference marker")
            return cached

        decoded = key.decode("latin-1")
        if len(self.kw_cache) < 1024:  # Prevent unbounded growth
            self.kw_cache[key] = decoded
        return decoded

    def read_string(self) -> bytes:
        data = self.raw_data
        pos = self.pos + 1  # Skip initial '('
        n = self.data_len

        # Fast path: manual byte scan for special chars — avoids regex match object alloc
        spec_table = _STRING_SPECIAL_TABLE
        end_idx = pos
        while end_idx < n and not spec_table[data[end_idx]]:
            end_idx += 1

        if end_idx < n and data[end_idx] == 41:  # )
            # Found end of string without any special characters
            self.pos = end_idx + 1
            raw = data[pos:end_idx]
            return bytes(raw) if isinstance(raw, memoryview) else raw

        # Slow path
        self.pos = pos
        out = bytearray()
        if end_idx > pos:
            out.extend(data[pos:end_idx])
            self.pos = end_idx

        depth = 1
        while self.pos < n:
            byte = data[self.pos]
            self.pos += 1
            if byte == 40:  # (
                depth += 1
                out.append(byte)
            elif byte == 41:  # )
                depth -= 1
                if depth == 0:
                    return bytes(out)
                out.append(byte)
            elif byte == 92:  # \
                if self.pos < n:
                    esc = data[self.pos]
                    self.pos += 1
                    if 48 <= esc <= 55:  # octal
                        oct_val = esc - 48
                        count = 1
                        while count < 3 and self.pos < n and 48 <= data[self.pos] <= 55:
                            oct_val = (oct_val << 3) | (data[self.pos] - 48)
                            self.pos += 1
                            count += 1
                        out.append(oct_val & 0xFF)
                    elif esc == 10:  # \n
                        pass
                    elif esc == 13:  # \r
                        if self.pos < n and data[self.pos] == 10:
                            self.pos += 1
                    else:
                        mapped = STRING_ESCAPE.get(esc)
                        if mapped is None:
                            out.append(esc)
                        else:
                            out.extend(mapped)
            elif byte == 13:  # \r
                out.append(byte)
            else:
                out.append(byte)
        raise PdfParseError("unterminated string")

    def read_hex_string(self) -> bytes:
        self.advance(1)  # Skip '<'
        match = HEX_END_RE.search(self.raw_data, self.pos)
        if not match:
            raise PdfParseError("unterminated hex string")
        marker = match.start()

        raw = self.raw_data[self.pos : marker]
        self.pos = marker + 1

        filtered = bytearray()
        for byte in raw:
            if byte in WHITESPACE:
                continue
            if not IS_HEX[byte]:
                raise PdfParseError("invalid hex string")
            filtered.append(byte)
        if len(filtered) & 1:
            filtered += b"0"
        try:
            return binascii.unhexlify(filtered)
        except binascii.Error:
            raise PdfParseError("invalid hex string")

    def read_name(self) -> memoryview:
        self.advance(1)  # Skip '/'
        match = SEPARATOR_RE.search(self.raw_data, self.pos)
        if match:
            end = match.start()
        else:
            end = self.data_len

        start = self.pos
        self.pos = end
        raw = self.raw_data[start:end]
        data = bytes(raw)
        if b"#" not in data:
            return raw
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            byte = data[i]
            if byte != 35:  # #
                out.append(byte)
                i += 1
                continue
            if i + 2 >= n:
                raise PdfParseError("invalid name escape")
            hi = HEX_VALUE[data[i + 1]]
            lo = HEX_VALUE[data[i + 2]]
            if hi == 255 or lo == 255:
                raise PdfParseError("invalid name escape")
            out.append((hi << 4) | lo)
            i += 3
        return memoryview(bytes(out))

    def apply_decipher(self, value: bytes, dictionary: dict[str, Any] | None = None) -> bytes:
        if self.decipher is None or self.current_obj_num is None:
            return value
        return self.decipher(self.current_obj_num, self.current_gen_num or 0, value, dictionary)

    def parse_number_or_keyword(self, raw: bytes) -> Any:
        if self.is_number_word(raw):
            return self.parse_number(raw)
        return self.parse_keyword(raw)

    def parse_object(self) -> Any:
        self.skip_ignored()
        data = self.raw_data
        pos = self.pos
        if pos >= self.data_len:
            raise PdfParseError("unexpected end of PDF input")

        byte = data[pos]
        if byte == 40:  # (
            return PdfString(self.apply_decipher(self.read_string()))
        if byte == 60:  # <
            if pos + 1 < self.data_len and data[pos + 1] == 60:
                return self.parse_dictionary_or_stream()
            return PdfString(self.apply_decipher(self.read_hex_string()))
        if byte == 91:  # [
            return self.parse_array()
        if byte == 47:  # /
            return PdfName_of(self.read_name())
        if byte == 62 and pos + 1 < self.data_len and data[pos + 1] == 62:
            raise PdfParseError("unexpected dictionary end")
        if byte == 93:
            raise PdfParseError("unexpected array end")

        scanned = self.scan_word_at(pos, skip_ignored=False)
        if scanned is None:
            raise PdfParseError("unexpected end of PDF input")
        raw, end = scanned
        self.pos = end
        if self.is_number_word(raw):
            next_token = self.scan_word_at(end)
            if next_token is not None:
                next_raw, next_end = next_token
                if self.is_integer_word(raw) and self.is_integer_word(next_raw):
                    next_next = self.scan_word_at(next_end)
                    if next_next is not None and next_next[0] == b"R":
                        self.pos = next_next[1]
                        return PdfReference(int(raw), int(next_raw))
            if 46 in raw:  # .
                return float(raw)
            return int(raw)
        return self.parse_keyword(raw)

    def parse_object_at(self, position: int) -> Any:
        self.rewind(position)
        return self.parse_object()

    def parse_indirect_object(self) -> Any:
        scanned = self.scan_word(skip_ignored=True)
        if scanned is None or not self.is_number_word(scanned[0]):
            raise PdfParseError("expected indirect object header")
        raw, end = scanned
        self.pos = end
        obj_num = int(raw)

        gen_num_raw = self.scan_word(skip_ignored=True)
        if gen_num_raw is None or not self.is_number_word(gen_num_raw[0]):
            raise PdfParseError("expected indirect object generation number")
        self.pos = gen_num_raw[1]
        gen_num = int(gen_num_raw[0])
        keyword = self.scan_word(skip_ignored=True)
        if keyword is None or keyword[0] != b"obj":
            raise PdfParseError("expected keyword 'obj'")
        self.pos = keyword[1]
        previous_obj = self.current_obj_num
        previous_gen = self.current_gen_num
        self.current_obj_num = obj_num
        self.current_gen_num = gen_num
        try:
            return self.parse_object()
        finally:
            self.current_obj_num = previous_obj
            self.current_gen_num = previous_gen

    def parse_array(self) -> list[Any]:
        values: list[Any] = []
        self.advance(1)
        while True:
            self.skip_ignored()
            if self.pos >= self.data_len:
                raise PdfParseError("unterminated array")
            if self.raw_data[self.pos] == 93:
                self.advance(1)
                return values
            values.append(self.parse_object())

    def parse_dictionary(self) -> dict[Any, Any]:
        values: dict[Any, Any] = {}
        self.advance(2)
        while True:
            self.skip_ignored()
            if self.pos >= self.data_len:
                raise PdfParseError("unterminated dictionary")
            if (
                self.raw_data[self.pos] == 62
                and self.pos + 1 < self.data_len
                and self.raw_data[self.pos + 1] == 62
            ):
                self.advance(2)
                return values
            if self.raw_data[self.pos] != 47:
                raise PdfParseError("dictionary keys must be names")

            key_bytes = self.read_name()
            # Use PdfName object as key for zero-copy and zero-decoding in loop
            key = PdfName_of(key_bytes)
            values[key] = self.parse_object()

    def parse_stream(self, dictionary: dict[Any, Any]) -> PdfStream:
        from core_pdf.streams.filters import normalize_stream_decode_spec

        self.skip_eol()
        # Lookup using PdfName to avoid decoding
        length = dictionary.get(PdfName_of(b"Length"))
        if isinstance(length, PdfReference):
            if self.reference_resolver is None:
                raise PdfParseError("stream length reference must be resolved by the caller")
            length = self.reference_resolver(length)
        if not isinstance(length, int) or length < 0:
            raise PdfParseError("invalid stream length")

        raw_data = self.read_bytes(length)
        if len(raw_data) != length:
            raise PdfParseError("truncated stream data")
        self.skip_ignored()
        if self.raw_data[self.pos : self.pos + 9] != b"endstream":
            raise PdfParseError("expected endstream")
        self.advance(9)

        raw_data = self.apply_decipher(raw_data, dictionary)
        stream_spec = normalize_stream_decode_spec(dictionary)
        return PdfStream(dictionary, raw_data, stream_spec)

    def parse_dictionary_or_stream(self) -> Any:
        dictionary = self.parse_dictionary()
        self.skip_ignored()
        if self.raw_data[self.pos : self.pos + 6] == b"stream":
            next_pos = self.pos + 6
            if next_pos < self.data_len and self.raw_data[next_pos] not in (10, 13):
                raise PdfParseError("expected end-of-line after stream keyword")
            self.advance(6)
            return self.parse_stream(dictionary)
        return dictionary

    def skip_eol(self) -> None:
        data = self.raw_data
        pos = self.pos
        if pos < self.data_len and data[pos] == 13:  # \r
            pos += 1
        if pos < self.data_len and data[pos] == 10:  # \n
            pos += 1
        self.rewind(pos)

    def skip_inline_image_separator(self) -> bool:
        start = self.pos
        self.skip_eol()
        if self.pos < self.data_len and self.raw_data[self.pos] in WHITESPACE:
            self.advance(1)
        return self.pos > start

    def parse_inline_image(self) -> InlineImage:
        from core_pdf.streams.filters import decode_stream_data, normalize_stream_decode_spec
        from core_pdf.fonts.data.core14 import INLINE_IMAGE_KEY_MAP

        dictionary: dict[Any, Any] = {}
        while True:
            self.skip_ignored()
            if self.pos >= self.data_len:
                raise PdfParseError("unterminated inline image")
            if self.raw_data[self.pos : self.pos + 2] == b"ID":
                self.advance(2)
                break
            if self.raw_data[self.pos] != 47:
                raise PdfParseError("inline image keys must be names")
            key_bytes = self.read_name()
            key = PdfName_of(key_bytes)
            dictionary[key] = self.parse_object()

        if not self.skip_inline_image_separator():
            raise PdfParseError("expected inline image data separator")
        start = self.pos
        data_bytes = bytes(self.raw_data[start:])
        pos = 0
        while True:
            marker = data_bytes.find(b"EI", pos)
            if marker < 0:
                raise PdfParseError("unterminated inline image data")
            after = marker + 2
            prev_ok = marker == 0 or data_bytes[marker - 1] in WHITESPACE
            next_ok = after >= len(data_bytes) or data_bytes[after] in WHITESPACE or data_bytes[after] in b"()<>[]{}/%"
            if prev_ok and next_ok:
                # Convert slice to bytes for rstrip (inline image data is usually small)
                image_data = data_bytes[:marker].rstrip(WHITESPACE)
                self.pos = start + after
                normalized: dict[Any, Any] = {}
                for k, value in dictionary.items():
                    # INLINE_IMAGE_KEY_MAP uses strings, so convert for lookup
                    k_str = str(k)
                    mapped_k = INLINE_IMAGE_KEY_MAP.get(k_str, k_str)
                    normalized[PdfName_of(mapped_k)] = value
                stream_spec = normalize_stream_decode_spec(normalized)
                return InlineImage(normalized, decode_stream_data(image_data, stream_spec))
            pos = marker + 1

    def dispatch_operations(self, op_handlers: Any, fast_op_handlers: Any, depth: int, operands: list[Any] | None = None) -> None:
        if operands is None:
            operands = [None] * 16
        op_count = 0
        raw_data = self.raw_data
        data_len = self.data_len

        # Pre-bind for speed
        parse_keyword = self.parse_keyword
        parse_number = self.parse_number
        is_num_word = self.is_number_word
        word_break = _WORD_BREAK
        is_word_start = IS_WORD_START
        # fast_op_handlers is now a 64K list for O(1) indexing
        op_get = op_handlers.get
        _store_operand = operands.__setitem__

        pos = self.pos
        while pos < data_len:
            byte = raw_data[pos]

            # 1. Inline skip_ignored (common whitespace)
            if byte <= 32:
                pos += 1
                while pos < data_len:
                    if raw_data[pos] > 32:
                        break
                    pos += 1
                if pos >= data_len:
                    break
                byte = raw_data[pos]

            # 2. Check for comment (rare)
            if byte == 37:  # %
                self.pos = pos
                self.skip_ignored()
                pos = self.pos
                if pos >= data_len:
                    break
                byte = raw_data[pos]

            # 3. Standard word (number or operator) - THE HOT PATH
            if is_word_start[byte]:
                # Manual byte scan — avoids 44M re.Match allocations (24GB churn).
                limit = pos + 1024 if pos + 1024 < data_len else data_len
                end = pos
                while end < limit:
                    b = raw_data[end]
                    if b <= 32 or word_break[b]:
                        break
                    end += 1
                if end == limit:
                    self.pos = pos
                    scanned = self.scan_word_at(pos, skip_ignored=False)
                    if scanned is None:
                        break
                    raw, end = scanned

                raw = raw_data[pos:end]
                n_raw = end - pos
                pos = end

                # Fast-path for numbers (highly frequent)
                first = raw[0]
                if 48 <= first <= 57 or first == 45 or first == 43:  # 0-9 or - or +
                    # Inline integer parsing for 1-4 digits
                    if n_raw == 1:
                        if 48 <= first <= 57:
                            _store_operand(op_count, first - 48)
                            op_count += 1
                            continue
                    elif n_raw == 2:
                        b1 = raw[1]
                        if 48 <= first <= 57 and 48 <= b1 <= 57:
                            _store_operand(op_count, (first - 48) * 10 + (b1 - 48))
                            op_count += 1
                            continue
                    elif n_raw == 3:
                        b1, b2 = raw[1], raw[2]
                        if 48 <= first <= 57 and 48 <= b1 <= 57 and 48 <= b2 <= 57:
                            _store_operand(op_count, (first - 48) * 100 + (b1 - 48) * 10 + (b2 - 48))
                            op_count += 1
                            continue
                    elif n_raw == 4:
                        b1, b2, b3 = raw[1], raw[2], raw[3]
                        if 48 <= first <= 57 and 48 <= b1 <= 57 and 48 <= b2 <= 57 and 48 <= b3 <= 57:
                            _store_operand(op_count, (first - 48) * 1000 + (b1 - 48) * 100 + (b2 - 48) * 10 + (b3 - 48))
                            op_count += 1
                            continue

                    # Try general number path
                    if 46 in raw:  # .
                        _store_operand(op_count, float(raw))
                        op_count += 1
                        continue
                    else:
                        # Manual isdigit on memoryview avoids bytes() allocation per call
                        if first in (43, 45):
                            all_digits = True
                            for i in range(1, n_raw):
                                if not (48 <= raw[i] <= 57):
                                    all_digits = False
                                    break
                        else:
                            all_digits = True
                            for i in range(n_raw):
                                if not (48 <= raw[i] <= 57):
                                    all_digits = False
                                    break
                        if all_digits:
                            _store_operand(op_count, int(raw))
                            op_count += 1
                            continue

                    # Fallback
                    if is_num_word(raw):
                        _store_operand(op_count, parse_number(raw))
                        op_count += 1
                        continue

                # Operator or other keyword
                if raw == b"BI":
                    self.pos = pos
                    image = self.parse_inline_image()
                    pos = self.pos
                    _store_operand(op_count, image)
                    op_count += 1
                    handler = op_get("BI")
                    if handler is not None:
                        handler(operands[:op_count], depth)
                    op_count = 0
                    continue

                handler = None
                if n_raw == 1:
                    handler = fast_op_handlers[raw_data[pos - 1] << 8]
                elif n_raw == 2:
                    handler = fast_op_handlers[(raw_data[pos - 2] << 8) | raw_data[pos - 1]]

                if handler is None:
                    op_name = parse_keyword(raw)
                    handler = op_get(op_name)

                if handler is not None:
                    handler(operands[:op_count], depth)
                op_count = 0
                continue

            # 4. Special characters (delimiters)
            pos, op_count = self.dispatch_delimiter(byte, pos, data_len, raw_data, operands, op_count)

        self.pos = pos

    def dispatch_delimiter(
        self, byte: int, pos: int, data_len: int, raw_data: memoryview, operands: list[Any], op_count: int
    ) -> tuple[int, int]:
        self.pos = pos
        if byte == 91:  # [
            operands[op_count] = self.parse_array()
            return self.pos, op_count + 1
        if byte == 60:  # <
            if pos + 1 < data_len and raw_data[pos + 1] == 60:
                operands[op_count] = self.parse_dictionary_or_stream()
            else:
                operands[op_count] = PdfString(self.apply_decipher(self.read_hex_string()))
            return self.pos, op_count + 1
        if byte == 40:  # (
            operands[op_count] = PdfString(self.apply_decipher(self.read_string()))
            return self.pos, op_count + 1
        if byte == 47:  # /
            operands[op_count] = PdfName_of(self.read_name())
            return self.pos, op_count + 1
        if byte == 62:  # >
            if pos + 1 < data_len and raw_data[pos + 1] == 62:
                return pos + 2, op_count
            return pos + 1, op_count
        if byte == 93:  # ]
            return pos + 1, op_count

        # Rare other cases
        return pos + 1, op_count

    def iter_content_operations(self) -> Iterator[tuple[str, tuple[Any, ...]]]:
        # This is now a slow-path for diagnostics/testing, the interpreter uses dispatch_operations.
        results: list[tuple[str, tuple[Any, ...]]] = []

        def collector(o, d, op_name):
            results.append((op_name, tuple(o)))

        class DictAdapter:
            def __init__(self, callback):
                self.callback = callback

            def get(self, k):
                return lambda o, d: self.callback(o, d, k)

        class FastAdapter:
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, k):
                if k is None:
                    return None
                if k > 255:
                    s = chr(k >> 8) + chr(k & 0xFF)
                else:
                    s = chr(k)
                return lambda o, d: self.callback(o, d, s)

        self.dispatch_operations(DictAdapter(collector), FastAdapter(collector), 0)
        yield from results
