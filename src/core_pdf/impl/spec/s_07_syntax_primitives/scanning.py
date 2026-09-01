# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from typing import Any, Protocol, cast

from core_pdf.impl.spec.s_07_syntax_primitives.tokens import SEPARATOR_TABLE, WS_TABLE

IS_NUMBER_CHAR = bytes([1 if i in b"+-0123456789." else 0 for i in range(256)])
PDF_IGNORED_RE = re.compile(b"(?:[\x00\t\n\x0c\r ]+|%[^\r\n]*(?:\r\n|\n\r|\r|\n)?)*")
EMPTY_TRANSLATE_TABLE = bytes.maketrans(b"", b"")

STRING_ESCAPE: dict[int, bytes] = {
    110: b"\n",
    114: b"\r",
    116: b"\t",
    98: b"\b",
    102: b"\f",
    40: b"(",
    41: b")",
    92: b"\\",
}

R_SENTINEL = object()
STRING_SPECIAL_TABLE = bytes([1 if i in b"()\\\r\n" else 0 for i in range(256)])
HEX_VALUE = bytes(
    [
        i - 48 if 48 <= i <= 57 else i - 55 if 65 <= i <= 70 else i - 87 if 97 <= i <= 102 else 255
        for i in range(256)
    ]
)


class FindableSizedBuffer(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, key: int | slice, /) -> Any: ...

    def find(self, sub: bytes, start: int = 0, end: int = -1, /) -> int: ...

    def rfind(self, sub: bytes, start: int = 0, end: int = -1, /) -> int: ...


def full_source_bytes(data: bytes | memoryview) -> bytes | None:
    if type(data) is bytes:
        return data
    assert isinstance(data, memoryview)
    source = data.obj
    if (
        type(source) is bytes
        and data.c_contiguous
        and data.itemsize == 1
        and data.nbytes == len(source)
    ):
        return source
    return None


def full_source_buffer(data: memoryview, data_len: int) -> FindableSizedBuffer | None:
    source_bytes = full_source_bytes(data)
    if source_bytes is not None and len(source_bytes) == data_len:
        return cast(FindableSizedBuffer, source_bytes)
    source = data.obj
    if hasattr(source, "find") and hasattr(source, "rfind") and hasattr(source, "__len__"):
        buffer = cast(FindableSizedBuffer, source)
        if (
            data.c_contiguous
            and data.itemsize == 1
            and data.nbytes == data_len
            and len(buffer) == data_len
        ):
            return buffer
    return None


def skip_pdf_ignored(data: bytes | memoryview, position: int, data_len: int) -> int:
    pos = position
    if pos >= data_len:
        return pos
    if isinstance(data, bytes) or data.c_contiguous:
        start = pos
        if data[pos] != 37:
            short_end = min(data_len, pos + 8)
            while pos < short_end and WS_TABLE[data[pos]]:
                pos += 1
            if pos >= data_len:
                return pos
            byte = data[pos]
            if byte != 37 and not WS_TABLE[byte]:
                return pos
            if byte != 37:
                pos = start
        match = PDF_IGNORED_RE.match(data, pos)
        if match is not None:
            return match.end()
    while pos < data_len:
        byte = data[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        if byte != 37:
            break
        pos += 1
        while pos < data_len:
            byte = data[pos]
            if byte == 10 or byte == 13:
                pos += 1
                if pos < data_len:
                    next_byte = data[pos]
                    if (byte == 13 and next_byte == 10) or (byte == 10 and next_byte == 13):
                        pos += 1
                break
            pos += 1
    return pos


def looks_like_indirect_object_header(data: memoryview, position: int, data_len: int) -> bool:
    pos = position
    if pos >= data_len or not (48 <= data[pos] <= 57):
        return False

    while pos < data_len and 48 <= data[pos] <= 57:
        pos += 1
    if pos >= data_len or not WS_TABLE[data[pos]]:
        return False
    while pos < data_len and WS_TABLE[data[pos]]:
        pos += 1

    if pos >= data_len or not (48 <= data[pos] <= 57):
        return False
    while pos < data_len and 48 <= data[pos] <= 57:
        pos += 1
    if pos >= data_len or not WS_TABLE[data[pos]]:
        return False
    while pos < data_len and WS_TABLE[data[pos]]:
        pos += 1

    return pos + 3 <= data_len and data[pos : pos + 3] == b"obj"


def is_digit_bytes(value: memoryview | bytes) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return value.isdigit()


def is_digit_bytes_from(value: memoryview | bytes, pos: int) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return value[pos:].isdigit()


def is_number_word_bytes(value: bytes) -> bool:
    if not value:
        return False
    if value.isdigit():
        return True
    first = value[0]
    if not IS_NUMBER_CHAR[first]:
        return False
    n = len(value)
    if n == 1:
        return 48 <= first <= 57
    if first == 43 or first == 45:
        rest = value[1:]
        if rest.isdigit():
            return True
        start_idx = 1
    else:
        start_idx = 0

    saw_digit = False
    saw_dot = False
    for byte in value[start_idx:]:
        if 48 <= byte <= 57:
            saw_digit = True
        elif byte == 46 and not saw_dot:
            saw_dot = True
        else:
            return False
    return saw_digit


def is_integer_word(value: memoryview | bytes) -> bool:
    if not value:
        return False
    first = value[0]
    if len(value) == 1:
        return 48 <= first <= 57

    if 48 <= first <= 57:
        return is_digit_bytes(value)

    if first not in (43, 45):
        return False

    return is_digit_bytes_from(value, 1)


def matches_keyword_with_one_substitution(data: memoryview, pos: int, keyword: bytes) -> bool:
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


def skip_comment(data: bytes | memoryview, pos: int, data_len: int) -> int:
    if type(data) is bytes:
        lf = data.find(b"\n", pos + 1, data_len)
        cr = data.find(b"\r", pos + 1, data_len)
        if lf < 0:
            return data_len if cr < 0 else cr
        return lf if cr < 0 else min(lf, cr)
    pos += 1
    while pos < data_len and data[pos] not in (10, 13):
        pos += 1
    return pos


def skip_literal_string(data: bytes | memoryview, pos: int, data_len: int) -> int:
    pos += 1
    depth = 1
    if type(data) is bytes:
        while pos < data_len:
            closed = data.find(b")", pos, data_len)
            search_end = data_len if closed < 0 else closed
            escaped = data.find(b"\\", pos, search_end)
            opened = data.find(b"(", pos, search_end)
            marker = (
                min(candidate for candidate in (escaped, opened, closed) if candidate >= 0)
                if escaped >= 0 or opened >= 0 or closed >= 0
                else -1
            )
            if marker < 0:
                return data_len
            byte = data[marker]
            if byte == 92:
                pos = min(marker + 2, data_len)
                continue
            if byte == 40:
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return marker + 1
            pos = marker + 1
        return pos
    while pos < data_len and depth:
        byte = data[pos]
        if byte == 92:
            pos = min(pos + 2, data_len)
            continue
        if byte == 40:
            depth += 1
        elif byte == 41:
            depth -= 1
        pos += 1
    return pos


def skip_hex_string(data: bytes | memoryview, pos: int, data_len: int) -> int:
    marker = data.find(b">", pos + 1, data_len) if type(data) is bytes else -1
    if marker >= 0:
        return marker + 1
    pos += 1
    while pos < data_len:
        if data[pos] == 62:
            return pos + 1
        pos += 1
    return pos


def skip_name(data: bytes | memoryview, pos: int, data_len: int) -> int:
    pos += 1
    while pos < data_len and not SEPARATOR_TABLE[data[pos]]:
        pos += 1
    return pos


__all__ = (
    "EMPTY_TRANSLATE_TABLE",
    "FindableSizedBuffer",
    "HEX_VALUE",
    "R_SENTINEL",
    "STRING_ESCAPE",
    "STRING_SPECIAL_TABLE",
    "full_source_buffer",
    "full_source_bytes",
    "is_integer_word",
    "is_number_word_bytes",
    "looks_like_indirect_object_header",
    "matches_keyword_with_one_substitution",
    "skip_comment",
    "skip_hex_string",
    "skip_literal_string",
    "skip_name",
    "skip_pdf_ignored",
)
