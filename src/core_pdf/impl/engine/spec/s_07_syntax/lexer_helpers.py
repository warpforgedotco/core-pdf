# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Protocol, cast, overload

from core_pdf.impl.engine.spec.s_07_syntax.tokens import WS_TABLE

IS_NUMBER_CHAR = bytes([1 if i in b"+-0123456789." else 0 for i in range(256)])
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

    @overload
    def __getitem__(self, key: int, /) -> int: ...

    @overload
    def __getitem__(self, key: slice, /) -> bytes: ...

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
        return source_bytes
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


def number_bytes(value: memoryview | bytes) -> bytes:
    return value.tobytes() if type(value) is memoryview else value


def parse_float_token(value: memoryview | bytes) -> float:
    return float(number_bytes(value))


def parse_int_token(value: memoryview | bytes) -> int:
    return int(number_bytes(value))


def is_digit_bytes(value: memoryview | bytes) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return value.isdigit()


def is_digit_bytes_from(value: memoryview | bytes, pos: int) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return value[pos:].isdigit()


def is_pdf_number_bytes(value: memoryview | bytes) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return is_number_word_bytes(value)


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


def is_number_word(value: memoryview | bytes) -> bool:
    if type(value) is memoryview:
        value = value.tobytes()
    return is_number_word_bytes(value)


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
