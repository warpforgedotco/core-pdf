# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast


def full_source_bytes(data: bytes | memoryview) -> bytes | None:
    if type(data) is bytes:
        return data
    data = cast(memoryview, data)
    source = data.obj
    if (
        type(source) is bytes
        and data.c_contiguous
        and data.itemsize == 1
        and data.nbytes == len(source)
    ):
        return source
    return None


def skip_comment(data: bytes | memoryview, pos: int, data_len: int) -> int:
    pos += 1
    while pos < data_len and data[pos] not in (10, 13):
        pos += 1
    return pos


def skip_literal_string(data: bytes | memoryview, pos: int, data_len: int) -> int:
    pos += 1
    depth = 1
    while pos < data_len and depth:
        byte = data[pos]
        if byte == 92:
            pos = min(pos + 2, data_len)
        elif byte == 40:
            depth += 1
            pos += 1
        elif byte == 41:
            depth -= 1
            pos += 1
        else:
            pos += 1
    return pos


def skip_hex_string(data: bytes | memoryview, pos: int, data_len: int) -> int:
    pos += 1
    while pos < data_len:
        if data[pos] == 62:
            return pos + 1
        pos += 1
    return pos
