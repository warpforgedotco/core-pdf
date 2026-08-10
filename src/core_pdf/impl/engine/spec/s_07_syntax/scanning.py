# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from core_pdf.impl.engine.spec.s_07_syntax.tokens import SEPARATOR_TABLE


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
    "skip_comment",
    "skip_hex_string",
    "skip_literal_string",
    "skip_name",
)
