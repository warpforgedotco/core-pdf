# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


def find_indirect_object_header(data: memoryview, search_start: int, search_end: int) -> int | None:
    raw = data[search_start:search_end]
    raw_bytes = raw.tobytes()
    pos = 0
    n = len(raw)
    whitespace = (0, 9, 10, 12, 13, 32)
    while pos < n:
        marker = raw_bytes.find(b"obj", pos)
        if marker < 0:
            return None
        parsed = parse_object_header_prefix(raw, marker, whitespace)
        if parsed is not None:
            return search_start + parsed
        pos = marker + 3
    return None


def parse_object_header_prefix(
    data: memoryview, marker: int, whitespace: tuple[int, ...]
) -> int | None:
    if marker + 3 < len(data) and data[marker + 3] not in whitespace:
        return None
    pos = marker - 1
    while pos >= 0 and data[pos] in whitespace:
        pos -= 1
    gen_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    gen_start = pos + 1
    if gen_start == gen_end:
        return None
    while pos >= 0 and data[pos] in whitespace:
        pos -= 1
    obj_end = pos + 1
    while pos >= 0 and 48 <= data[pos] <= 57:
        pos -= 1
    obj_start = pos + 1
    if obj_start == obj_end:
        return None
    if pos >= 0 and data[pos] not in whitespace:
        return None
    return obj_start
