# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core_pdf.impl.engine.spec.s_09_fonts.data.core14 import PDFDOC_ENCODING_OVERRIDES

BYTE_CACHE = [bytes([i]) for i in range(256)]
CHR_TABLE: list[str] = [chr(i) for i in range(256)]
PDFDOC_ENCODING_TABLE: list[str] = [
    PDFDOC_ENCODING_OVERRIDES.get(i, CHR_TABLE[i]) for i in range(256)
]


class CMapWithRanges(Protocol):
    @property
    def code_space_ranges(self) -> Sequence[tuple[bytes, bytes]]: ...


def decode_pdf_text_string(data: bytes) -> str:
    if data.startswith(b"\xfe\xff"):
        try:
            return data[2:].decode("utf-16-be")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16BE data") from exc
    if data.startswith(b"\xff\xfe"):
        try:
            return data[2:].decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-16LE data") from exc
    return "".join(PDFDOC_ENCODING_TABLE[b] for b in data)


def decode_utf16be(data: bytes) -> str:
    if not data:
        return ""
    if len(data) == 1:
        return chr(data[0])
    if data.startswith((b"\xfe\xff", b"\xff\xfe")):
        try:
            return data.decode("utf-16")
        except (UnicodeDecodeError, ValueError):
            raise ValueError("invalid UTF-16BE data")
    buf = data if len(data) % 2 == 0 else b"\x00" + data
    try:
        return buf.decode("utf-16-be", "replace")
    except (UnicodeDecodeError, ValueError):
        return data.decode("latin-1", "replace")


def split_data_by_ranges(data: bytes, ranges: Sequence[tuple[bytes, bytes]]) -> list[bytes]:
    if not data:
        return []

    n = len(data)
    if not ranges:
        if n % 2 == 0:
            return [data[i : i + 2] for i in range(0, n, 2)]
        return [data[i : i + 1] for i in range(n)]

    ranges_by_len: dict[int, list[tuple[bytes, bytes]]] = {}
    for start, end in ranges:
        length = len(start)
        if length == 0 or len(end) != length or start > end:
            raise ValueError("invalid code space ranges")
        ranges_by_len.setdefault(length, []).append((start, end))

    lengths = sorted(ranges_by_len.keys(), reverse=True)
    if lengths == [2] and n % 2 == 0:
        chunk0 = data[0:2]
        if any(start <= chunk0 <= end for start, end in ranges_by_len[2]):
            return [data[i : i + 2] for i in range(0, n, 2)]

    chunks: list[bytes] = []
    pos = 0
    while pos < n:
        match_found = False
        for length in lengths:
            if pos + length > n:
                continue
            chunk = data[pos : pos + length]
            for start, end in ranges_by_len[length]:
                if start <= chunk <= end:
                    chunks.append(chunk)
                    pos += length
                    match_found = True
                    break
            if match_found:
                break
        if not match_found:
            chunks.append(data[pos : pos + 1])
            pos += 1
    return chunks


def split_chunks(data: bytes, is_cid: bool, cmap: CMapWithRanges | None) -> list[bytes]:
    if not data:
        return []
    ranges = cmap.code_space_ranges if cmap else []
    if is_cid or ranges:
        return split_data_by_ranges(data, ranges)
    return [BYTE_CACHE[b] for b in data]
