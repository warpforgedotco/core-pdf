"""Tokenization helpers for PDF CMap programs."""

from __future__ import annotations

import binascii
import re
import typing

from core_pdf.impl.engine.spec.s_09_fonts.cmap_pdf_string import decode_pdf_literal_string

PDF_WHITESPACE_BYTES = bytes([1 if byte in b"\x00\t\n\f\r " else 0 for byte in range(256)])
CMAP_HEX_TOKEN_RE = re.compile(rb"<(?!<)[^>]*>")


def iter_blocks(data: bytes | memoryview, begin: bytes, end: bytes) -> typing.Iterator[bytes]:
    if not isinstance(data, bytes):
        data = bytes(data)
    search_start = 0
    begin_length = len(begin)
    end_length = len(end)
    while True:
        begin_pos = data.find(begin, search_start)
        if begin_pos < 0:
            return
        block_start = begin_pos + begin_length
        end_pos = data.find(end, block_start)
        if end_pos < 0:
            return
        yield data[block_start:end_pos]
        search_start = end_pos + end_length


def internal_scan_cmap_literal_string_end(data: bytes, pos: int) -> tuple[int, bool]:
    """Scan a ``(...)`` literal string starting at ``pos``.

    Returns ``(end, terminated)``: ``end`` is the position just past the closing
    unescaped ``)`` if the string is properly balanced, otherwise ``len(data)`` with
    ``terminated=False``.
    """
    end = pos + 1
    depth = 1
    n = len(data)
    while end < n and depth:
        current = data[end]
        if current == 92:
            end += 2
            continue
        if current == 40:
            depth += 1
        elif current == 41:
            depth -= 1
        end += 1
    return end, depth == 0


def skip_cmap_literal_string(data: bytes, pos: int) -> int:
    end, ignored_terminated = internal_scan_cmap_literal_string_end(data, pos)
    return end


def internal_scan_cmap_array_end(data: bytes, pos: int) -> tuple[int, bool]:
    """Scan a ``[...]`` array starting at ``pos``.

    Returns ``(end, terminated)``: ``end`` is the position just past the matching
    closing ``]`` if properly balanced, otherwise ``len(data)`` with ``terminated=False``.
    """
    end = pos + 1
    depth = 1
    n = len(data)
    while end < n and depth:
        current = data[end]
        if current == 37:
            while end < n and data[end] not in (10, 13):
                end += 1
            continue
        if current == 40:
            end, ignored_terminated = internal_scan_cmap_literal_string_end(data, end)
            continue
        if current == 60:
            close = data.find(b">", end + 1)
            if close < 0:
                return n, False
            end = close + 1
            continue
        if current == 91:
            depth += 1
        elif current == 93:
            depth -= 1
        end += 1
    return end, depth == 0


def skip_cmap_array(data: bytes, pos: int) -> int:
    end, ignored_terminated = internal_scan_cmap_array_end(data, pos)
    return end


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
    if not include_arrays and not include_words and b"(" not in data and b"%" not in data:
        return CMAP_HEX_TOKEN_RE.findall(data)
    tokens: list[bytes] = []
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if byte == 37:
            while pos < n and data[pos] not in (10, 13):
                pos += 1
            continue
        if byte == 60:
            if pos + 1 < n and data[pos + 1] == 60:
                pos += 2
                continue
            end = data.find(b">", pos + 1)
            if end < 0:
                break
            candidate = data[pos : end + 1]
            tokens.append(candidate)
            pos = end + 1
            continue
        if byte == 40:
            end, terminated = internal_scan_cmap_literal_string_end(data, pos)
            if terminated:
                tokens.append(data[pos:end])
            pos = end
            continue
        if include_arrays and byte == 91:
            end, terminated = internal_scan_cmap_array_end(data, pos)
            if terminated:
                tokens.append(data[pos:end])
                pos = end
                continue
        if include_words and not PDF_WHITESPACE_BYTES[byte]:
            end = pos + 1
            while end < n and not PDF_WHITESPACE_BYTES[data[end]] and data[end] not in b"[]<>()/%":
                end += 1
            tokens.append(data[pos:end])
            pos = end
            continue
        pos += 1
    return tokens


def cmap_noncomment_words(data: bytes) -> list[bytes]:
    return cmap_tokens(data, include_words=True)


def cmap_usecmap_name(data: bytes) -> str | None:
    words = cmap_noncomment_words(data)
    for index, word in enumerate(words[1:], start=1):
        if word != b"usecmap":
            continue
        name = words[index - 1]
        if not name.startswith(b"/"):
            continue
        try:
            return name[1:].decode("latin-1")
        except UnicodeDecodeError:
            return None
    return None


def cmap_metadata(data: bytes) -> tuple[str | None, int | None]:
    words = cmap_noncomment_words(data)
    usecmap_name: str | None = None
    wmode: int | None = None
    usecmap_checked = False
    wmode_checked = False
    for index, word in enumerate(words):
        if not usecmap_checked and index > 0 and word == b"usecmap":
            usecmap_checked = True
            name = words[index - 1]
            if name.startswith(b"/"):
                try:
                    usecmap_name = name[1:].decode("latin-1")
                except UnicodeDecodeError:
                    usecmap_name = None
        if not wmode_checked and index + 2 < len(words) and word == b"/WMode":
            wmode_checked = True
            if words[index + 2] == b"def":
                try:
                    value = int(words[index + 1])
                except ValueError:
                    continue
                if value in {0, 1}:
                    wmode = value
        if usecmap_checked and wmode_checked:
            break
    return usecmap_name, wmode


def decode_cmap_hex_token(token: bytes) -> bytes:
    raw = token[1:-1].translate(None, b"\x00\t\n\f\r ")
    if len(raw) & 1:
        raw += b"0"
    try:
        return binascii.unhexlify(raw)
    except binascii.Error as exc:
        raise ValueError("invalid CMap hex string") from exc


def decode_cmap_token(token: bytes) -> bytes:
    if token.startswith(b"<"):
        return decode_cmap_hex_token(token)
    try:
        return decode_pdf_literal_string(token)
    except ValueError as exc:
        raise ValueError("invalid CMap literal string") from exc
