"""Tokenization helpers for PDF CMap programs."""

from __future__ import annotations

import typing

from core_pdf.impl.engine.spec.s_09_fonts.cmap_pdf_string import decode_pdf_literal_string

HEX_BYTES = bytes([1 if byte in b"0123456789abcdefABCDEF" else 0 for byte in range(256)])
PDF_WHITESPACE_BYTES = bytes([1 if byte in b"\x00\t\n\f\r " else 0 for byte in range(256)])


def iter_blocks(data: bytes | memoryview, begin: bytes, end: bytes) -> typing.Iterator[bytes]:
    if not isinstance(data, bytes):
        data = bytes(data)
    block_start: int | None = None
    for word, start, stop in cmap_word_spans(data):
        if block_start is None:
            if word == begin:
                block_start = stop
        elif word == end:
            yield data[block_start:start]
            block_start = None


def skip_cmap_literal_string(data: bytes, pos: int) -> int:
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
    return end


def skip_cmap_array(data: bytes, pos: int) -> int:
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
            end = skip_cmap_literal_string(data, end)
            continue
        if current == 60:
            close = data.find(b">", end + 1)
            if close < 0:
                return n
            end = close + 1
            continue
        if current == 91:
            depth += 1
        elif current == 93:
            depth -= 1
        end += 1
    return end


def cmap_word_spans(data: bytes) -> typing.Iterator[tuple[bytes, int, int]]:
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if byte == 37:
            while pos < n and data[pos] not in (10, 13):
                pos += 1
            continue
        if byte == 40:
            pos = skip_cmap_literal_string(data, pos)
            continue
        if byte == 60:
            if pos + 1 < n and data[pos + 1] == 60:
                pos += 2
                continue
            close = data.find(b">", pos + 1)
            if close < 0:
                return
            pos = close + 1
            continue
        if byte == 91:
            pos = skip_cmap_array(data, pos)
            continue
        if PDF_WHITESPACE_BYTES[byte] or byte in b"[]<>()/%":
            pos += 1
            continue
        start = pos
        pos += 1
        while pos < n and not PDF_WHITESPACE_BYTES[data[pos]] and data[pos] not in b"[]<>()/%":
            pos += 1
        yield data[start:pos], start, pos


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
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
            end = pos + 1
            depth = 1
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
            if depth == 0:
                tokens.append(data[pos:end])
            pos = end
            continue
        if include_arrays and byte == 91:
            end = pos + 1
            depth = 1
            while end < n and depth:
                current = data[end]
                if current == 37:
                    while end < n and data[end] not in (10, 13):
                        end += 1
                    continue
                if current == 40:
                    end += 1
                    string_depth = 1
                    while end < n and string_depth:
                        current = data[end]
                        if current == 92:
                            end += 2
                            continue
                        if current == 40:
                            string_depth += 1
                        elif current == 41:
                            string_depth -= 1
                        end += 1
                    continue
                if current == 60:
                    close = data.find(b">", end + 1)
                    if close < 0:
                        break
                    end = close + 1
                    continue
                if current == 91:
                    depth += 1
                elif current == 93:
                    depth -= 1
                end += 1
            if depth == 0:
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
    if not all(HEX_BYTES[item] for item in raw):
        raise ValueError("invalid CMap hex string")
    if len(raw) & 1:
        raw += b"0"
    return bytes.fromhex(raw.decode("ascii"))


def decode_cmap_token(token: bytes) -> bytes:
    if token.startswith(b"<"):
        return decode_cmap_hex_token(token)
    try:
        return decode_pdf_literal_string(token)
    except ValueError as exc:
        raise ValueError("invalid CMap literal string") from exc
