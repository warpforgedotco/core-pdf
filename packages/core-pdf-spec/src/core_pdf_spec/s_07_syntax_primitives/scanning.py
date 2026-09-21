# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax_primitives.numbers import parse_identifier_tokens
from core_pdf_spec.s_07_syntax_primitives.tokens import LexicalRules, lexical_rules

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

STRING_SPECIAL_TABLE = bytes([1 if i in b"()\\\r\n" else 0 for i in range(256)])
internal_STRING_SPECIAL_RE = re.compile(b"[" + re.escape(b"()\\\r\n") + b"]")
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


def skip_pdf_ignored(
    data: bytes | memoryview,
    position: int,
    data_len: int,
    *,
    rules: LexicalRules | None = None,
) -> int:
    rules = lexical_rules() if rules is None else rules
    ws_table = rules.whitespace_table
    pos = position
    if pos >= data_len:
        return pos
    if isinstance(data, bytes) or data.c_contiguous:
        start = pos
        if data[pos] != 37:
            short_end = min(data_len, pos + 8)
            while pos < short_end and ws_table[data[pos]]:
                pos += 1
            if pos >= data_len:
                return pos
            byte = data[pos]
            if byte != 37 and not ws_table[byte]:
                return pos
            if byte != 37:
                pos = start
        match = rules.ignored_re.match(data, pos, data_len)
        if match is not None:
            return match.end()
    while pos < data_len:
        byte = data[pos]
        if ws_table[byte]:
            pos += 1
            continue
        if byte != 37:
            break
        pos += 1
        while pos < data_len:
            byte = data[pos]
            if byte in {10, 13}:
                pos += 1
                if pos < data_len:
                    next_byte = data[pos]
                    if (byte == 13 and next_byte == 10) or (byte == 10 and next_byte == 13):
                        pos += 1
                break
            pos += 1
    return pos


def looks_like_indirect_object_header(
    data: memoryview,
    position: int,
    data_len: int,
    *,
    rules: LexicalRules | None = None,
    parse_identifier: Callable[[bytes, bytes], tuple[int, int]] | None = None,
) -> bool:
    rules = lexical_rules() if rules is None else rules
    pos = position
    tokens: list[bytes] = []
    for _ in range(3):
        start = pos
        while pos < data_len and not rules.separator_table[data[pos]]:
            pos += 1
        if start == pos:
            return False
        tokens.append(bytes(data[start:pos]))
        if len(tokens) < 3:
            pos = skip_pdf_ignored(data, pos, data_len, rules=rules)
    if tokens[2] != b"obj":
        return False
    try:
        if parse_identifier is None:
            parse_identifier_tokens(tokens[0], tokens[1], canonical=rules.canonical_identifiers)
        else:
            parse_identifier(tokens[0], tokens[1])
    except PdfParseError:
        return False
    return True


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


def read_literal_string(
    data: bytes | memoryview,
    pos: int,
    data_len: int,
    *,
    unknown_escape: Callable[[int], bytes] | None = None,
    eol_pair: Callable[[int, int], bool] | None = None,
) -> tuple[bytes | None, int]:
    pos += 1
    if isinstance(data, memoryview) and data.format != "B":
        end_idx = pos
    elif isinstance(data, bytes) or data.c_contiguous:
        match = internal_STRING_SPECIAL_RE.search(data, pos, data_len)
        end_idx = data_len if match is None else match.start()
    else:
        end_idx = pos
        while end_idx < data_len and not STRING_SPECIAL_TABLE[data[end_idx]]:
            end_idx += 1

    if end_idx < data_len and data[end_idx] == 41:
        return bytes(data[pos:end_idx]), end_idx + 1

    out = bytearray()
    if end_idx > pos:
        prefix = data[pos:end_idx]
        out.extend(prefix if isinstance(prefix, bytes) or prefix.c_contiguous else prefix.tobytes())
        pos = end_idx

    depth = 1
    while pos < data_len:
        byte = data[pos]
        pos += 1
        if byte == 40:
            depth += 1
            out.append(byte)
        elif byte == 41:
            depth -= 1
            if depth == 0:
                return bytes(out), pos
            out.append(byte)
        elif byte == 92:
            if pos >= data_len:
                continue
            esc = data[pos]
            pos += 1
            if 48 <= esc <= 55:
                oct_val = esc - 48
                count = 1
                while count < 3 and pos < data_len and 48 <= data[pos] <= 55:
                    oct_val = (oct_val << 3) | (data[pos] - 48)
                    pos += 1
                    count += 1
                out.append(oct_val & 0xFF)
            elif esc in (10, 13):
                if pos < data_len and (
                    eol_pair(esc, data[pos])
                    if eol_pair is not None
                    else esc == 13 and data[pos] == 10
                ):
                    pos += 1
            elif (mapped := STRING_ESCAPE.get(esc)) is not None:
                out.extend(mapped)
            elif unknown_escape is not None:
                out.extend(unknown_escape(esc))
            else:
                out.append(esc)
        elif byte in {13, 10}:
            out.append(10)
            if pos < data_len:
                next_byte = data[pos]
                if (
                    eol_pair(byte, next_byte)
                    if eol_pair is not None
                    else byte == 13 and next_byte == 10
                ):
                    pos += 1
        else:
            out.append(byte)
    return None, pos


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


def skip_name(
    data: bytes | memoryview, pos: int, data_len: int, *, rules: LexicalRules | None = None
) -> int:
    separators = (lexical_rules() if rules is None else rules).separator_table
    pos += 1
    while pos < data_len and not separators[data[pos]]:
        pos += 1
    return pos


__all__ = (
    "EMPTY_TRANSLATE_TABLE",
    "FindableSizedBuffer",
    "HEX_VALUE",
    "STRING_ESCAPE",
    "STRING_SPECIAL_TABLE",
    "full_source_buffer",
    "full_source_bytes",
    "looks_like_indirect_object_header",
    "read_literal_string",
    "skip_comment",
    "skip_hex_string",
    "skip_literal_string",
    "skip_name",
    "skip_pdf_ignored",
)
