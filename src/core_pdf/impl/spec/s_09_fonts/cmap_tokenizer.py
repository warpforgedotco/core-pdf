"""Tokenization helpers for PDF CMap programs."""

from __future__ import annotations

import binascii
import typing
from dataclasses import dataclass

from core_pdf.impl.spec.s_07_syntax_primitives.scanning import read_literal_string
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import (
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
)


def decode_pdf_literal_string(data: bytes | bytearray | memoryview) -> bytes:
    raw = memoryview(data)
    n = len(raw)
    if n < 2 or raw[0] != 40:
        raise ValueError("invalid PDF literal string")

    value, ignored_end = read_literal_string(raw, 0, n)
    if value is None:
        raise ValueError("unterminated PDF literal string")
    return value


CMapTokenKind = typing.Literal["array", "delimiter", "hex", "literal", "procedure", "word"]


@dataclass(frozen=True, slots=True)
class CMapToken:
    """One lexical CMap token and its location in the source program."""

    value: bytes
    start: int
    end: int
    kind: CMapTokenKind


@dataclass(frozen=True, slots=True)
class CMapBlock:
    """The operands between an exact ``begin*``/``end*`` operator pair."""

    data: bytes
    tokens: tuple[CMapToken, ...]

    def token_values(
        self, *, include_arrays: bool = False, include_words: bool = False
    ) -> list[bytes]:
        kinds: set[CMapTokenKind] = {"hex", "literal"}
        if include_arrays:
            kinds.add("array")
        if include_words:
            kinds.update(("delimiter", "word"))
        return [token.value for token in self.tokens if token.kind in kinds]


@dataclass(frozen=True, slots=True)
class CMapProgram:
    """A tokenized CMap program shared by its semantic compilers."""

    data: bytes
    tokens: tuple[CMapToken, ...]

    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview) -> "CMapProgram":
        source = data if type(data) is bytes else bytes(data)
        tokens = tuple(internal_cmap_token_spans(source, group_arrays=True))
        return cls(source, internal_scoped_cmap_tokens(tokens))

    def blocks(self, begin: bytes, end: bytes) -> typing.Iterator[CMapBlock]:
        """Yield blocks delimited by exact word tokens.

        Grouped array and string tokens ensure that operator-looking contents do
        not accidentally terminate a surrounding CMap section.
        """
        for ignored_begin, block in self.blocks_in_order({begin: end}):
            yield block

    def blocks_in_order(
        self, delimiters: dict[bytes, bytes]
    ) -> typing.Iterator[tuple[bytes, CMapBlock]]:
        """Yield selected block kinds in their original program order."""
        begin_keyword: bytes | None = None
        end_keyword: bytes | None = None
        block_start: int | None = None
        block_tokens: list[CMapToken] = []
        for token in self.tokens:
            if block_start is None:
                if (
                    token.kind == "word"
                    and (matched_end := delimiters.get(token.value)) is not None
                ):
                    begin_keyword = token.value
                    end_keyword = matched_end
                    block_start = token.end
                continue
            if token.kind == "word" and token.value == end_keyword:
                assert begin_keyword is not None
                yield (
                    begin_keyword,
                    CMapBlock(self.data[block_start : token.start], tuple(block_tokens)),
                )
                begin_keyword = None
                end_keyword = None
                block_start = None
                block_tokens.clear()
                continue
            block_tokens.append(token)


def iter_blocks(data: bytes | memoryview, begin: bytes, end: bytes) -> typing.Iterator[bytes]:
    """Yield CMap blocks delimited by exact, non-comment operator tokens."""
    for block in CMapProgram.parse(data).blocks(begin, end):
        yield block.data


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
            end += 1
            if end < n:
                if data[end] == 13 and end + 1 < n and data[end + 1] == 10:
                    end += 2
                else:
                    end += 1
            continue
        if current == 40:
            depth += 1
        elif current == 41:
            depth -= 1
        end += 1
    return min(end, n), depth == 0


def internal_scan_cmap_composite_end(data: bytes, pos: int) -> tuple[int, bool]:
    """Scan a ``[...]`` array or ``{...}`` procedure as one composite object."""
    opening = data[pos]
    match opening:
        case 91:
            closing = 93
        case 123:
            closing = 125
        case _:
            raise ValueError("invalid CMap composite opener")
    end = pos + 1
    n = len(data)
    while end < n:
        current = data[end]
        match current:
            case 37:  # comment
                while end < n and data[end] not in (10, 13):
                    end += 1
            case 40:  # literal string
                end, ignored_terminated = internal_scan_cmap_literal_string_end(data, end)
            case 60:  # hex string or dictionary opener
                if end + 1 < n and data[end + 1] == 60:
                    end += 2
                    continue
                close = data.find(b">", end + 1)
                if close < 0:
                    return n, False
                end = close + 1
            case 91 | 123:  # nested array or procedure
                end, terminated = internal_scan_cmap_composite_end(data, end)
                if not terminated:
                    return n, False
            case _ if current == closing:
                return end + 1, True
            case _:
                end += 1
    return n, False


def internal_scoped_cmap_tokens(tokens: tuple[CMapToken, ...]) -> tuple[CMapToken, ...]:
    """Restrict a CMap program to its top-level ``begincmap`` region when present."""
    begin_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.kind == "word" and token.value == b"begincmap"
        ),
        None,
    )
    scope_start = begin_index + 1 if begin_index is not None else 0
    end_index = next(
        (
            index
            for index, token in enumerate(tokens[scope_start:], start=scope_start)
            if token.kind == "word" and token.value == b"endcmap"
        ),
        len(tokens),
    )
    return tokens[scope_start:end_index]


def internal_cmap_token_spans(data: bytes, *, group_arrays: bool) -> typing.Iterator[CMapToken]:
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        match byte:
            case 37:  # comment
                while pos < n and data[pos] not in (10, 13):
                    pos += 1
            case 40:  # literal string
                end, terminated = internal_scan_cmap_literal_string_end(data, pos)
                if terminated:
                    yield CMapToken(data[pos:end], pos, end, "literal")
                pos = end
            case 60:  # hex string or dictionary opener
                if pos + 1 < n and data[pos + 1] == 60:
                    yield CMapToken(b"<<", pos, pos + 2, "delimiter")
                    pos += 2
                    continue
                end = data.find(b">", pos + 1)
                if end < 0:
                    return
                end += 1
                yield CMapToken(data[pos:end], pos, end, "hex")
                pos = end
            case 62:  # dictionary closer or stray delimiter
                end = pos + (2 if pos + 1 < n and data[pos + 1] == 62 else 1)
                yield CMapToken(data[pos:end], pos, end, "delimiter")
                pos = end
            case 91 if group_arrays:
                end, terminated = internal_scan_cmap_composite_end(data, pos)
                if terminated:
                    yield CMapToken(data[pos:end], pos, end, "array")
                pos = end
            case 123:  # procedure
                end, terminated = internal_scan_cmap_composite_end(data, pos)
                if terminated:
                    yield CMapToken(data[pos:end], pos, end, "procedure")
                pos = end
            case 47:  # name object
                end = pos + 1
                while end < n and not SEPARATOR_TABLE[data[end]]:
                    end += 1
                yield CMapToken(data[pos:end], pos, end, "word")
                pos = end
            case 91 | 93 | 41 | 125:
                yield CMapToken(data[pos : pos + 1], pos, pos + 1, "delimiter")
                pos += 1
            case _:
                end = pos + 1
                while end < n and not SEPARATOR_TABLE[data[end]]:
                    end += 1
                yield CMapToken(data[pos:end], pos, end, "word")
                pos = end


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
    tokens = internal_cmap_token_spans(data, group_arrays=include_arrays)
    kinds: set[CMapTokenKind] = {"hex", "literal"}
    if include_arrays:
        kinds.add("array")
    if include_words:
        kinds.update(("delimiter", "word"))
    return [token.value for token in tokens if token.kind in kinds]


def cmap_metadata(data: bytes | CMapProgram) -> tuple[str | None, int | None]:
    program = data if isinstance(data, CMapProgram) else CMapProgram.parse(data)
    words = [token.value for token in program.tokens if token.kind == "word"]
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
    raw = token[1:-1].translate(None, WHITESPACE)
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
