# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import binascii
import typing
from typing import ClassVar, Self

from core_adobe_fonts.cmap.lexical import (
    SEPARATOR_TABLE,
    WHITESPACE,
    WS_TABLE,
    read_literal_string,
)
from core_records import Record, frozen_setattr


def decode_literal_string(data: bytes | bytearray | memoryview) -> bytes:
    raw = memoryview(data)
    n = len(raw)
    if n < 2 or raw[0] != 40:
        raise ValueError("invalid literal string")

    value, ignored_end = read_literal_string(raw, 0, n)
    if value is None:
        raise ValueError("unterminated literal string")
    return value


CMapTokenKind = typing.Literal["array", "delimiter", "hex", "literal", "procedure", "word"]


class CMapToken(Record):
    __slots__ = ("value", "start", "end", "kind")

    value: bytes
    start: int
    end: int
    kind: CMapTokenKind

    __fields__: ClassVar[tuple[str, ...]] = ("value", "start", "end", "kind")
    __match_args__ = ("value", "start", "end", "kind")

    def __init__(self, value: bytes, start: int, end: int, kind: CMapTokenKind) -> None:
        frozen_setattr(self, "value", value)
        frozen_setattr(self, "start", start)
        frozen_setattr(self, "end", end)
        frozen_setattr(self, "kind", kind)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.value == other.value
            and self.start == other.start
            and self.end == other.end
            and self.kind == other.kind
        )

    def __hash__(self) -> int:
        return hash((self.value, self.start, self.end, self.kind))


class CMapTokenRun(Record):
    """Source bytes and the tokens read from them, shared by a CMap and its blocks."""

    __slots__ = ("data", "tokens")

    data: bytes
    tokens: tuple[CMapToken, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("data", "tokens")
    __match_args__ = ("data", "tokens")

    def __init__(self, data: bytes, tokens: tuple[CMapToken, ...]) -> None:
        frozen_setattr(self, "data", data)
        frozen_setattr(self, "tokens", tokens)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.data == other.data and self.tokens == other.tokens

    def __hash__(self) -> int:
        return hash((self.data, self.tokens))


class CMapBlock(CMapTokenRun):
    __slots__ = ()

    def token_values(
        self, *, include_arrays: bool = False, include_words: bool = False
    ) -> list[bytes]:
        return cmap_token_values(
            self.tokens, include_arrays=include_arrays, include_words=include_words
        )


class CMapProgram(CMapTokenRun):
    __slots__ = ()

    # Parsing is strict. A subclass relaxes it through the hooks below: how the
    # source is tokenised, which whole-program rules apply, and what a block's
    # count, operands, and terminator must satisfy.

    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview) -> Self:
        source = data if type(data) is bytes else bytes(data)
        tokens = cls.read_tokens(source)
        scoped_tokens = scope_cmap_tokens(tokens)
        cls.validate_program(tokens, scoped_tokens)
        return cls(source, scoped_tokens)

    @classmethod
    def read_tokens(cls, source: bytes) -> tuple[CMapToken, ...]:
        return tuple(iter_cmap_tokens(source, group_arrays=True))

    @classmethod
    def validate_program(
        cls, tokens: tuple[CMapToken, ...], scoped_tokens: tuple[CMapToken, ...]
    ) -> None:
        if any(
            token.kind == "word" and token.value == b"begincmap" for token in tokens
        ) and not any(token.kind == "word" and token.value == b"endcmap" for token in tokens):
            raise ValueError("unterminated CMap program")
        operators = {token.value for token in scoped_tokens if token.kind == "word"}
        if {b"usecmap", b"begincodespacerange"} <= operators:
            raise ValueError("CMap usecmap cannot redefine codespacerange")

    def blocks(self, begin: bytes, end: bytes) -> typing.Iterator[CMapBlock]:
        for ignored_begin, block in self.blocks_in_order({begin: end}):
            yield block

    def blocks_in_order(
        self, delimiters: dict[bytes, bytes]
    ) -> typing.Iterator[tuple[bytes, CMapBlock]]:
        begin_keyword: bytes | None = None
        end_keyword: bytes | None = None
        block_start: int | None = None
        block_tokens: list[CMapToken] = []
        declared_count = 0
        for token_index, token in enumerate(self.tokens):
            if block_start is None:
                if (
                    token.kind == "word"
                    and (matched_end := delimiters.get(token.value)) is not None
                ):
                    declared_count = self.block_count(token_index)
                    begin_keyword = token.value
                    end_keyword = matched_end
                    block_start = token.end
                continue
            if token.kind == "word" and token.value == end_keyword:
                assert begin_keyword is not None
                self.validate_block(begin_keyword, block_tokens, declared_count)
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
        if block_start is not None:
            self.reject_unterminated_block()

    def block_count(self, begin_index: int) -> int:
        """The entry count before the begin operator at `begin_index`."""
        if begin_index == 0:
            raise ValueError("missing CMap block count")
        count = self.tokens[begin_index - 1]
        if count.kind != "word" or not count.value.isdigit():
            raise ValueError("invalid CMap block count")
        declared_count = int(count.value)
        if not 0 <= declared_count <= 100:
            raise ValueError("CMap block count exceeds 100")
        return declared_count

    def validate_block(
        self, begin_keyword: bytes, block_tokens: list[CMapToken], declared_count: int
    ) -> None:
        arity = (
            3 if begin_keyword.endswith(b"range") and begin_keyword != b"begincodespacerange" else 2
        )
        if len(block_tokens) != declared_count * arity:
            raise ValueError("CMap block count does not match operands")
        for index, operand in enumerate(block_tokens):
            position = index % arity
            if position < arity - 1 or begin_keyword == b"begincodespacerange":
                allowed = {"hex"}
            elif begin_keyword in {
                b"begincidchar",
                b"begincidrange",
                b"beginnotdefchar",
                b"beginnotdefrange",
            }:
                allowed = {"word"}
            else:
                allowed = {"hex", "array"} if begin_keyword == b"beginbfrange" else {"hex"}
            if operand.kind not in allowed:
                raise ValueError("invalid CMap mapping operand type")

    def reject_unterminated_block(self) -> None:
        """A block without its end operator; returning drops the block."""
        raise ValueError("unterminated CMap mapping block")


def scan_cmap_literal_string_end(data: bytes, pos: int) -> tuple[int, bool]:
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


def scan_cmap_composite_end(data: bytes, pos: int) -> tuple[int, bool]:
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
            case 37:
                while end < n and data[end] not in (10, 13):
                    end += 1
            case 40:
                end, ignored_terminated = scan_cmap_literal_string_end(data, end)
            case 60:
                if end + 1 < n and data[end + 1] == 60:
                    end += 2
                    continue
                close = data.find(b">", end + 1)
                if close < 0:
                    return n, False
                end = close + 1
            case 91 | 123:
                end, terminated = scan_cmap_composite_end(data, end)
                if not terminated:
                    return n, False
            case _ if current == closing:
                return end + 1, True
            case _:
                end += 1
    return n, False


def scope_cmap_tokens(tokens: tuple[CMapToken, ...]) -> tuple[CMapToken, ...]:
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


def iter_cmap_tokens(data: bytes, *, group_arrays: bool) -> typing.Iterator[CMapToken]:
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if WS_TABLE[byte]:
            pos += 1
            continue
        match byte:
            case 37:
                while pos < n and data[pos] not in (10, 13):
                    pos += 1
            case 40:
                end, terminated = scan_cmap_literal_string_end(data, pos)
                if not terminated:
                    raise ValueError("unterminated CMap literal string")
                yield CMapToken(data[pos:end], pos, end, "literal")
                pos = end
            case 60:
                if pos + 1 < n and data[pos + 1] == 60:
                    yield CMapToken(b"<<", pos, pos + 2, "delimiter")
                    pos += 2
                    continue
                end = data.find(b">", pos + 1)
                if end < 0:
                    raise ValueError("unterminated CMap hex string")
                end += 1
                yield CMapToken(data[pos:end], pos, end, "hex")
                pos = end
            case 62:
                end = pos + (2 if pos + 1 < n and data[pos + 1] == 62 else 1)
                yield CMapToken(data[pos:end], pos, end, "delimiter")
                pos = end
            case 91 if group_arrays:
                end, terminated = scan_cmap_composite_end(data, pos)
                if not terminated:
                    raise ValueError("unterminated CMap array")
                yield CMapToken(data[pos:end], pos, end, "array")
                pos = end
            case 123:
                end, terminated = scan_cmap_composite_end(data, pos)
                if not terminated:
                    raise ValueError("unterminated CMap procedure")
                yield CMapToken(data[pos:end], pos, end, "procedure")
                pos = end
            case 47:
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
    tokens = iter_cmap_tokens(data, group_arrays=include_arrays)
    return cmap_token_values(tokens, include_arrays=include_arrays, include_words=include_words)


def cmap_token_values(
    tokens: typing.Iterable[CMapToken], *, include_arrays: bool, include_words: bool
) -> list[bytes]:
    kinds: set[CMapTokenKind] = {"hex", "literal"}
    if include_arrays:
        kinds.add("array")
    if include_words:
        kinds.update(("delimiter", "word"))
    return [token.value for token in tokens if token.kind in kinds]


def cmap_metadata(data: bytes | CMapProgram) -> tuple[str | None, int | None]:
    program = data if isinstance(data, CMapProgram) else CMapProgram.parse(data)
    tokens = program.tokens
    usecmap_name: str | None = None
    wmode: int | None = None
    for index, token in enumerate(tokens):
        if token.kind != "word":
            continue
        if token.value == b"usecmap":
            if (
                index == 0
                or tokens[index - 1].kind != "word"
                or not tokens[index - 1].value.startswith(b"/")
            ):
                raise ValueError("invalid CMap usecmap operand")
            usecmap_name = tokens[index - 1].value[1:].decode("latin-1")
            if not usecmap_name:
                raise ValueError("empty CMap usecmap name")
        elif token.value == b"/WMode":
            if (
                index + 2 >= len(tokens)
                or tokens[index + 2].value != b"def"
                or tokens[index + 1].value not in {b"0", b"1"}
            ):
                raise ValueError("invalid CMap WMode")
            wmode = int(tokens[index + 1].value)
    return usecmap_name, wmode


def decode_cmap_hex_token(token: bytes) -> bytes:
    if len(token) < 2 or not token.startswith(b"<") or not token.endswith(b">"):
        raise ValueError("invalid CMap hex string")
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
        return decode_literal_string(token)
    except ValueError as exc:
        raise ValueError("invalid CMap literal string") from exc


__all__ = [
    "decode_literal_string",
    "CMapTokenKind",
    "CMapToken",
    "CMapBlock",
    "CMapProgram",
    "scope_cmap_tokens",
    "iter_cmap_tokens",
    "cmap_tokens",
    "cmap_metadata",
    "decode_cmap_hex_token",
    "decode_cmap_token",
]
