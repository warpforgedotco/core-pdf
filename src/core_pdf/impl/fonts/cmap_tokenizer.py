from __future__ import annotations

import re
import typing
from collections.abc import Callable

from core_adobe_fonts.cmap.decoder import (
    CMapDecoder as PdfCMapDecoder,
)
from core_adobe_fonts.cmap.decoder import (
    CMapResourceResolver,
    CodeRangeT,
)
from core_adobe_fonts.cmap.ranges import (
    CIDRange,
    range_offset,
    remove_codes_in_range,
    validate_codespace_range,
)
from core_adobe_fonts.cmap.tokenizer import (
    CMapBlock,
    CMapToken,
    iter_cmap_tokens,
    scope_cmap_tokens,
)
from core_adobe_fonts.cmap.tokenizer import CMapProgram as PdfCMapProgram
from core_adobe_fonts.cmap.tokenizer import (
    decode_cmap_hex_token as decode_spec_cmap_hex_token,
)
from core_adobe_fonts.cmap.tokenizer import (
    decode_cmap_token as decode_spec_cmap_token,
)
from core_pdf_spec.s_07_syntax_primitives.scanning import read_literal_string

LEGACY_EOL_PAIR = re.compile(rb"\r\n|\n\r")


def decode_cmap_hex_token(token: bytes) -> bytes:
    if not token.startswith(b"<") or not token.endswith(b">"):
        token = b"<" + token[1:-1] + b">"
    return decode_spec_cmap_hex_token(token)


def decode_cmap_token(token: bytes) -> bytes:
    if token.startswith(b"<"):
        return decode_cmap_hex_token(token)
    if not token.startswith(b"("):
        return decode_spec_cmap_token(token)
    raw = memoryview(LEGACY_EOL_PAIR.sub(b"\r\n", token))
    if len(raw) < 2:
        raise ValueError("invalid PDF literal string")
    value, _ = read_literal_string(raw, 0, len(raw))
    if value is None:
        raise ValueError("unterminated PDF literal string")
    return value


class CMapProgram(PdfCMapProgram):
    @classmethod
    def parse(cls, data: bytes | bytearray | memoryview) -> CMapProgram:
        source = bytes(data)
        tokens: list[CMapToken] = []
        try:
            for token in iter_cmap_tokens(source, group_arrays=True):
                tokens.append(token)  # noqa: PERF402
        except ValueError:
            pass
        return cls(source, scope_cmap_tokens(tuple(tokens)))

    def blocks_in_order(
        self, delimiters: dict[bytes, bytes]
    ) -> typing.Iterator[tuple[bytes, CMapBlock]]:
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


def cmap_metadata(data: bytes | PdfCMapProgram) -> tuple[str | None, int | None]:
    program = data if isinstance(data, PdfCMapProgram) else CMapProgram.parse(data)
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


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
    tokens: list[CMapToken] = []
    try:
        for token in iter_cmap_tokens(data, group_arrays=include_arrays):
            tokens.append(token)  # noqa: PERF402
    except ValueError:
        pass
    return CMapBlock(data, tuple(tokens)).token_values(
        include_arrays=include_arrays, include_words=include_words
    )


def iter_blocks(data: bytes | memoryview, begin: bytes, end: bytes) -> typing.Iterator[bytes]:
    for block in CMapProgram.parse(data).blocks(begin, end):
        yield block.data


class CMapDecoder(PdfCMapDecoder):
    __slots__ = ()

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: CMapResourceResolver | None = None,
        inheritance_depth: int = 0,
        empty: bool = False,
        ancestor_names: tuple[str, ...] = (),
    ) -> None:
        if not empty and inheritance_depth > 5:
            raise ValueError("CMap usecmap nesting too deep")
        super().__init__(
            data,
            usecmap_resolver=usecmap_resolver,
            inheritance_depth=inheritance_depth,
            empty=empty,
            ancestor_names=ancestor_names,
        )

    @staticmethod
    def parse_program(data: bytes) -> CMapProgram:
        return CMapProgram.parse(data)

    @staticmethod
    def program_metadata(program: PdfCMapProgram) -> tuple[str | None, int | None]:
        return cmap_metadata(program)

    def validate_mappings(self) -> None:
        pass

    def parse_char_block(
        self,
        block: CMapBlock,
        mappings: dict[bytes, int],
    ) -> None:
        items = block.token_values(include_words=True)
        if len(items) % 2 != 0:
            items = items[:-1]
        for i in range(0, len(items), 2):
            code_token, cid_token = items[i], items[i + 1]
            if not (code_token.startswith(b"<") and code_token.endswith(b">")):
                continue
            try:
                code = decode_spec_cmap_hex_token(code_token)
                cid = int(cid_token)
            except ValueError, UnicodeDecodeError:
                continue
            if not code or not (0 <= cid <= 0xFFFF):
                continue
            mappings[code] = cid

    def parse_range_block(
        self,
        block: CMapBlock,
        mappings: dict[bytes, int],
        ranges: list[CodeRangeT],
        make_range: Callable[[bytes, bytes, int], CodeRangeT],
    ) -> None:
        items = block.token_values(include_words=True)
        if len(items) % 3 != 0:
            items = items[: len(items) - (len(items) % 3)]
        for i in range(0, len(items), 3):
            start_token, end_token, cid_token = items[i], items[i + 1], items[i + 2]
            if not (
                start_token.startswith(b"<")
                and start_token.endswith(b">")
                and end_token.startswith(b"<")
                and end_token.endswith(b">")
            ):
                continue
            try:
                start_bytes = decode_spec_cmap_hex_token(start_token)
                end_bytes = decode_spec_cmap_hex_token(end_token)
                cid = int(cid_token)
                validate_codespace_range(start_bytes, end_bytes)
            except ValueError, UnicodeDecodeError:
                continue
            if not (0 <= cid <= 0xFFFF):
                continue
            if make_range is CIDRange:
                last_cid = cid + range_offset(
                    end_bytes,
                    start_bytes,
                    end_bytes,
                    validate_range=False,
                    validate_code=False,
                )
                if not (0 <= last_cid <= 0xFFFF):
                    continue
            remove_codes_in_range(mappings, start_bytes, end_bytes)
            ranges.append(make_range(start_bytes, end_bytes, cid))

    @staticmethod
    def decode_codespace_token(token: bytes) -> bytes:
        return decode_cmap_hex_token(token)

    @staticmethod
    def resolve_usecmap(
        name: str,
        *,
        usecmap_resolver: CMapResourceResolver | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> PdfCMapDecoder | None:
        if name in {"OneByteIdentityH", "OneByteIdentityV"}:
            return CMapDecoder.identity(byte_width=1, wmode=int(name.endswith("V")))
        if name in ancestor_names:
            return None
        if usecmap_resolver is not None:
            resolved = usecmap_resolver(name)
            if resolved is not None:
                return CMapDecoder(
                    resolved,
                    usecmap_resolver=usecmap_resolver,
                    inheritance_depth=depth,
                    ancestor_names=(*ancestor_names, name),
                )
        if name in {"Identity-H", "Identity-V"}:
            return CMapDecoder.identity(byte_width=2, wmode=int(name.endswith("-V")))
        return None
