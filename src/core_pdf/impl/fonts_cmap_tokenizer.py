from __future__ import annotations

import re

from core_adobe_fonts.cmap.decoder import (
    CMapDecoder as PdfCMapDecoder,
)
from core_adobe_fonts.cmap.decoder import CMapResourceResolver
from core_adobe_fonts.cmap.tokenizer import (
    CMapBlock,
    CMapToken,
    iter_cmap_tokens,
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


def collect_cmap_tokens(data: bytes, *, group_arrays: bool) -> tuple[CMapToken, ...]:
    """Tokenise as far as the source allows; a malformed tail yields what came before."""
    tokens: list[CMapToken] = []
    try:
        for token in iter_cmap_tokens(data, group_arrays=group_arrays):
            tokens.append(token)  # noqa: PERF402
    except ValueError:
        pass
    return tuple(tokens)


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
    """A CMap read as far as it goes: no program, count, or operand rules apply."""

    __slots__ = ()

    @classmethod
    def read_tokens(cls, source: bytes) -> tuple[CMapToken, ...]:
        return collect_cmap_tokens(source, group_arrays=True)

    @classmethod
    def validate_program(
        cls, tokens: tuple[CMapToken, ...], scoped_tokens: tuple[CMapToken, ...]
    ) -> None:
        pass

    def block_count(self, begin_index: int) -> int:  # noqa: ARG002
        return 0

    def validate_block(
        self, begin_keyword: bytes, block_tokens: list[CMapToken], declared_count: int
    ) -> None:
        pass

    def reject_unterminated_block(self) -> None:
        pass


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
    tokens = collect_cmap_tokens(data, group_arrays=include_arrays)
    return CMapBlock(data, tokens).token_values(
        include_arrays=include_arrays, include_words=include_words
    )


class CMapDecoder(PdfCMapDecoder):
    __slots__ = ()

    max_inheritance_depth = 5

    @staticmethod
    def parse_program(data: bytes) -> CMapProgram:
        return CMapProgram.parse(data)

    @staticmethod
    def program_metadata(program: PdfCMapProgram) -> tuple[str | None, int | None]:
        return cmap_metadata(program)

    def validate_mappings(self) -> None:
        pass

    def reject_mapping(self, reason: str, cause: BaseException | None = None) -> None:
        pass

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
