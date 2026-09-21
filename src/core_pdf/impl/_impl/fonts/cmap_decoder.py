"""Application CMap aliases around the PDF CMap decoder."""

from __future__ import annotations

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
from core_adobe_fonts.cmap.tokenizer import CMapBlock
from core_adobe_fonts.cmap.tokenizer import CMapProgram as PdfCMapProgram
from core_adobe_fonts.cmap.tokenizer import (
    decode_cmap_hex_token as decode_spec_cmap_hex_token,
)
from core_pdf.impl._impl.fonts.cmap_tokenizer import (
    CMapProgram,
    cmap_metadata,
    decode_cmap_hex_token,
)


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
        """Collect the `<code> cid` pairs from one character-mapping block."""
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
            except (ValueError, UnicodeDecodeError):
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
        """Collect range triples, dropping explicit codes the range supersedes."""
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
                # Also rejects empty or mismatched start/end lengths.
                validate_codespace_range(start_bytes, end_bytes)
            except (ValueError, UnicodeDecodeError):
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
