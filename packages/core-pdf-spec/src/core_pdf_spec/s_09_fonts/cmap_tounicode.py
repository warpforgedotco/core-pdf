"""Literal UTF-16BE ToUnicode mappings and CMap inheritance."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from core_pdf_spec.s_09_fonts.cmap_ranges import (
    CodeSpaceRanges,
    internal_validate_pdf_codespace,
)
from core_pdf_spec.s_09_fonts.cmap_tokenizer import (
    CMapProgram,
    cmap_metadata,
    cmap_tokens,
    decode_cmap_hex_token,
)


def decode_utf16be(data: bytes) -> str:
    """ToUnicode destinations are UTF-16BE strings, including literal U+0000."""
    return data.decode("utf-16-be")


@dataclass(frozen=True, slots=True)
class ParsedToUnicodeCMap:
    code_space_ranges: tuple[tuple[bytes, bytes], ...]
    mappings: dict[bytes, str]
    usecmap_name: str | None


@dataclass(frozen=True, slots=True)
class CMapMappingRecord:
    """Literal operands of one character or range mapping, before decoding."""

    source: bytes
    destination: bytes
    source_end: bytes | None = None


@dataclass(frozen=True, slots=True)
class CMapMappingBlock:
    operator: bytes
    operands: tuple[bytes, ...]
    stride: int

    @property
    def trailing_operand_count(self) -> int:
        return len(self.operands) % self.stride

    def records(self) -> Iterator[CMapMappingRecord]:
        for index in range(0, len(self.operands) - self.trailing_operand_count, self.stride):
            if self.stride == 2:
                yield CMapMappingRecord(self.operands[index], self.operands[index + 1])
            else:
                yield CMapMappingRecord(
                    self.operands[index], self.operands[index + 2], self.operands[index + 1]
                )


def cmap_mapping_blocks(
    program: CMapProgram, *, include_cid_ranges: bool = False
) -> Iterator[CMapMappingBlock]:
    """Traverse raw mapping records in source order without choosing error recovery.

    Numeric CID ranges are valid encoding-CMap syntax. A ToUnicode compiler
    selects whether to inspect those records in addition to its bf mappings.
    Trailing operands remain visible so callers can reject or recover them.
    """
    delimiters = {b"beginbfchar": b"endbfchar", b"beginbfrange": b"endbfrange"}
    if include_cid_ranges:
        delimiters[b"begincidrange"] = b"endcidrange"
    for operator, block in program.blocks_in_order(delimiters):
        yield CMapMappingBlock(
            operator,
            tuple(
                block.token_values(
                    include_arrays=operator == b"beginbfrange",
                    include_words=operator == b"begincidrange",
                )
            ),
            2 if operator == b"beginbfchar" else 3,
        )


@dataclass(frozen=True, slots=True)
class CMapSourceRange:
    first: int
    last: int
    width: int

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    def source_at(self, offset: int) -> bytes:
        return (self.first + offset).to_bytes(self.width, "big")


def cmap_source_range(start: bytes, end: bytes) -> CMapSourceRange:
    """Validate decoded range endpoints without selecting an expansion limit."""
    if not 1 <= len(start) <= 4:
        raise ValueError("invalid ToUnicode character code length")
    first, last = int.from_bytes(start, "big"), int.from_bytes(end, "big")
    if len(start) != len(end) or first > last:
        raise ValueError("invalid ToUnicode bfrange")
    return CMapSourceRange(first, last, len(start))


class ToUnicodeCMap:
    code_space_ranges: CodeSpaceRanges
    mappings: dict[bytes, str]
    decode_lengths: tuple[int, ...]
    __slots__ = ("code_space_ranges", "mappings", "decode_lengths")

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: Callable[[str], bytes | None] | None = None,
        inheritance_depth: int = 0,
        empty: bool = False,
        ancestor_names: tuple[str, ...] = (),
    ) -> None:
        self.code_space_ranges = []
        self.mappings = {}
        if empty:
            self.decode_lengths = ()
            return
        source = data if type(data) is bytes else bytes(data)
        parsed = self.parse_program(source)
        parent: ToUnicodeCMap | None = None
        parent_name = parsed.usecmap_name
        if parent_name is not None:
            parent = self.resolve_parent(
                parent_name, usecmap_resolver, inheritance_depth + 1, ancestor_names
            )
        self.code_space_ranges = (
            tuple(parent.code_space_ranges if parent else ()) + parsed.code_space_ranges
        )
        self.mappings = dict(parent.mappings) if parent else {}
        self.mappings.update(parsed.mappings)
        self.validate_mappings()
        self.decode_lengths = tuple(
            sorted(
                length
                for length in (
                    {len(end) for _, end in self.code_space_ranges}
                    | {len(k) for k in self.mappings}
                )
                if length > 0
            )
            or {1}
        )

    @staticmethod
    def parse_program(data: bytes) -> ParsedToUnicodeCMap:
        return parse_to_unicode_cmap(data)

    def validate_mappings(self) -> None:
        """Validate the complete codespace and mappings after parent resolution.

        Recovery adapters may override this completion step for recovered maps.
        """
        internal_validate_pdf_codespace(self.code_space_ranges, self.mappings)

    def resolve_parent(
        self,
        name: str,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> ToUnicodeCMap | None:
        """Load a required parent; unavailable or cyclic links raise ValueError."""
        if name in ancestor_names:
            raise ValueError("cyclic ToUnicode CMap usecmap")
        data = resolver(name) if resolver is not None else None
        if data is None:
            raise ValueError(f"unresolved ToUnicode CMap usecmap: {name}")
        return self.load_parent(data, resolver, depth, (*ancestor_names, name))

    def load_parent(
        self,
        data: bytes,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> ToUnicodeCMap | None:
        return type(self)(
            data, usecmap_resolver=resolver, inheritance_depth=depth, ancestor_names=ancestor_names
        )

    def lookup(self, code: bytes) -> str | None:
        return self.mappings.get(code)


def parse_to_unicode_cmap(data: bytes) -> ParsedToUnicodeCMap:
    """Parse local definitions, deferring inherited codespace checks to resolution."""
    program = CMapProgram.parse(data)
    ranges: list[tuple[bytes, bytes]] = []
    mappings: dict[bytes, str] = {}
    for block in program.blocks(b"begincodespacerange", b"endcodespacerange"):
        values = block.token_values()
        if len(values) % 2:
            raise ValueError("invalid ToUnicode codespacerange")
        for index in range(0, len(values), 2):
            start, end = (decode_cmap_hex_token(value) for value in values[index : index + 2])
            ranges.append((start, end))
    for mapping_block in cmap_mapping_blocks(program):
        if mapping_block.trailing_operand_count:
            raise ValueError("invalid ToUnicode mapping operands")
        for record in mapping_block.records():
            start = decode_cmap_hex_token(record.source)
            if not start:
                raise ValueError("empty ToUnicode character code")
            if record.source_end is None:
                mappings[start] = decode_utf16be(decode_cmap_hex_token(record.destination))
                continue
            source_range = cmap_source_range(start, decode_cmap_hex_token(record.source_end))
            destination = record.destination
            if destination.startswith(b"["):
                destinations = cmap_tokens(destination)
                if len(destinations) != source_range.count:
                    raise ValueError("invalid ToUnicode bfrange destination array")
                texts = [decode_utf16be(decode_cmap_hex_token(item)) for item in destinations]
            else:
                base = decode_cmap_hex_token(destination)
                texts = []
                for offset in range(source_range.count):
                    incremented = (int.from_bytes(base, "big") + offset).to_bytes(len(base), "big")
                    texts.append(decode_utf16be(incremented))
            mappings.update(
                (source_range.source_at(offset), text) for offset, text in enumerate(texts)
            )
    parent = cmap_metadata(program)[0]
    if parent is None:
        internal_validate_pdf_codespace(ranges, mappings)
    return ParsedToUnicodeCMap(tuple(ranges), mappings, parent)


__all__ = [
    "decode_utf16be",
    "ParsedToUnicodeCMap",
    "CMapMappingRecord",
    "CMapMappingBlock",
    "cmap_mapping_blocks",
    "CMapSourceRange",
    "cmap_source_range",
    "ToUnicodeCMap",
    "parse_to_unicode_cmap",
]
