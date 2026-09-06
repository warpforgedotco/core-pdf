"""Literal UTF-16BE ToUnicode mappings and CMap inheritance."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from core_pdf.impl.spec.s_09_fonts.cmap_ranges import (
    MAX_CMAP_RANGE_SPAN,
    CodeSpaceRanges,
    code_in_ranges,
    ranges_overlap,
    validate_codespace_range,
)
from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import (
    CMapProgram,
    cmap_metadata,
    cmap_tokens,
    decode_cmap_hex_token,
    decode_cmap_token,
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
    if not start:
        raise ValueError("empty ToUnicode character code")
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
        internal_depth: int = 0,
        internal_empty: bool = False,
    ) -> None:
        self.code_space_ranges = []
        self.mappings = {}
        if internal_empty:
            self.decode_lengths = ()
            return
        if internal_depth > 16:
            raise ValueError("ToUnicode CMap UseCMap recursion limit exceeded")
        source = data if type(data) is bytes else bytes(data)
        parsed = self.parse_program(source)
        parent: ToUnicodeCMap | None = None
        parent_name = parsed.usecmap_name
        if parent_name is not None and usecmap_resolver is not None:
            parent_data = usecmap_resolver(parent_name)
            if parent_data is not None:
                parent = self.load_parent(parent_data, usecmap_resolver, internal_depth + 1)
        self.code_space_ranges = (
            tuple(parent.code_space_ranges if parent else ()) + parsed.code_space_ranges
        )
        self.mappings = dict(parent.mappings) if parent else {}
        self.mappings.update(parsed.mappings)
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

    def load_parent(
        self, data: bytes, resolver: Callable[[str], bytes | None], depth: int
    ) -> ToUnicodeCMap | None:
        return type(self)(data, usecmap_resolver=resolver, internal_depth=depth)

    def lookup(self, code: bytes) -> str | None:
        return self.mappings.get(code)


def parse_to_unicode_cmap(data: bytes) -> ParsedToUnicodeCMap:
    program = CMapProgram.parse(data)
    ranges: list[tuple[bytes, bytes]] = []
    mappings: dict[bytes, str] = {}
    for block in program.blocks(b"begincodespacerange", b"endcodespacerange"):
        values = block.token_values()
        if len(values) % 2:
            raise ValueError("invalid ToUnicode codespacerange")
        for index in range(0, len(values), 2):
            start, end = (decode_cmap_hex_token(value) for value in values[index : index + 2])
            validate_codespace_range(start, end)
            if any(ranges_overlap((start, end), previous) for previous in ranges):
                raise ValueError("overlapping ToUnicode codespacerange")
            ranges.append((start, end))
    for mapping_block in cmap_mapping_blocks(program):
        if mapping_block.trailing_operand_count:
            raise ValueError("invalid ToUnicode mapping operands")
        for record in mapping_block.records():
            start = decode_cmap_hex_token(record.source)
            if not start:
                raise ValueError("empty ToUnicode character code")
            if record.source_end is None:
                mappings[start] = decode_utf16be(decode_cmap_token(record.destination))
                continue
            source_range = cmap_source_range(start, decode_cmap_hex_token(record.source_end))
            if source_range.count > MAX_CMAP_RANGE_SPAN:
                raise ValueError("invalid ToUnicode bfrange")
            destination = record.destination
            if destination.startswith(b"["):
                destinations = cmap_tokens(destination)
                if len(destinations) != source_range.count:
                    raise ValueError("invalid ToUnicode bfrange destination array")
                texts = [decode_utf16be(decode_cmap_token(item)) for item in destinations]
            else:
                base = decode_cmap_token(destination)
                texts = []
                for offset in range(source_range.count):
                    incremented = (int.from_bytes(base, "big") + offset).to_bytes(len(base), "big")
                    texts.append(decode_utf16be(incremented))
            mappings.update(
                (source_range.source_at(offset), text) for offset, text in enumerate(texts)
            )
    parent = cmap_metadata(program)[0]
    if not ranges and parent is None:
        raise ValueError("missing ToUnicode codespacerange")
    if ranges and any(not code_in_ranges(code, ranges) for code in mappings):
        raise ValueError("ToUnicode mapping outside codespace")
    return ParsedToUnicodeCMap(tuple(ranges), mappings, parent)
