# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, NoReturn, Self

from core_adobe_fonts.cmap.ranges import (
    CodeSpaceRanges,
    validate_effective_codespace,
)
from core_adobe_fonts.cmap.tokenizer import (
    CMapProgram,
    cmap_metadata,
    cmap_tokens,
    decode_cmap_hex_token,
)

frozen_setattr = object.__setattr__


def decode_utf16be(data: bytes) -> str:
    return data.decode("utf-16-be")


class ParsedToUnicodeCMap:
    __slots__ = ("code_space_ranges", "mappings", "usecmap_name")

    code_space_ranges: tuple[tuple[bytes, bytes], ...]
    mappings: dict[bytes, str]
    usecmap_name: str | None

    __fields__: ClassVar[tuple[str, ...]] = ("code_space_ranges", "mappings", "usecmap_name")
    __match_args__ = ("code_space_ranges", "mappings", "usecmap_name")

    def __init__(
        self,
        code_space_ranges: tuple[tuple[bytes, bytes], ...],
        mappings: dict[bytes, str],
        usecmap_name: str | None,
    ) -> None:
        frozen_setattr(self, "code_space_ranges", code_space_ranges)
        frozen_setattr(self, "mappings", mappings)
        frozen_setattr(self, "usecmap_name", usecmap_name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"code_space_ranges={self.code_space_ranges!r}, "
            f"mappings={self.mappings!r}, "
            f"usecmap_name={self.usecmap_name!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code_space_ranges == other.code_space_ranges
            and self.mappings == other.mappings
            and self.usecmap_name == other.usecmap_name
        )

    def __hash__(self) -> int:
        return hash((self.code_space_ranges, self.mappings, self.usecmap_name))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        code_space_ranges = changes.pop("code_space_ranges", self.code_space_ranges)
        mappings = changes.pop("mappings", self.mappings)
        usecmap_name = changes.pop("usecmap_name", self.usecmap_name)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(code_space_ranges, mappings, usecmap_name)


class CMapMappingRecord:
    __slots__ = ("source", "destination", "source_end")

    source: bytes
    destination: bytes
    source_end: bytes | None

    __fields__: ClassVar[tuple[str, ...]] = ("source", "destination", "source_end")
    __match_args__ = ("source", "destination", "source_end")

    def __init__(self, source: bytes, destination: bytes, source_end: bytes | None = None) -> None:
        frozen_setattr(self, "source", source)
        frozen_setattr(self, "destination", destination)
        frozen_setattr(self, "source_end", source_end)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"source={self.source!r}, "
            f"destination={self.destination!r}, "
            f"source_end={self.source_end!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.source == other.source
            and self.destination == other.destination
            and self.source_end == other.source_end
        )

    def __hash__(self) -> int:
        return hash((self.source, self.destination, self.source_end))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        source = changes.pop("source", self.source)
        destination = changes.pop("destination", self.destination)
        source_end = changes.pop("source_end", self.source_end)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(source, destination, source_end)


class CMapMappingBlock:
    __slots__ = ("operator", "operands", "stride")

    operator: bytes
    operands: tuple[bytes, ...]
    stride: int

    __fields__: ClassVar[tuple[str, ...]] = ("operator", "operands", "stride")
    __match_args__ = ("operator", "operands", "stride")

    def __init__(self, operator: bytes, operands: tuple[bytes, ...], stride: int) -> None:
        frozen_setattr(self, "operator", operator)
        frozen_setattr(self, "operands", operands)
        frozen_setattr(self, "stride", stride)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"operator={self.operator!r}, "
            f"operands={self.operands!r}, "
            f"stride={self.stride!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.operator == other.operator
            and self.operands == other.operands
            and self.stride == other.stride
        )

    def __hash__(self) -> int:
        return hash((self.operator, self.operands, self.stride))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        operator = changes.pop("operator", self.operator)
        operands = changes.pop("operands", self.operands)
        stride = changes.pop("stride", self.stride)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(operator, operands, stride)

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


class CMapSourceRange:
    __slots__ = ("first", "last", "width")

    first: int
    last: int
    width: int

    __fields__: ClassVar[tuple[str, ...]] = ("first", "last", "width")
    __match_args__ = ("first", "last", "width")

    def __init__(self, first: int, last: int, width: int) -> None:
        frozen_setattr(self, "first", first)
        frozen_setattr(self, "last", last)
        frozen_setattr(self, "width", width)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"first={self.first!r}, "
            f"last={self.last!r}, "
            f"width={self.width!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.first == other.first and self.last == other.last and self.width == other.width

    def __hash__(self) -> int:
        return hash((self.first, self.last, self.width))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        first = changes.pop("first", self.first)
        last = changes.pop("last", self.last)
        width = changes.pop("width", self.width)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(first, last, width)

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    def source_at(self, offset: int) -> bytes:
        return (self.first + offset).to_bytes(self.width, "big")


def cmap_source_range(start: bytes, end: bytes) -> CMapSourceRange:
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
        ancestor_names: tuple[str, ...] = (),
    ) -> None:
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
        validate_effective_codespace(self.code_space_ranges, self.mappings)

    def resolve_parent(
        self,
        name: str,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> ToUnicodeCMap | None:
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
        validate_effective_codespace(ranges, mappings)
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
