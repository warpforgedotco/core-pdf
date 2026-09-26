# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from typing import ClassVar

from core_records import Record, frozen_setattr

CodeSpaceRanges = list[tuple[bytes, bytes]] | tuple[tuple[bytes, bytes], ...]


class CIDRange(Record):
    __slots__ = ("start", "end", "first_cid")

    start: bytes
    end: bytes
    first_cid: int

    __fields__: ClassVar[tuple[str, ...]] = ("start", "end", "first_cid")
    __match_args__ = ("start", "end", "first_cid")

    def __init__(self, start: bytes, end: bytes, first_cid: int) -> None:
        frozen_setattr(self, "start", start)
        frozen_setattr(self, "end", end)
        frozen_setattr(self, "first_cid", first_cid)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.first_cid == other.first_cid
        )

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.first_cid))

    def contains(self, code: bytes) -> bool:
        return code_in_range(code, self.start, self.end)


class NotdefRange(Record):
    __slots__ = ("start", "end", "cid")

    start: bytes
    end: bytes
    cid: int

    __fields__: ClassVar[tuple[str, ...]] = ("start", "end", "cid")
    __match_args__ = ("start", "end", "cid")

    def __init__(self, start: bytes, end: bytes, cid: int) -> None:
        frozen_setattr(self, "start", start)
        frozen_setattr(self, "end", end)
        frozen_setattr(self, "cid", cid)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.start == other.start and self.end == other.end and self.cid == other.cid

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.cid))

    def contains(self, code: bytes) -> bool:
        return code_in_range(code, self.start, self.end)


def code_in_range(code: bytes, start: bytes, end: bytes) -> bool:
    length = len(code)
    if length != len(start) or length != len(end):
        return False
    if length == 1:
        return start[0] <= code[0] <= end[0]
    if length == 2:
        return start[0] <= code[0] <= end[0] and start[1] <= code[1] <= end[1]
    for code_byte, start_byte, end_byte in zip(code, start, end, strict=True):
        if not start_byte <= code_byte <= end_byte:
            return False
    return True


def code_in_ranges(code: bytes, ranges: typing.Iterable[tuple[bytes, bytes]]) -> bool:
    return any(code_in_range(code, start, end) for start, end in ranges)


def ranges_overlap(
    left: tuple[bytes, bytes],
    right: tuple[bytes, bytes],
) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    return len(left_start) == len(right_start) and all(
        left_start_byte <= right_end_byte and right_start_byte <= left_end_byte
        for left_start_byte, left_end_byte, right_start_byte, right_end_byte in zip(
            left_start, left_end, right_start, right_end
        )
    )


def validate_codespace_range(start: bytes, end: bytes) -> None:
    if not start or len(start) != len(end):
        raise ValueError("invalid CMap range")
    if any(start_byte > end_byte for start_byte, end_byte in zip(start, end)):
        raise ValueError("invalid CMap range")


def validate_effective_codespace(ranges: CodeSpaceRanges, codes: typing.Iterable[bytes]) -> None:
    if not ranges:
        raise ValueError("missing CMap codespacerange")
    for index, (start, end) in enumerate(ranges):
        validate_codespace_range(start, end)
        if len(start) > 4:
            raise ValueError("invalid PDF CMap character code length")
        if any(ranges_overlap((start, end), previous) for previous in ranges[:index]):
            raise ValueError("overlapping CMap codespacerange")
    if any(not code_in_ranges(code, ranges) for code in codes):
        raise ValueError("CMap mapping outside codespace")


def range_offset(
    code: bytes,
    start: bytes,
    end: bytes,
    *,
    validate_range: bool = True,
    validate_code: bool = True,
) -> int:
    if validate_range:
        validate_codespace_range(start, end)
    if validate_code and not code_in_range(code, start, end):
        raise ValueError("code outside CMap range")
    if len(code) == 1:
        return code[0] - start[0]
    if len(code) == 2:
        return (code[0] - start[0]) * (end[1] - start[1] + 1) + code[1] - start[1]
    offset = 0
    stride = 1
    for code_byte, start_byte, end_byte in reversed(tuple(zip(code, start, end))):
        offset += (code_byte - start_byte) * stride
        stride *= end_byte - start_byte + 1
    return offset


def iter_codespace_range(start: bytes, end: bytes) -> typing.Iterator[bytes]:
    validate_codespace_range(start, end)

    current = bytearray(start)
    while True:
        yield bytes(current)
        index = len(current) - 1
        while index >= 0:
            next_byte = current[index] + 1
            if next_byte <= end[index]:
                current[index] = next_byte
                break
            current[index] = start[index]
            index -= 1
        if index < 0:
            return


def remove_codes_in_range(mapping: dict[bytes, int], start: bytes, end: bytes) -> None:
    if (
        start
        and len(start) == len(end)
        and all(start_byte <= end_byte for start_byte, end_byte in zip(start, end))
    ):
        range_size = 1
        for start_byte, end_byte in zip(start, end):
            range_size *= end_byte - start_byte + 1
        if range_size <= len(mapping):
            for code in iter_codespace_range(start, end):
                mapping.pop(code, None)
            return
    for code in tuple(code for code in mapping if len(code) == len(start)):
        if code_in_range(code, start, end):
            del mapping[code]


__all__ = [
    "CodeSpaceRanges",
    "CIDRange",
    "NotdefRange",
    "code_in_range",
    "code_in_ranges",
    "ranges_overlap",
    "validate_codespace_range",
    "range_offset",
    "iter_codespace_range",
    "remove_codes_in_range",
    "validate_effective_codespace",
]
