"""Code-space and CID range helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass

CodeSpaceRanges = list[tuple[bytes, bytes]] | tuple[tuple[bytes, bytes], ...]


def unicode_scalar_or_replacement(codepoint: int) -> str:
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return chr(codepoint)
    return "\ufffd"


def expand_range(start: int, end: int, source_hex_len: int, base_dst: str) -> dict[bytes, str]:
    mapping: dict[bytes, str] = {}
    if source_hex_len <= 0 or end >= 1 << (source_hex_len * 8):
        raise ValueError("invalid ToUnicode CMap bfrange")
    for i in range(start, end + 1):
        offset = i - start
        if not base_dst:
            mapping[i.to_bytes(source_hex_len, "big")] = ""
            continue
        units = [ord(c) for c in base_dst]
        units[-1] += offset
        mapping[i.to_bytes(source_hex_len, "big")] = "".join(
            unicode_scalar_or_replacement(u) for u in units
        )
    return mapping


@dataclass(frozen=True, slots=True)
class CIDRange:
    start: bytes
    end: bytes
    first_cid: int

    def contains(self, code: bytes) -> bool:
        return code_in_range(code, self.start, self.end)

    def cid_for(self, code: bytes) -> int:
        return self.first_cid + range_offset(code, self.start, self.end)


@dataclass(frozen=True, slots=True)
class NotdefRange:
    start: bytes
    end: bytes
    cid: int

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
