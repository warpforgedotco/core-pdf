# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from collections.abc import Iterator

HEX_BYTES = frozenset(b"0123456789abcdefABCDEF \t\r\n")


def decrypt_type1(data: bytes, key: int) -> bytes:
    output = bytearray(len(data))
    state = key
    for index, cipher in enumerate(data):
        output[index] = cipher ^ (state >> 8)
        state = ((cipher + state) * 52845 + 22719) & 0xFFFF
    return bytes(output)


def decode_eexec_payload(data: bytes, length1: int | None) -> bytes:
    if length1 is not None:
        if not 0 < length1 < len(data):
            raise ValueError("invalid Type 1 Length1")
        encrypted = data[length1:]
    else:
        marker = data.find(b"currentfile eexec")
        if marker < 0:
            raise ValueError("Type 1 eexec section is missing")
        encrypted = data[marker + len(b"currentfile eexec") :].lstrip()
    sample = encrypted[:4]
    if sample and all(byte in HEX_BYTES for byte in sample):
        compact = bytes(byte for byte in encrypted if byte not in b" \t\r\n")
        if len(compact) % 2:
            raise ValueError("odd Type 1 hexadecimal eexec payload")
        try:
            encrypted = bytes.fromhex(compact.decode("ascii"))
        except ValueError as exc:
            raise ValueError("invalid hexadecimal Type 1 eexec section") from exc
    decrypted = decrypt_type1(encrypted, 55665)
    if len(decrypted) < 4:
        raise ValueError("truncated Type 1 eexec section")
    return decrypted[4:]


def binary_entries(data: bytes, pattern: re.Pattern[bytes]) -> Iterator[tuple[bytes, bytes]]:
    for match in pattern.finditer(data):
        length = int(match.group(2))
        start = match.end()
        end = start + length
        if length < 0 or end > len(data):
            raise ValueError("truncated Type 1 binary entry")
        yield match.group(1), data[start:end]


def decode_charstring(encrypted: bytes, len_iv: int) -> bytes:
    if len_iv < -1:
        raise ValueError("invalid Type 1 lenIV")
    if len_iv == -1:
        return encrypted
    if len(encrypted) < len_iv:
        raise ValueError("truncated Type 1 charstring prefix")
    return decrypt_type1(encrypted, 4330)[len_iv:]


TYPE1_ENCODING_ENTRY_RE = re.compile(rb"\bdup\s+(\d{1,3})\s+/([A-Za-z0-9_.]+)\s+put\b")


def parse_type1_font_program_encoding(font_program: bytes | memoryview) -> dict[int, str]:
    data = bytes(font_program)
    eexec_pos = data.find(b"currentfile eexec")
    if eexec_pos >= 0:
        data = data[:eexec_pos]

    differences: dict[int, str] = {}
    for match in TYPE1_ENCODING_ENTRY_RE.finditer(data):
        code = int(match.group(1))
        if code > 255:
            raise ValueError("Type 1 encoding code outside byte range")
        differences[code] = match.group(2).decode("latin-1")
    return differences


__all__ = [
    "decrypt_type1",
    "decode_eexec_payload",
    "binary_entries",
    "decode_charstring",
    "TYPE1_ENCODING_ENTRY_RE",
    "parse_type1_font_program_encoding",
]
