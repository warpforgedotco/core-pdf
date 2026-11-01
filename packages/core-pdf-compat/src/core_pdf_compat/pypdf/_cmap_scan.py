# Follows pypdf._cmap, Copyright (c) 2006, Mathieu Fenniak, under its BSD-style license.
"""pypdf's size limit and warnings for a /ToUnicode CMap. core-pdf does the decoding."""

from __future__ import annotations

import logging
from binascii import Error as BinasciiError
from binascii import unhexlify

from core_pdf_compat.pypdf.errors import LimitReachedError

logger = logging.getLogger(__name__)

MAPPING_LIMIT = 100_000


def _prepared(data: bytes) -> bytes:
    prepared = data.strip()
    for keyword in (b"beginbfchar", b"endbfchar", b"beginbfrange", b"endbfrange"):
        prepared = prepared.replace(keyword, b"\n" + keyword + b"\n")
    parts = prepared.replace(b"<<", b"\n{\n").replace(b">>", b"\n}\n").split(b"<")
    for index, part in enumerate(parts):
        end = part.find(b">")
        if end >= 0:
            content = part[:end].replace(b" ", b"") if end else b"."
            parts[index] = content + b" " + part[end + 1 :]
    joined = b" ".join(parts)
    return joined.replace(b"[", b" [ ").replace(b"]", b" ]\n ").replace(b"\r", b"\n")


def _check_size(size: int) -> None:
    if size > MAPPING_LIMIT:
        raise LimitReachedError(f"Maximum /ToUnicode size limit reached: {size} > {MAPPING_LIMIT}.")


class _Scan:
    def __init__(self) -> None:
        self.entries = 0
        self.open_list = False

    def add_listed(self, tokens: list[bytes]) -> bool:
        for token in tokens:
            if token == b"]":
                return True
            self.entries += 1
            _check_size(self.entries)
            unhexlify(token)
        return False

    def range_line(self, line: bytes) -> None:
        tokens = line.split()
        _check_size(self.entries)
        if self.open_list:
            self.open_list = not self.add_listed(tokens)
            return
        first, last = int(tokens[0], 16), int(tokens[1], 16)
        if tokens[2] == b"[":
            self.open_list = not self.add_listed(tokens[3:])
        else:
            int(tokens[2], 16)
            size = max(0, last - first + 1)
            _check_size(self.entries + size)
            self.entries += size

    def char_line(self, line: bytes) -> None:
        tokens = [token for token in line.split(b" ") if token]
        _check_size(self.entries + len(tokens) // 2)
        for source, target in zip(tokens[::2], tokens[1::2], strict=False):
            if target != b".":
                try:
                    unhexlify(target)
                except BinasciiError as error:
                    logger.warning("Got invalid hex string: %s (%r)", error, target)
            unhexlify(source)
            self.entries += 1


def scan_to_unicode(data: bytes) -> None:
    scan = _Scan()
    in_range = in_char = False
    for raw_line in _prepared(data).split(b"\n"):
        line = raw_line.strip(b" \t")
        if not line or line.startswith(b"%"):
            continue
        line = line.replace(b"\t", b" ")
        if b"beginbfrange" in line:
            in_range = True
        elif b"endbfrange" in line:
            in_range = False
        elif b"beginbfchar" in line:
            in_char = True
        elif b"endbfchar" in line:
            in_char = False
        elif in_range or in_char:
            try:
                scan.range_line(line) if in_range else scan.char_line(line)
            except (ValueError, IndexError) as error:
                logger.warning("Skipping broken line %r: %s", line, error)
