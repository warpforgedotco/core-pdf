"""CMap decoding for Unicode and CID fonts."""

from __future__ import annotations

import array
import sys
import re
import typing
from typing import TYPE_CHECKING

from core_pdf.fonts.encoding import BYTE_CACHE, decode_utf16be

if TYPE_CHECKING:
    from core_pdf.syntax.primitives import PdfStream

from core_pdf.syntax.primitives import MISSING

# Pre-compiled regex patterns
RE_CODESPACERANGE = re.compile(b"begincodespacerange(.*?)endcodespacerange", re.DOTALL)
RE_TOKENS = re.compile(b"<([0-9A-Fa-f]+)>")
RE_BFCHAR = re.compile(b"beginbfchar(.*?)endbfchar", re.DOTALL)
RE_ITEMS = re.compile(b"(<[0-9A-Fa-f]+>|\\([^)]*\\))")
RE_BFRANGE = re.compile(b"beginbfrange(.*?)endbfrange", re.DOTALL)
RE_RANGE_ITEMS = re.compile(b"(<[0-9A-Fa-f]+>|\\([^)]*\\)|\\[[^\\]]*\\])")
RE_DSTS = re.compile(b"(<[0-9A-Fa-f]+>|\\([^)]*\\))")

def expand_range(start: int, end: int, source_hex_len: int, base_dst: str) -> dict[bytes, str]:
    mapping: dict[bytes, str] = {}
    for i in range(start, end + 1):
        offset = i - start
        if not base_dst:
            mapping[i.to_bytes(source_hex_len, "big")] = ""
            continue
        units = [ord(c) for c in base_dst]
        units[-1] += offset
        mapping[i.to_bytes(source_hex_len, "big")] = "".join(chr(u) for u in units)
    return mapping


class ToUnicodeCMap:
    """Decodes string bytes to Unicode using a /ToUnicode CMap."""

    __slots__ = (
        "code_space_ranges",
        "mappings",
        "decode_lengths",
        "fast_decode_table",
        "fast_decode_table_2byte",
    )

    def __init__(self, stream: PdfStream) -> None:
        self.code_space_ranges: list[tuple[bytes, bytes]] = []
        self.mappings: dict[bytes, str] = {}
        self.fast_decode_table: list[str] | None = None
        self.fast_decode_table_2byte: list[str] | None = None

        data = stream.data
        self.parse_codespace_ranges(data)
        self.parse_bfchar(data)
        self.parse_bfrange(data)

        self.decode_lengths = sorted(
            ({len(end) for _, end in self.code_space_ranges} | {len(k) for k in self.mappings}) or {1},
            reverse=True,
        )
        self.precalculate_fast_tables()

    def parse_codespace_ranges(self, data: bytes) -> None:
        for block in RE_CODESPACERANGE.findall(data):
            tokens = RE_TOKENS.findall(block)
            if len(tokens) % 2 != 0:
                raise ValueError("invalid ToUnicode CMap codespacerange")
            for i in range(0, len(tokens), 2):
                try:
                    start = bytes.fromhex(tokens[i].decode())
                    end = bytes.fromhex(tokens[i + 1].decode())
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("invalid ToUnicode CMap codespacerange") from exc
                self.code_space_ranges.append((start, end))

    def parse_bfchar(self, data: bytes) -> None:
        for block in RE_BFCHAR.findall(data):
            items = RE_ITEMS.findall(block)
            if len(items) % 2 != 0:
                raise ValueError("invalid ToUnicode CMap bfchar")
            for i in range(0, len(items), 2):
                src_tok = items[i]
                dst_tok = items[i + 1]
                try:
                    src = (
                        bytes.fromhex(src_tok[1:-1].decode())
                        if src_tok.startswith(b"<")
                        else src_tok[1:-1]
                    )
                    dst = (
                        decode_utf16be(bytes.fromhex(dst_tok[1:-1].decode()))
                        if dst_tok.startswith(b"<")
                        else decode_utf16be(dst_tok[1:-1])
                    )
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("invalid ToUnicode CMap bfchar") from exc

                self.mappings[src] = dst
                if len(src) == 1:
                    self.mappings[b"\x00" + src] = dst
                elif len(src) == 2 and src[0] == 0:
                    self.mappings[src[1:]] = dst

    def parse_bfrange(self, data: bytes) -> None:
        for block in RE_BFRANGE.findall(data):
            items = RE_RANGE_ITEMS.findall(block)
            if not items or len(items) % 3 != 0:
                raise ValueError("invalid ToUnicode CMap bfrange")
            idx = 0
            while idx <= len(items) - 3:
                t1, t2, t3 = items[idx], items[idx + 1], items[idx + 2]
                if not (t1.startswith(b"<") and t2.startswith(b"<")):
                    raise ValueError("invalid ToUnicode CMap bfrange")
                try:
                    start_code = int(t1[1:-1].decode(), 16)
                    end_code = int(t2[1:-1].decode(), 16)
                    src_len = (len(t1) - 2) // 2
                except (ValueError, UnicodeDecodeError, IndexError) as exc:
                    raise ValueError("invalid ToUnicode CMap bfrange") from exc
                if start_code > end_code:
                    raise ValueError("invalid ToUnicode CMap bfrange")

                if t3.startswith(b"["):
                    dsts = RE_DSTS.findall(t3)
                    if not dsts:
                        raise ValueError("invalid ToUnicode CMap bfrange")
                    for i, dst_tok in enumerate(dsts):
                        if start_code + i > end_code:
                            break
                        try:
                            dst = (
                                decode_utf16be(bytes.fromhex(dst_tok[1:-1].decode()))
                                if dst_tok.startswith(b"<")
                                else decode_utf16be(dst_tok[1:-1])
                            )
                        except (ValueError, UnicodeDecodeError) as exc:
                            raise ValueError("invalid ToUnicode CMap bfrange") from exc
                        src = (start_code + i).to_bytes(src_len, "big")
                        self.mappings[src] = dst
                        if src_len == 1:
                            self.mappings[b"\x00" + src] = dst
                        elif src_len == 2 and src[0] == 0:
                            self.mappings[src[1:]] = dst
                    idx += 3
                elif t3.startswith(b"<") or t3.startswith(b"("):
                    try:
                        base_dst = (
                            decode_utf16be(t3[1:-1].decode()) if t3.startswith(b"<") else decode_utf16be(t3[1:-1])
                        )
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise ValueError("invalid ToUnicode CMap bfrange") from exc
                    m = expand_range(start_code, end_code, src_len, base_dst)
                    for k, v in m.items():
                        self.mappings[k] = v
                        if src_len == 1:
                            self.mappings[b"\x00" + k] = v
                        elif src_len == 2 and k[0] == 0:
                            self.mappings[k[1:]] = v
                    idx += 3
                else:
                    raise ValueError("invalid ToUnicode CMap bfrange")

    def precalculate_fast_tables(self) -> None:
        if 1 in self.decode_lengths:
            table = [MISSING] * 256
            for i in range(256):
                b = BYTE_CACHE[i]
                res = self.mappings.get(b)
                if res is None:
                    res = self.mappings.get(b"\x00" + b)
                table[i] = res if res is not None else ""
            self.fast_decode_table = typing.cast("list[str]", table)

        if 2 in self.decode_lengths:
            table2 = [""] * 65536
            for k, v in self.mappings.items():
                if len(k) == 2:
                    code = (k[0] << 8) | k[1]
                    table2[code] = v
            self.fast_decode_table_2byte = table2

    def decode(self, data: bytes) -> str:
        """Robust greedy decoding with multiple fallback attempts."""
        if not data:
            return ""

        # FAST PATH: single-byte mappings
        if self.fast_decode_table is not None and (not self.decode_lengths or self.decode_lengths == [1]):
            table = self.fast_decode_table
            result = "".join(map(table.__getitem__, data))
            return result

        # FAST PATH: two-byte mappings (Identity-H/V etc)
        n = len(data)
        if self.fast_decode_table_2byte is not None and self.decode_lengths == [2] and n % 2 == 0:
            table = self.fast_decode_table_2byte
            if n > 64:
                cids = array.array("H", data)
                if sys.byteorder == "little":
                    cids.byteswap()
                result = "".join(map(table.__getitem__, cids))
            else:
                # Faster manual loop for short strings
                out_small = []
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    out_small.append(table[code])
                result = "".join(out_small)

            if "\x00" in result:
                return result.replace("\x00", "")
            return result

        mappings = self.mappings
        lengths = self.decode_lengths or [1]
        n = len(data)
        out: list[str] = []
        out_append = out.append
        pos = 0
        bc = BYTE_CACHE

        mappings_get = mappings.get
        while pos < n:
            match_found = False
            for length in lengths:
                if pos + length > n:
                    continue

                # Use pre-allocated BYTE_CACHE for 1-byte chunks
                if length == 1:
                    chunk = bc[data[pos]]
                else:
                    chunk = data[pos : pos + length]

                result = mappings_get(chunk)
                if result is not None:
                    out_append(result)
                    pos += length
                    match_found = True
                    break

            if match_found:
                continue

            # Fallback 1: match current byte directly
            chunk1 = bc[data[pos]]
            result = mappings_get(chunk1)
            if result is not None:
                out_append(result)
                pos += 1
                continue

            # Fallback 2: match b'\x00' + current byte
            chunk01 = b"\x00" + chunk1
            result = mappings_get(chunk01)
            if result is not None:
                out_append(result)
                pos += 1
                continue

            # Fallback 3: CID
            if n - pos >= 2:
                cid = (data[pos] << 8) | data[pos + 1]
                pos += 2
                if cid < 0x110000 and cid != 0:
                    out_append(chr(cid))
                else:
                    out_append("\ufffd")
            else:
                out_append(chr(data[pos]))
                pos += 1

        result = "".join(out)
        if "\x00" in result:
            return result.replace("\x00", "")
        return result


class CMapDecoder:
    """Decodes string bytes using a predefined or embedded CMap."""

    __slots__ = ("code_space_ranges",)

    def __init__(self, stream: PdfStream) -> None:
        self.code_space_ranges: list[tuple[bytes, bytes]] = []
        for block in RE_CODESPACERANGE.findall(stream.data):
            tokens = RE_TOKENS.findall(block)
            if len(tokens) % 2 != 0:
                raise ValueError("invalid CMap codespacerange")
            for i in range(0, len(tokens), 2):
                try:
                    start = bytes.fromhex(tokens[i].decode())
                    end = bytes.fromhex(tokens[i + 1].decode())
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("invalid CMap codespacerange") from exc
                self.code_space_ranges.append((start, end))
