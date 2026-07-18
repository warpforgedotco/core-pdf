from __future__ import annotations

import array
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from core_cmap.impl.cid.encoding import BYTE_CACHE, decode_utf16be
from core_cmap.impl.cid.pdf_string import decode_pdf_literal_string

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


HEX_BYTES = bytes([1 if byte in b"0123456789abcdefABCDEF" else 0 for byte in range(256)])
PDF_WHITESPACE_BYTES = bytes([1 if byte in b"\x00\t\n\f\r " else 0 for byte in range(256)])


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


def iter_blocks(data: bytes | memoryview, begin: bytes, end: bytes) -> typing.Iterator[bytes]:
    if not isinstance(data, bytes):
        data = bytes(data)
    block_start: int | None = None
    for word, start, stop in cmap_word_spans(data):
        if block_start is None:
            if word == begin:
                block_start = stop
        elif word == end:
            yield data[block_start:start]
            block_start = None


def skip_cmap_literal_string(data: bytes, pos: int) -> int:
    end = pos + 1
    depth = 1
    n = len(data)
    while end < n and depth:
        current = data[end]
        if current == 92:
            end += 2
            continue
        if current == 40:
            depth += 1
        elif current == 41:
            depth -= 1
        end += 1
    return end


def skip_cmap_array(data: bytes, pos: int) -> int:
    end = pos + 1
    depth = 1
    n = len(data)
    while end < n and depth:
        current = data[end]
        if current == 37:
            while end < n and data[end] not in (10, 13):
                end += 1
            continue
        if current == 40:
            end = skip_cmap_literal_string(data, end)
            continue
        if current == 60:
            close = data.find(b">", end + 1)
            if close < 0:
                return n
            end = close + 1
            continue
        if current == 91:
            depth += 1
        elif current == 93:
            depth -= 1
        end += 1
    return end


def cmap_word_spans(data: bytes) -> typing.Iterator[tuple[bytes, int, int]]:
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if byte == 37:
            while pos < n and data[pos] not in (10, 13):
                pos += 1
            continue
        if byte == 40:
            pos = skip_cmap_literal_string(data, pos)
            continue
        if byte == 60:
            if pos + 1 < n and data[pos + 1] == 60:
                pos += 2
                continue
            close = data.find(b">", pos + 1)
            if close < 0:
                return
            pos = close + 1
            continue
        if byte == 91:
            pos = skip_cmap_array(data, pos)
            continue
        if PDF_WHITESPACE_BYTES[byte] or byte in b"[]<>()/%":
            pos += 1
            continue
        start = pos
        pos += 1
        while pos < n and not PDF_WHITESPACE_BYTES[data[pos]] and data[pos] not in b"[]<>()/%":
            pos += 1
        yield data[start:pos], start, pos


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
    tokens: list[bytes] = []
    pos = 0
    n = len(data)
    while pos < n:
        byte = data[pos]
        if byte == 37:
            while pos < n and data[pos] not in (10, 13):
                pos += 1
            continue
        if byte == 60:
            if pos + 1 < n and data[pos + 1] == 60:
                pos += 2
                continue
            end = data.find(b">", pos + 1)
            if end < 0:
                break
            candidate = data[pos : end + 1]
            tokens.append(candidate)
            pos = end + 1
            continue
        if byte == 40:
            end = pos + 1
            depth = 1
            while end < n and depth:
                current = data[end]
                if current == 92:
                    end += 2
                    continue
                if current == 40:
                    depth += 1
                elif current == 41:
                    depth -= 1
                end += 1
            if depth == 0:
                tokens.append(data[pos:end])
            pos = end
            continue
        if include_arrays and byte == 91:
            end = pos + 1
            depth = 1
            while end < n and depth:
                current = data[end]
                if current == 37:
                    while end < n and data[end] not in (10, 13):
                        end += 1
                    continue
                if current == 40:
                    end += 1
                    string_depth = 1
                    while end < n and string_depth:
                        current = data[end]
                        if current == 92:
                            end += 2
                            continue
                        if current == 40:
                            string_depth += 1
                        elif current == 41:
                            string_depth -= 1
                        end += 1
                    continue
                if current == 60:
                    close = data.find(b">", end + 1)
                    if close < 0:
                        break
                    end = close + 1
                    continue
                if current == 91:
                    depth += 1
                elif current == 93:
                    depth -= 1
                end += 1
            if depth == 0:
                tokens.append(data[pos:end])
                pos = end
                continue
        if include_words and not PDF_WHITESPACE_BYTES[byte]:
            end = pos + 1
            while end < n and not PDF_WHITESPACE_BYTES[data[end]] and data[end] not in b"[]<>()/%":
                end += 1
            tokens.append(data[pos:end])
            pos = end
            continue
        pos += 1
    return tokens


def cmap_noncomment_words(data: bytes) -> list[bytes]:
    return cmap_tokens(data, include_words=True)


def cmap_usecmap_name(data: bytes) -> str | None:
    words = cmap_noncomment_words(data)
    for index, word in enumerate(words[1:], start=1):
        if word != b"usecmap":
            continue
        name = words[index - 1]
        if not name.startswith(b"/"):
            continue
        try:
            return name[1:].decode("latin-1")
        except UnicodeDecodeError:
            return None
    return None


def cmap_wmode(data: bytes) -> int | None:
    words = cmap_noncomment_words(data)
    for index, word in enumerate(words[:-2]):
        if word == b"/WMode" and words[index + 2] == b"def":
            try:
                value = int(words[index + 1])
            except ValueError:
                return None
            return value if value in {0, 1} else None
    return None


def decode_cmap_hex_token(token: bytes) -> bytes:
    raw = token[1:-1].translate(None, b"\x00\t\n\f\r ")
    if not all(HEX_BYTES[item] for item in raw):
        raise ValueError("invalid CMap hex string")
    if len(raw) & 1:
        raw += b"0"
    return bytes.fromhex(raw.decode("ascii"))


def decode_cmap_token(token: bytes) -> bytes:
    if token.startswith(b"<"):
        return decode_cmap_hex_token(token)
    try:
        return decode_pdf_literal_string(token)
    except ValueError as exc:
        raise ValueError("invalid CMap literal string") from exc


def code_in_range(code: bytes, start: bytes, end: bytes) -> bool:
    return len(code) == len(start) == len(end) and all(
        start_byte <= code_byte <= end_byte
        for code_byte, start_byte, end_byte in zip(code, start, end)
    )


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


def range_offset(code: bytes, start: bytes, end: bytes) -> int:
    validate_codespace_range(start, end)
    if not code_in_range(code, start, end):
        raise ValueError("code outside CMap range")
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
    for code in tuple(mapping):
        if code_in_range(code, start, end):
            del mapping[code]


class ToUnicodeCMap:
    code_space_ranges: CodeSpaceRanges
    mappings: dict[bytes, str]
    decode_lengths: tuple[int, ...]
    fast_decode_table: list[str] | tuple[str, ...] | None
    fast_decode_table_2byte: list[str] | None

    __slots__ = (
        "code_space_ranges",
        "mappings",
        "decode_lengths",
        "fast_decode_table",
        "fast_decode_table_2byte",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: Callable[[str], bytes | None] | None = None,
        _depth: int = 0,
    ) -> None:
        if _depth > 16:
            raise ValueError("ToUnicode CMap UseCMap recursion limit exceeded")
        parsed = _parse_to_unicode_cmap(bytes(data))
        parent: ToUnicodeCMap | None = None
        parent_name = cmap_usecmap_name(bytes(data))
        if parent_name is not None and usecmap_resolver is not None:
            parent_data = usecmap_resolver(parent_name)
            if parent_data is not None:
                try:
                    parent = ToUnicodeCMap(
                        parent_data,
                        usecmap_resolver=usecmap_resolver,
                        _depth=_depth + 1,
                    )
                except ValueError:
                    parent = None
        self.code_space_ranges = tuple(parent.code_space_ranges if parent else ()) + parsed[0]
        self.mappings = dict(parent.mappings) if parent else {}
        self.mappings.update(parsed[1])
        self.decode_lengths = tuple(
            sorted(
                {len(end) for _, end in self.code_space_ranges} | {len(k) for k in self.mappings}
            )
            or {1}
        )
        self.fast_decode_table = None
        self.precalculate_fast_tables()
        self.fast_decode_table_2byte: list[str] | None = None

    def parse_codespace_ranges(self, data: bytes) -> None:
        code_space_ranges = typing.cast("list[tuple[bytes, bytes]]", self.code_space_ranges)
        saw_codespace_block = False
        valid_range_count = 0
        for block in iter_blocks(data, b"begincodespacerange", b"endcodespacerange"):
            saw_codespace_block = True
            tokens = cmap_tokens(block)
            if len(tokens) % 2 != 0:
                tokens = tokens[:-1]
            for i in range(0, len(tokens), 2):
                try:
                    start = decode_cmap_hex_token(tokens[i])
                    end = decode_cmap_hex_token(tokens[i + 1])
                    validate_codespace_range(start, end)
                except (ValueError, UnicodeDecodeError):
                    continue
                if any(ranges_overlap((start, end), existing) for existing in code_space_ranges):
                    raise ValueError("invalid ToUnicode CMap codespacerange")
                code_space_ranges.append((start, end))
                valid_range_count += 1
        if saw_codespace_block and valid_range_count == 0:
            raise ValueError("invalid ToUnicode CMap codespacerange")

    def parse_bfchar(self, data: bytes) -> None:
        for block in iter_blocks(data, b"beginbfchar", b"endbfchar"):
            items = cmap_tokens(block)
            if len(items) < 2:
                continue
            if len(items) % 2 != 0:
                items = items[:-1]
            for i in range(0, len(items), 2):
                src_tok = items[i]
                dst_tok = items[i + 1]
                try:
                    src = decode_cmap_token(src_tok)
                    dst = decode_utf16be(decode_cmap_token(dst_tok))
                except (ValueError, UnicodeDecodeError):
                    continue

                self.mappings[src] = dst

    def parse_bfrange(self, data: bytes) -> None:
        invalid_range_count = 0
        valid_range_count = 0
        for block in iter_blocks(data, b"beginbfrange", b"endbfrange"):
            items = cmap_tokens(block, include_arrays=True)
            if not items:
                continue
            if len(items) % 3 != 0:
                invalid_range_count += 1
                items = items[: len(items) - (len(items) % 3)]
            idx = 0
            while idx <= len(items) - 3:
                t1, t2, t3 = items[idx], items[idx + 1], items[idx + 2]
                if not (t1.startswith(b"<") and t2.startswith(b"<")):
                    invalid_range_count += 1
                    idx += 3
                    continue
                try:
                    start_bytes = decode_cmap_hex_token(t1)
                    end_bytes = decode_cmap_hex_token(t2)
                    start_code = int.from_bytes(start_bytes, "big")
                    end_code = int.from_bytes(end_bytes, "big")
                    src_len = len(start_bytes)
                except (ValueError, UnicodeDecodeError, IndexError):
                    invalid_range_count += 1
                    idx += 3
                    continue
                if len(start_bytes) != len(end_bytes):
                    invalid_range_count += 1
                    idx += 3
                    continue
                if start_code > end_code:
                    invalid_range_count += 1
                    idx += 3
                    continue

                if t3.startswith(b"["):
                    dsts = cmap_tokens(t3)
                    if not dsts:
                        invalid_range_count += 1
                        idx += 3
                        continue
                    added = False
                    for i, dst_tok in enumerate(dsts):
                        if start_code + i > end_code:
                            break
                        try:
                            dst = decode_utf16be(decode_cmap_token(dst_tok))
                        except (ValueError, UnicodeDecodeError):
                            continue
                        src = (start_code + i).to_bytes(src_len, "big")
                        self.mappings[src] = dst
                        added = True
                    if added:
                        valid_range_count += 1
                    else:
                        invalid_range_count += 1
                    idx += 3
                elif t3.startswith(b"<") or t3.startswith(b"("):
                    try:
                        base_dst = decode_utf16be(decode_cmap_token(t3))
                        m = expand_range(start_code, end_code, src_len, base_dst)
                    except (ValueError, UnicodeDecodeError):
                        invalid_range_count += 1
                        idx += 3
                        continue
                    for k, v in m.items():
                        self.mappings[k] = v
                    valid_range_count += 1
                    idx += 3
                else:
                    invalid_range_count += 1
                    idx += 3
        if invalid_range_count and not valid_range_count:
            raise ValueError("invalid ToUnicode CMap bfrange")

    def precalculate_fast_tables(self) -> None:
        if 1 in self.decode_lengths:
            table = [""] * 256
            for i in range(256):
                b = BYTE_CACHE[i]
                res = self.mappings.get(b)
                table[i] = res if res is not None else chr(i)
            self.fast_decode_table = table

    def get_fast_decode_table_2byte(self) -> list[str] | None:
        if self.fast_decode_table_2byte is not None:
            return self.fast_decode_table_2byte
        if 2 not in self.decode_lengths:
            return None
        table2 = [
            unicode_scalar_or_replacement(code) if code != 0 else "\ufffd" for code in range(65536)
        ]
        for k, v in self.mappings.items():
            if len(k) == 2:
                code = (k[0] << 8) | k[1]
                table2[code] = v
        self.fast_decode_table_2byte = table2
        return table2

    def decode(self, data: bytes) -> str:
        if not data:
            return ""

        if self.fast_decode_table is not None and (
            not self.decode_lengths or self.decode_lengths == (1,)
        ):
            table = self.fast_decode_table
            result = "".join(table[byte] for byte in data)
            if "\x00" in result:
                return result.replace("\x00", "")
            return result

        n = len(data)
        if n == 1 and self.fast_decode_table is not None:
            result = self.fast_decode_table[data[0]]
            if result:
                return result

        if self.decode_lengths == (2,) and n % 2 == 0:
            table2 = self.get_fast_decode_table_2byte()
            if table2 is None:
                return ""
            if n > 64:
                cids = array.array("H", data)
                if sys.byteorder == "little":
                    cids.byteswap()
                result = "".join(map(table2.__getitem__, cids))
            else:
                out_small = []
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    out_small.append(table2[code])
                result = "".join(out_small)

            if "\x00" in result:
                return result.replace("\x00", "")
            return result

        mappings = self.mappings
        lengths = self.decode_lengths or (1,)
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

                if length == 1:
                    chunk = bc[data[pos]]
                else:
                    chunk = data[pos : pos + length]

                mapped = mappings_get(chunk)
                if mapped is not None:
                    out_append(mapped)
                    pos += length
                    match_found = True
                    break

            if match_found:
                continue

            chunk1 = bc[data[pos]]
            mapped1 = mappings_get(chunk1)
            if mapped1 is not None:
                out_append(mapped1)
                pos += 1
                continue

            if 1 not in lengths and n - pos >= 2:
                cid = (data[pos] << 8) | data[pos + 1]
                pos += 2
                out_append(unicode_scalar_or_replacement(cid) if cid != 0 else "\ufffd")
            else:
                out_append(chr(data[pos]))
                pos += 1

        result = "".join(out)
        if "\x00" in result:
            return result.replace("\x00", "")
        return result


ParsedToUnicodeCMap = tuple[
    tuple[tuple[bytes, bytes], ...],
    dict[bytes, str],
    tuple[int, ...],
    tuple[str, ...] | None,
]


@lru_cache(maxsize=256)
def _parse_to_unicode_cmap(data: bytes) -> ParsedToUnicodeCMap:
    cmap = object.__new__(ToUnicodeCMap)
    cmap.code_space_ranges = []
    cmap.mappings = {}
    cmap.fast_decode_table = None
    cmap.fast_decode_table_2byte = None

    cmap.parse_codespace_ranges(data)
    cmap.parse_bfchar(data)
    cmap.parse_bfrange(data)

    cmap.decode_lengths = tuple(
        sorted(
            (
                {len(end) for ignored, end in cmap.code_space_ranges}
                | {len(k) for k in cmap.mappings}
            )
            or {1},
            reverse=False,
        )
    )
    cmap.precalculate_fast_tables()
    fast_decode_table = (
        tuple(cmap.fast_decode_table) if cmap.fast_decode_table is not None else None
    )
    return (
        tuple(cmap.code_space_ranges),
        cmap.mappings,
        tuple(cmap.decode_lengths),
        fast_decode_table,
    )


class CMapDecoder:
    code_space_ranges: list[tuple[bytes, bytes]]
    cid_mappings: dict[bytes, int]
    cid_ranges: list[CIDRange]
    decode_lengths: tuple[int, ...]
    notdef_mappings: dict[bytes, int]
    notdef_ranges: list[NotdefRange]
    default_to_identity: bool
    wmode: int

    __slots__ = (
        "code_space_ranges",
        "cid_mappings",
        "cid_ranges",
        "decode_lengths",
        "notdef_mappings",
        "notdef_ranges",
        "default_to_identity",
        "wmode",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: CMapResourceResolver | None = None,
        _depth: int = 0,
    ) -> None:
        if _depth > 5:
            raise ValueError("CMap usecmap nesting too deep")
        data = bytes(data)
        self.code_space_ranges: list[tuple[bytes, bytes]] = []
        self.cid_mappings: dict[bytes, int] = {}
        self.cid_ranges: list[CIDRange] = []
        self.notdef_mappings: dict[bytes, int] = {}
        self.notdef_ranges: list[NotdefRange] = []
        self.default_to_identity = False
        self.wmode = 0
        usecmap_name = cmap_usecmap_name(data)
        if usecmap_name is not None:
            parent = self.resolve_usecmap(
                usecmap_name, usecmap_resolver=usecmap_resolver, depth=_depth + 1
            )
            if parent is not None:
                self.inherit(parent)
        local_wmode = cmap_wmode(data)
        if local_wmode is not None:
            self.wmode = local_wmode

        for block in iter_blocks(data, b"begincodespacerange", b"endcodespacerange"):
            tokens = cmap_tokens(block)
            if len(tokens) % 2 != 0:
                raise ValueError("invalid CMap codespacerange")
            for i in range(0, len(tokens), 2):
                try:
                    start = decode_cmap_hex_token(tokens[i])
                    end = decode_cmap_hex_token(tokens[i + 1])
                    validate_codespace_range(start, end)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("invalid CMap codespacerange") from exc
                if any(
                    ranges_overlap((start, end), existing) for existing in self.code_space_ranges
                ):
                    raise ValueError("overlapping CMap codespacerange")
                self.code_space_ranges.append((start, end))
        if not self.code_space_ranges:
            if usecmap_name in {"Identity-H", "Identity-V"}:
                self.code_space_ranges.append((b"\x00\x00", b"\xff\xff"))
                self.default_to_identity = True
            elif usecmap_name in {"OneByteIdentityH", "OneByteIdentityV"}:
                self.code_space_ranges.append((b"\x00", b"\xff"))
                self.default_to_identity = True
        self.parse_cidchar(data)
        self.parse_cidrange(data)
        self.parse_notdefchar(data)
        self.parse_notdefrange(data)
        self.decode_lengths = tuple(
            sorted(
                (
                    {len(end) for ignored, end in self.code_space_ranges}
                    | {len(k) for k in self.cid_mappings}
                    | {len(item.end) for item in self.cid_ranges}
                    | {len(k) for k in self.notdef_mappings}
                    | {len(item.end) for item in self.notdef_ranges}
                )
                or {1},
                reverse=False,
            )
        )
        self.freeze()

    @classmethod
    def identity(cls, *, byte_width: int = 2, wmode: int = 0) -> "CMapDecoder":
        cmap = typing.cast(typing.Any, object.__new__(cls))
        if byte_width == 1:
            cmap.code_space_ranges = ((b"\x00", b"\xff"),)
            cmap.decode_lengths = (1,)
        else:
            cmap.code_space_ranges = ((b"\x00\x00", b"\xff\xff"),)
            cmap.decode_lengths = (2,)
        cmap.cid_mappings = MappingProxyType({})
        cmap.cid_ranges = ()
        cmap.notdef_mappings = MappingProxyType({})
        cmap.notdef_ranges = ()
        cmap.default_to_identity = True
        cmap.wmode = wmode
        return typing.cast("CMapDecoder", cmap)

    def freeze(self) -> None:
        frozen = typing.cast(typing.Any, self)
        frozen.code_space_ranges = tuple(self.code_space_ranges)
        frozen.cid_mappings = MappingProxyType(dict(self.cid_mappings))
        frozen.cid_ranges = tuple(self.cid_ranges)
        frozen.notdef_mappings = MappingProxyType(dict(self.notdef_mappings))
        frozen.notdef_ranges = tuple(self.notdef_ranges)

    @staticmethod
    def resolve_usecmap(
        name: str,
        *,
        usecmap_resolver: CMapResourceResolver | None,
        depth: int,
    ) -> "CMapDecoder | None":
        if name in {"Identity-H", "Identity-V"}:
            return CMapDecoder.identity(byte_width=2, wmode=int(name.endswith("-V")))
        if name in {"OneByteIdentityH", "OneByteIdentityV"}:
            return CMapDecoder.identity(byte_width=1, wmode=int(name.endswith("V")))
        if usecmap_resolver is None:
            return None
        resolved = usecmap_resolver(name)
        if resolved is None:
            return None
        if isinstance(resolved, CMapDecoder):
            return resolved
        return CMapDecoder(
            resolved,
            usecmap_resolver=usecmap_resolver,
            _depth=depth,
        )

    def inherit(self, parent: "CMapDecoder") -> None:
        self.code_space_ranges.extend(parent.code_space_ranges)
        self.cid_mappings.update(parent.cid_mappings)
        self.cid_ranges.extend(parent.cid_ranges)
        self.notdef_mappings.update(parent.notdef_mappings)
        self.notdef_ranges.extend(parent.notdef_ranges)
        self.default_to_identity = parent.default_to_identity
        self.wmode = parent.wmode

    def parse_cidchar(self, data: bytes) -> None:
        for block in iter_blocks(data, b"begincidchar", b"endcidchar"):
            items = cmap_noncomment_words(block)
            if len(items) % 2 != 0:
                items = items[:-1]
            for i in range(0, len(items), 2):
                code_token, cid_token = items[i], items[i + 1]
                if not (code_token.startswith(b"<") and code_token.endswith(b">")):
                    continue
                try:
                    code = decode_cmap_hex_token(code_token)
                    cid = int(cid_token)
                except (ValueError, UnicodeDecodeError):
                    continue
                self.cid_mappings[code] = cid

    def parse_cidrange(self, data: bytes) -> None:
        for block in iter_blocks(data, b"begincidrange", b"endcidrange"):
            items = cmap_noncomment_words(block)
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
                    start_bytes = decode_cmap_hex_token(start_token)
                    end_bytes = decode_cmap_hex_token(end_token)
                    cid = int(cid_token)
                except (ValueError, UnicodeDecodeError):
                    continue
                if len(start_bytes) != len(end_bytes):
                    continue
                try:
                    validate_codespace_range(start_bytes, end_bytes)
                except ValueError:
                    continue
                remove_codes_in_range(self.cid_mappings, start_bytes, end_bytes)
                self.cid_ranges.append(CIDRange(start_bytes, end_bytes, cid))

    def parse_notdefchar(self, data: bytes) -> None:
        for block in iter_blocks(data, b"beginnotdefchar", b"endnotdefchar"):
            items = cmap_noncomment_words(block)
            if len(items) % 2 != 0:
                items = items[:-1]
            for i in range(0, len(items), 2):
                code_token, cid_token = items[i], items[i + 1]
                if not (code_token.startswith(b"<") and code_token.endswith(b">")):
                    continue
                try:
                    code = decode_cmap_hex_token(code_token)
                    cid = int(cid_token)
                except (ValueError, UnicodeDecodeError):
                    continue
                self.notdef_mappings[code] = cid

    def parse_notdefrange(self, data: bytes) -> None:
        for block in iter_blocks(data, b"beginnotdefrange", b"endnotdefrange"):
            items = cmap_noncomment_words(block)
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
                    start_bytes = decode_cmap_hex_token(start_token)
                    end_bytes = decode_cmap_hex_token(end_token)
                    cid = int(cid_token)
                    validate_codespace_range(start_bytes, end_bytes)
                except (ValueError, UnicodeDecodeError):
                    continue
                remove_codes_in_range(self.notdef_mappings, start_bytes, end_bytes)
                self.notdef_ranges.append(NotdefRange(start_bytes, end_bytes, cid))

    def mapped_cid(self, code: bytes) -> int | None:
        cid = self.cid_mappings.get(code)
        if cid is not None:
            return cid
        for cid_range in reversed(self.cid_ranges):
            if cid_range.contains(code):
                return cid_range.cid_for(code)
        return None

    def mapped_notdef(self, code: bytes) -> int | None:
        cid = self.notdef_mappings.get(code)
        if cid is not None:
            return cid
        for notdef_range in reversed(self.notdef_ranges):
            if notdef_range.contains(code):
                return notdef_range.cid
        return None

    def decode(self, data: bytes) -> list[tuple[bytes, int]]:
        if not data:
            return []
        out: list[tuple[bytes, int]] = []
        pos = 0
        n = len(data)
        ranges = self.code_space_ranges
        while pos < n:
            matched = False
            for length in self.decode_lengths:
                if pos + length > n:
                    continue
                chunk = data[pos : pos + length]
                if ranges and not code_in_ranges(chunk, ranges):
                    continue
                cid = self.mapped_cid(chunk)
                if cid is None:
                    cid = self.mapped_notdef(chunk)
                if cid is None:
                    cid = int.from_bytes(chunk, "big") if self.default_to_identity and chunk else 0
                out.append((chunk, cid))
                pos += length
                matched = True
                break
            if not matched:
                out.append((data[pos : pos + 1], 0))
                pos += 1
        return out


CMapResourceResolver = Callable[[str], bytes | bytearray | memoryview | CMapDecoder | None]
