"""ToUnicode CMap parsing and decoding."""

from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy

from core_pdf.impl.engine.spec.s_09_fonts.cmap_encoding import BYTE_CACHE, decode_utf16be
from core_pdf.impl.engine.spec.s_09_fonts.cmap_ranges import (
    MAX_CMAP_RANGE_SPAN,
    CodeSpaceRanges,
    expand_range,
    ranges_overlap,
    unicode_scalar_or_replacement,
    validate_codespace_range,
)
from core_pdf.impl.engine.spec.s_09_fonts.cmap_tokenizer import (
    CMapBlock,
    CMapProgram,
    cmap_metadata,
    cmap_tokens,
    decode_cmap_hex_token,
    decode_cmap_token,
)


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
        internal_depth: int = 0,
        internal_empty: bool = False,
    ) -> None:
        self.code_space_ranges = []
        self.mappings = {}
        self.fast_decode_table = None
        self.fast_decode_table_2byte = None
        if internal_empty:
            self.decode_lengths = ()
            return
        if internal_depth > 16:
            raise ValueError("ToUnicode CMap UseCMap recursion limit exceeded")
        source = data if type(data) is bytes else bytes(data)
        parsed = parse_to_unicode_cmap(source)
        parent: ToUnicodeCMap | None = None
        parent_name = parsed.usecmap_name
        if parent_name is not None and usecmap_resolver is not None:
            parent_data = usecmap_resolver(parent_name)
            if parent_data is not None:
                try:
                    parent = ToUnicodeCMap(
                        parent_data,
                        usecmap_resolver=usecmap_resolver,
                        internal_depth=internal_depth + 1,
                    )
                except ValueError:
                    parent = None
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
        parsed_fast_table = parsed.fast_decode_table
        if parent is None and parsed_fast_table is not None:
            self.fast_decode_table = parsed_fast_table
        else:
            self.fast_decode_table = None
            self.precalculate_fast_tables()
        self.fast_decode_table_2byte = None

    def parse_codespace_ranges(self, program: CMapProgram) -> None:
        code_space_ranges = typing.cast("list[tuple[bytes, bytes]]", self.code_space_ranges)
        saw_codespace_block = False
        valid_range_count = 0
        for block in program.blocks(b"begincodespacerange", b"endcodespacerange"):
            saw_codespace_block = True
            tokens = block.token_values()
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

    def parse_mapping_blocks(self, program: CMapProgram) -> None:
        """Compile character mappings in order so succeeding definitions win."""
        delimiters = {
            b"beginbfchar": b"endbfchar",
            b"beginbfrange": b"endbfrange",
            b"begincidrange": b"endcidrange",
        }
        invalid_range_count = 0
        valid_range_count = 0
        for begin_keyword, block in program.blocks_in_order(delimiters):
            match begin_keyword:
                case b"beginbfchar":
                    self.parse_bfchar_block(block)
                case b"beginbfrange":
                    block_invalid, block_valid = self.parse_bfrange_block(block)
                    invalid_range_count += block_invalid
                    valid_range_count += block_valid
                case b"begincidrange":
                    self.parse_cidrange_block(block)
        if invalid_range_count and not valid_range_count:
            raise ValueError("invalid ToUnicode CMap bfrange")

    def parse_bfchar_block(self, block: CMapBlock) -> None:
        items = block.token_values()
        if len(items) < 2:
            return
        if len(items) % 2 != 0:
            items = items[:-1]
        for i in range(0, len(items), 2):
            src_tok = items[i]
            dst_tok = items[i + 1]
            try:
                src = decode_cmap_token(src_tok)
                if not src:
                    continue
                dst = decode_utf16be(decode_cmap_token(dst_tok))
            except (ValueError, UnicodeDecodeError):
                # PostScript hex strings pad an odd final nibble with zero.
                # If corruption starts another ``<`` before the closing
                # delimiter, pdfminer's parser retains the valid prefix as
                # the destination and abandons the now-misaligned operands
                # that follow in this bfchar block.
                if dst_tok.startswith(b"<") and b"<" in dst_tok[1:]:
                    prefix = dst_tok[1 : dst_tok.find(b"<", 1)]
                    try:
                        src = decode_cmap_token(src_tok)
                        if not src:
                            continue
                        if len(prefix) % 2:
                            prefix += b"0"
                        dst = decode_utf16be(bytes.fromhex(prefix.decode("ascii")))
                    except (ValueError, UnicodeDecodeError):
                        break
                    self.mappings[src] = dst
                    break
                continue

            self.mappings[src] = dst

    def parse_bfrange_block(self, block: CMapBlock) -> tuple[int, int]:
        invalid_range_count = 0
        valid_range_count = 0
        items = block.token_values(include_arrays=True)
        if not items:
            return invalid_range_count, valid_range_count
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
            if not start_bytes or len(start_bytes) != len(end_bytes):
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
                    mappings = expand_range(start_code, end_code, src_len, base_dst)
                except (ValueError, UnicodeDecodeError):
                    invalid_range_count += 1
                    idx += 3
                    continue
                self.mappings.update(mappings)
                valid_range_count += 1
                idx += 3
            else:
                invalid_range_count += 1
                idx += 3
        return invalid_range_count, valid_range_count

    def parse_cidrange_block(self, block: CMapBlock) -> None:
        """Parse numeric CID ranges accepted by PDFMiner in ToUnicode maps.

        Although a conforming ToUnicode CMap normally uses ``bfrange``, some
        producers emit ``cidrange`` records whose numeric destination is a
        Unicode scalar. PostScript CMap parsing accepts those records, and
        PDFMiner consequently exposes their text. Retain that recovery without
        changing ordinary encoding-CMap semantics.
        """
        items = block.token_values(include_words=True)
        if len(items) % 3 != 0:
            items = items[: len(items) - (len(items) % 3)]
        for index in range(0, len(items), 3):
            try:
                start_bytes = decode_cmap_hex_token(items[index])
                end_bytes = decode_cmap_hex_token(items[index + 1])
                destination = int(items[index + 2])
            except (ValueError, UnicodeDecodeError):
                continue
            if not start_bytes or len(start_bytes) != len(end_bytes):
                continue
            start = int.from_bytes(start_bytes, "big")
            end = int.from_bytes(end_bytes, "big")
            if start > end or end - start + 1 > MAX_CMAP_RANGE_SPAN:
                continue
            source_length = len(start_bytes)
            for offset, source in enumerate(range(start, end + 1)):
                self.mappings[source.to_bytes(source_length, "big")] = (
                    unicode_scalar_or_replacement(destination + offset)
                )

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

    def decode(self, data: bytes, *, preserve_nulls: bool = False) -> str:
        if not data:
            return ""

        if self.fast_decode_table is not None and (
            not self.decode_lengths or self.decode_lengths == (1,)
        ):
            table = self.fast_decode_table
            result = "".join(map(table.__getitem__, data))
            if not preserve_nulls and "\x00" in result:
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
                cids = numpy.frombuffer(data, dtype=">u2").tolist()
                result = "".join([table2[cid] for cid in cids])
            else:
                out_small = []
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    out_small.append(table2[code])
                result = "".join(out_small)

            if not preserve_nulls and "\x00" in result:
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
                if length <= 0 or pos + length > n:
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
        if not preserve_nulls and "\x00" in result:
            return result.replace("\x00", "")
        return result


@dataclass(frozen=True, slots=True)
class ParsedToUnicodeCMap:
    code_space_ranges: tuple[tuple[bytes, bytes], ...]
    mappings: dict[bytes, str]
    fast_decode_table: tuple[str, ...] | None
    usecmap_name: str | None


@lru_cache(maxsize=4096)
def parse_to_unicode_cmap(data: bytes) -> ParsedToUnicodeCMap:
    cmap = ToUnicodeCMap(b"", internal_empty=True)
    program = CMapProgram.parse(data)

    cmap.parse_mapping_blocks(program)
    try:
        cmap.parse_codespace_ranges(program)
    except ValueError:
        # A number of producers write a numerically ordered codespace whose
        # individual bytes are not ordered (for example ``<0083> <020c>``).
        # That is not a valid rectangular CMap codespace, but the explicit
        # bfchar/bfrange entries remain unambiguous.  PostScript CMap parsers
        # such as PDFMiner retain those entries, so recover them instead of
        # rejecting the complete ToUnicode map.  A map with no usable entries
        # still raises, preserving validation for genuinely empty corruption.
        if not cmap.mappings:
            raise
        cmap.code_space_ranges = []

    cmap.decode_lengths = tuple(
        sorted(
            length
            for length in (
                {len(end) for ignored, end in cmap.code_space_ranges}
                | {len(k) for k in cmap.mappings}
            )
            if length > 0
        )
        or (1,)
    )
    cmap.precalculate_fast_tables()
    fast_decode_table: tuple[str, ...] | None = (
        tuple(cmap.fast_decode_table) if cmap.fast_decode_table is not None else None
    )
    return ParsedToUnicodeCMap(
        code_space_ranges=tuple(cmap.code_space_ranges),
        mappings=cmap.mappings,
        fast_decode_table=fast_decode_table,
        usecmap_name=cmap_metadata(program)[0],
    )
