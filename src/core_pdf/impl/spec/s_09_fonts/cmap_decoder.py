"""CMap to CID parsing and decoding."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from core_pdf.impl.spec.s_09_fonts.cmap_encoding import BYTE_CACHE
from core_pdf.impl.spec.s_09_fonts.cmap_ranges import (
    CIDRange,
    NotdefRange,
    code_in_range,
    range_offset,
    ranges_overlap,
    remove_codes_in_range,
    validate_codespace_range,
)
from core_pdf.impl.spec.s_09_fonts.cmap_tokenizer import (
    CMapBlock,
    CMapProgram,
    cmap_metadata,
    decode_cmap_hex_token,
)

if TYPE_CHECKING:
    # numpy is imported lazily at each call site to keep module import cheap.
    import numpy

CodeRangeT = TypeVar("CodeRangeT", CIDRange, NotdefRange)
internal_MIN_CID = 0
internal_MAX_CID = 0xFFFF


def internal_valid_cid(cid: int) -> bool:
    return internal_MIN_CID <= cid <= internal_MAX_CID


class CMapDecoder:
    code_space_ranges: list[tuple[bytes, bytes]]
    cid_mappings: dict[bytes, int]
    cid_ranges: list[CIDRange]
    cid_ranges_by_length: dict[int, tuple[CIDRange, ...]]
    decode_lengths: tuple[int, ...]
    notdef_mappings: dict[bytes, int]
    notdef_ranges: list[NotdefRange]
    notdef_ranges_by_length: dict[int, tuple[NotdefRange, ...]]
    code_space_ranges_by_length: dict[int, tuple[tuple[bytes, bytes], ...]]
    default_to_identity: bool
    wmode: int
    internal_entry_cache: dict[int | bytes, tuple[tuple[bytes, int], int]]

    __slots__ = (
        "code_space_ranges",
        "cid_mappings",
        "cid_ranges",
        "cid_ranges_by_length",
        "decode_lengths",
        "notdef_mappings",
        "notdef_ranges",
        "notdef_ranges_by_length",
        "code_space_ranges_by_length",
        "default_to_identity",
        "wmode",
        "internal_entry_cache",
    )

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: CMapResourceResolver | None = None,
        internal_depth: int = 0,
        internal_empty: bool = False,
    ) -> None:
        self.code_space_ranges = []
        self.cid_mappings = {}
        self.cid_ranges = []
        self.cid_ranges_by_length = {}
        self.notdef_mappings = {}
        self.notdef_ranges = []
        self.notdef_ranges_by_length = {}
        self.code_space_ranges_by_length = {}
        self.default_to_identity = False
        self.wmode = 0
        self.internal_entry_cache = {}
        if internal_empty:
            self.decode_lengths = ()
            return
        if internal_depth > 5:
            raise ValueError("CMap usecmap nesting too deep")
        data = bytes(data)
        program = CMapProgram.parse(data)
        usecmap_name, local_wmode = cmap_metadata(program)
        if usecmap_name is not None:
            parent = self.resolve_usecmap(
                usecmap_name, usecmap_resolver=usecmap_resolver, depth=internal_depth + 1
            )
            if parent is not None:
                self.inherit(parent)
        if local_wmode is not None:
            self.wmode = local_wmode

        for block in program.blocks(b"begincodespacerange", b"endcodespacerange"):
            tokens = block.token_values()
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
        self.parse_mapping_blocks(program)
        self.decode_lengths = tuple(
            sorted(
                length
                for length in (
                    {len(end) for ignored, end in self.code_space_ranges}
                    | {len(k) for k in self.cid_mappings}
                    | {len(item.end) for item in self.cid_ranges}
                    | {len(k) for k in self.notdef_mappings}
                    | {len(item.end) for item in self.notdef_ranges}
                )
                if length > 0
            )
            or (1,)
        )
        self.freeze()

    @classmethod
    def identity(cls, *, byte_width: int = 2, wmode: int = 0) -> "CMapDecoder":
        cmap = cls(b"", internal_empty=True)
        if byte_width == 1:
            cmap.code_space_ranges = [(b"\x00", b"\xff")]
            cmap.decode_lengths = (1,)
        else:
            cmap.code_space_ranges = [(b"\x00\x00", b"\xff\xff")]
            cmap.decode_lengths = (2,)
        cmap.default_to_identity = True
        cmap.wmode = wmode
        cmap.freeze()
        return cmap

    def freeze(self) -> None:
        self.cid_ranges_by_length = index_ranges_by_length(self.cid_ranges)
        self.notdef_ranges_by_length = index_ranges_by_length(self.notdef_ranges)
        self.code_space_ranges_by_length = index_code_space_ranges(self.code_space_ranges)

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
            internal_depth=depth,
        )

    def inherit(self, parent: "CMapDecoder") -> None:
        self.code_space_ranges.extend(parent.code_space_ranges)
        self.cid_mappings.update(parent.cid_mappings)
        self.cid_ranges.extend(parent.cid_ranges)
        self.notdef_mappings.update(parent.notdef_mappings)
        self.notdef_ranges.extend(parent.notdef_ranges)
        self.default_to_identity = parent.default_to_identity
        self.wmode = parent.wmode

    def parse_mapping_blocks(self, program: CMapProgram) -> None:
        """Compile mapping sections in order so succeeding definitions win."""
        delimiters = {
            b"begincidchar": b"endcidchar",
            b"begincidrange": b"endcidrange",
            b"beginnotdefchar": b"endnotdefchar",
            b"beginnotdefrange": b"endnotdefrange",
        }
        for begin_keyword, block in program.blocks_in_order(delimiters):
            match begin_keyword:
                case b"begincidchar":
                    self.parse_char_block(block, self.cid_mappings)
                case b"begincidrange":
                    self.parse_range_block(block, self.cid_mappings, self.cid_ranges, CIDRange)
                case b"beginnotdefchar":
                    self.parse_char_block(block, self.notdef_mappings)
                case b"beginnotdefrange":
                    self.parse_range_block(
                        block,
                        self.notdef_mappings,
                        self.notdef_ranges,
                        NotdefRange,
                    )

    def parse_char_block(
        self,
        block: CMapBlock,
        mappings: dict[bytes, int],
    ) -> None:
        """Collect the `<code> cid` pairs from one character-mapping block."""
        items = block.token_values(include_words=True)
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
            if not code or not internal_valid_cid(cid):
                continue
            mappings[code] = cid

    def parse_range_block(
        self,
        block: CMapBlock,
        mappings: dict[bytes, int],
        ranges: list[CodeRangeT],
        make_range: Callable[[bytes, bytes, int], CodeRangeT],
    ) -> None:
        """Collect range triples, dropping explicit codes the range supersedes."""
        items = block.token_values(include_words=True)
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
                # Also rejects empty or mismatched start/end lengths.
                validate_codespace_range(start_bytes, end_bytes)
            except (ValueError, UnicodeDecodeError):
                continue
            if not internal_valid_cid(cid):
                continue
            if make_range is CIDRange:
                last_cid = cid + range_offset(
                    end_bytes,
                    start_bytes,
                    end_bytes,
                    validate_range=False,
                    validate_code=False,
                )
                if not internal_valid_cid(last_cid):
                    continue
            remove_codes_in_range(mappings, start_bytes, end_bytes)
            ranges.append(make_range(start_bytes, end_bytes, cid))

    def mapped_cid(self, code: bytes) -> int | None:
        cid = self.cid_mappings.get(code)
        if cid is not None:
            return cid
        for cid_range in self.cid_ranges_by_length.get(len(code), ()):
            if cid_range.contains(code):
                return cid_range.first_cid + range_offset(
                    code,
                    cid_range.start,
                    cid_range.end,
                    validate_range=False,
                    validate_code=False,
                )
        return None

    def mapped_notdef(self, code: bytes) -> int | None:
        cid = self.notdef_mappings.get(code)
        if cid is not None:
            return cid
        for notdef_range in self.notdef_ranges_by_length.get(len(code), ()):
            if notdef_range.contains(code):
                return notdef_range.cid
        return None

    def decode_entries(self, data: bytes) -> list[tuple[bytes, int]]:
        if not data:
            return []
        if (
            self.default_to_identity
            and not self.cid_mappings
            and not self.cid_ranges
            and not self.notdef_mappings
            and not self.notdef_ranges
            and len(self.code_space_ranges) == 1
        ):
            start, end = self.code_space_ranges[0]
            if start == b"\x00" and end == b"\xff" and self.decode_lengths == (1,):
                return [(BYTE_CACHE[value], value) for value in data]
            if start == b"\x00\x00" and end == b"\xff\xff" and self.decode_lengths == (2,):
                limit = len(data) - (len(data) & 1)
                identity_result = [
                    (
                        data[pos : pos + 2],
                        (data[pos] << 8) | data[pos + 1],
                    )
                    for pos in range(0, limit, 2)
                ]
                if limit != len(data):
                    identity_result.append((BYTE_CACHE[data[-1]], 0))
                return identity_result
        if (
            self.decode_lengths == (1,)
            and self.code_space_ranges == [(b"\x00", b"\xff")]
            and not self.cid_ranges
            and not self.notdef_ranges
        ):
            cid_mappings = self.cid_mappings
            notdef_mappings = self.notdef_mappings
            default_to_identity = self.default_to_identity
            result: list[tuple[bytes, int]] = []
            for value in data:
                code = BYTE_CACHE[value]
                cid = cid_mappings.get(code)
                if cid is None:
                    cid = notdef_mappings.get(code)
                if cid is None:
                    cid = value if default_to_identity else 0
                result.append((code, cid))
            return result
        out: list[tuple[bytes, int]] = []
        pos = 0
        n = len(data)
        ranges = self.code_space_ranges_by_length
        decode_lengths = self.decode_lengths
        # Resolution is a pure function of the bytes at pos (up to the longest
        # decode length), so memoize it per decoder. Near the end of the data,
        # longer lengths get skipped, so those positions bypass the cache.
        max_length = max(decode_lengths, default=1)
        cache = self.internal_entry_cache
        cache_get = cache.get
        single_byte = max_length == 1
        cache_limit = n - max_length
        key: int | bytes | None
        while pos < n:
            if pos <= cache_limit:
                key = data[pos] if single_byte else data[pos : pos + max_length]
                cached = cache_get(key)
                if cached is not None:
                    out.append(cached[0])
                    pos += cached[1]
                    continue
            else:
                key = None
            matched = False
            for length in decode_lengths:
                if length <= 0 or pos + length > n:
                    continue
                chunk = BYTE_CACHE[data[pos]] if length == 1 else data[pos : pos + length]
                length_ranges = ranges.get(length)
                if ranges and not length_ranges:
                    continue
                if length_ranges and not any(
                    code_in_range(chunk, start, end) for start, end in length_ranges
                ):
                    continue
                cid = self.mapped_cid(chunk)
                if cid is None:
                    cid = self.mapped_notdef(chunk)
                if cid is None:
                    cid = int.from_bytes(chunk, "big") if self.default_to_identity and chunk else 0
                entry = (chunk, cid)
                if key is not None and len(cache) < 65536:
                    cache[key] = (entry, length)
                out.append(entry)
                pos += length
                matched = True
                break
            if not matched:
                # ISO 32000-1 9.7.6.3: an invalid code does not consume one
                # byte -- "The length of the codes in the chosen codespace
                # range determines the total number of bytes to consume from
                # the string." Consuming one byte instead desynchronized every
                # following code in the string.
                length = self.internal_invalid_code_length(data, pos, n)
                entry = (bytes(data[pos : pos + length]), 0)
                if key is not None and len(cache) < 65536:
                    cache[key] = (entry, length)
                out.append(entry)
                pos += length
        return out

    def internal_invalid_code_length(
        self, data: bytes | bytearray | memoryview, pos: int, limit: int
    ) -> int:
        """Bytes to consume for a code matching no codespace range (9.7.6.3).

        (a) If the first byte matches no range's first byte, the range having
        the shortest codes is chosen. (b) Otherwise the longest partial match
        wins, ties going to the shortest codes.
        """
        ranges = self.code_space_ranges_by_length
        if not ranges:
            return 1
        lengths = sorted(ranges)
        shortest = lengths[0]
        best_partial = 0
        chosen = shortest
        # Ascending lengths, and a strict improvement test, so a tie keeps the
        # shortest codes without a second comparison.
        for length in lengths:
            for start, end in ranges[length]:
                matched_bytes = 0
                while (
                    matched_bytes < length
                    and pos + matched_bytes < limit
                    and start[matched_bytes] <= data[pos + matched_bytes] <= end[matched_bytes]
                ):
                    matched_bytes += 1
                if matched_bytes > best_partial:
                    best_partial = matched_bytes
                    chosen = length
        return chosen if best_partial else shortest

    def decode_cids_array(
        self, data: bytes | bytearray | memoryview
    ) -> numpy.ndarray[Any, Any] | None:
        """Decode uniform identity CMaps without allocating code-byte tuples.

        ``None`` signals that the CMap requires the general object-oriented
        decoder.  The numeric path is intentionally conservative: callers can
        use it for width/advance calculations while retaining exact fallback
        semantics for arbitrary CMaps.
        """
        if not data:
            import numpy

            return numpy.empty(0, dtype=numpy.int64)
        if (
            not self.default_to_identity
            or self.cid_mappings
            or self.cid_ranges
            or self.notdef_mappings
            or self.notdef_ranges
            or len(self.code_space_ranges) != 1
        ):
            return None
        start, end = self.code_space_ranges[0]
        import numpy

        if start == b"\x00" and end == b"\xff" and self.decode_lengths == (1,):
            return numpy.frombuffer(bytes(data), dtype=numpy.uint8).astype(numpy.int64, copy=False)
        if start == b"\x00\x00" and end == b"\xff\xff" and self.decode_lengths == (2,):
            raw = bytes(data)
            limit = len(raw) - (len(raw) & 1)
            values = numpy.frombuffer(raw, dtype=">u2", count=limit // 2).astype(
                numpy.int64, copy=False
            )
            if limit == len(raw):
                return values
            return numpy.concatenate((values, numpy.array([0], dtype=numpy.int64)))
        return None


CMapResourceResolver = Callable[[str], bytes | bytearray | memoryview | CMapDecoder | None]


def index_ranges_by_length(ranges: list[CodeRangeT]) -> dict[int, tuple[CodeRangeT, ...]]:
    """Bucket code ranges by code length, later definitions first."""
    indexed: dict[int, list[CodeRangeT]] = {}
    for item in reversed(ranges):
        indexed.setdefault(len(item.start), []).append(item)
    return {length: tuple(items) for length, items in indexed.items()}


def index_code_space_ranges(
    ranges: list[tuple[bytes, bytes]],
) -> dict[int, tuple[tuple[bytes, bytes], ...]]:
    indexed: dict[int, list[tuple[bytes, bytes]]] = {}
    for item in ranges:
        indexed.setdefault(len(item[0]), []).append(item)
    return {length: tuple(items) for length, items in indexed.items()}
