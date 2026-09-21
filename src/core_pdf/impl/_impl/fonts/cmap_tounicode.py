"""ToUnicode CMap parsing and decoding."""

from __future__ import annotations

from collections.abc import Callable

from core_adobe_fonts.cmap.ranges import (
    ranges_overlap,
    validate_codespace_range,
)
from core_pdf.impl._impl.fonts.cmap_ranges import (
    MAX_CMAP_RANGE_SPAN,
    expand_range,
    unicode_scalar_or_replacement,
)
from core_pdf.impl._impl.fonts.cmap_tokenizer import (
    CMapProgram,
    cmap_metadata,
    cmap_tokens,
    decode_cmap_hex_token,
    decode_cmap_token,
)
from core_pdf_spec.s_09_fonts.cmap_tounicode import (
    CMapMappingBlock,
    ParsedToUnicodeCMap,
    cmap_mapping_blocks,
    cmap_source_range,
    decode_utf16be,
)
from core_pdf_spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap as PdfToUnicodeCMap


def internal_decode_utf16be(data: bytes) -> str:
    if not data:
        return ""
    if data.startswith(b"\xfe\xff"):
        data = data[2:]
    if len(data) == 1:
        return chr(data[0])
    buffer = data if len(data) % 2 == 0 else b"\x00" + data
    try:
        return decode_utf16be(buffer)
    except UnicodeDecodeError:
        return buffer.decode("utf-16-be", "replace")


def internal_parse_codespace_ranges(program: CMapProgram) -> tuple[tuple[bytes, bytes], ...]:
    code_space_ranges: list[tuple[bytes, bytes]] = []
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
    return tuple(code_space_ranges)


def internal_parse_mapping_blocks(program: CMapProgram, mappings: dict[bytes, str]) -> None:
    """Compile character mappings in order so succeeding definitions win."""
    invalid_range_count = 0
    valid_range_count = 0
    for block in cmap_mapping_blocks(program, include_cid_ranges=True):
        match block.operator:
            case b"beginbfchar":
                internal_parse_bfchar_block(block, mappings)
            case b"beginbfrange":
                block_invalid, block_valid = internal_parse_bfrange_block(block, mappings)
                invalid_range_count += block_invalid
                valid_range_count += block_valid
            case b"begincidrange":
                internal_parse_cidrange_block(block, mappings)
    if invalid_range_count and not valid_range_count:
        raise ValueError("invalid ToUnicode CMap bfrange")


def internal_parse_bfchar_block(block: CMapMappingBlock, mappings: dict[bytes, str]) -> None:
    for record in block.records():
        src_tok = record.source
        dst_tok = record.destination
        try:
            src = decode_cmap_token(src_tok)
            if not src:
                continue
            dst = internal_decode_utf16be(decode_cmap_token(dst_tok))
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
                    dst = internal_decode_utf16be(bytes.fromhex(prefix.decode("ascii")))
                except (ValueError, UnicodeDecodeError):
                    break
                mappings[src] = dst
                break
            continue

        mappings[src] = dst


def internal_parse_bfrange_block(
    block: CMapMappingBlock, mappings: dict[bytes, str]
) -> tuple[int, int]:
    invalid_range_count = int(bool(block.trailing_operand_count))
    valid_range_count = 0
    for record in block.records():
        t1, t2, t3 = record.source, record.source_end, record.destination
        assert t2 is not None
        if not (t1.startswith(b"<") and t2.startswith(b"<")):
            invalid_range_count += 1
            continue
        try:
            source_range = cmap_source_range(decode_cmap_hex_token(t1), decode_cmap_hex_token(t2))
        except (ValueError, UnicodeDecodeError, IndexError):
            invalid_range_count += 1
            continue

        if t3.startswith(b"["):
            dsts = cmap_tokens(t3)
            if not dsts:
                invalid_range_count += 1
                continue
            added = False
            for offset, dst_tok in enumerate(dsts):
                if offset >= source_range.count:
                    break
                try:
                    dst = internal_decode_utf16be(decode_cmap_token(dst_tok))
                except (ValueError, UnicodeDecodeError):
                    continue
                mappings[source_range.source_at(offset)] = dst
                added = True
            if added:
                valid_range_count += 1
            else:
                invalid_range_count += 1
        elif t3.startswith(b"<") or t3.startswith(b"("):
            try:
                base_dst = internal_decode_utf16be(decode_cmap_token(t3))
                expanded = expand_range(
                    source_range.first, source_range.last, source_range.width, base_dst
                )
            except (ValueError, UnicodeDecodeError):
                invalid_range_count += 1
                continue
            mappings.update(expanded)
            valid_range_count += 1
        else:
            invalid_range_count += 1
    return invalid_range_count, valid_range_count


def internal_parse_cidrange_block(block: CMapMappingBlock, mappings: dict[bytes, str]) -> None:
    """Parse numeric CID ranges accepted by PDFMiner in ToUnicode maps.

    Although a conforming ToUnicode CMap normally uses ``bfrange``, some
    producers emit ``cidrange`` records whose numeric destination is a
    Unicode scalar. PostScript CMap parsing accepts those records, and
    PDFMiner consequently exposes their text. Retain that recovery without
    changing ordinary encoding-CMap semantics.
    """
    for record in block.records():
        assert record.source_end is not None
        try:
            source_range = cmap_source_range(
                decode_cmap_hex_token(record.source), decode_cmap_hex_token(record.source_end)
            )
            destination = int(record.destination)
        except (ValueError, UnicodeDecodeError):
            continue
        if source_range.count > MAX_CMAP_RANGE_SPAN:
            continue
        for offset in range(source_range.count):
            mappings[source_range.source_at(offset)] = unicode_scalar_or_replacement(
                destination + offset
            )


class ToUnicodeCMap(PdfToUnicodeCMap):
    __slots__ = ()

    @staticmethod
    def parse_program(data: bytes) -> ParsedToUnicodeCMap:
        return parse_to_unicode_cmap(data)

    def validate_mappings(self) -> None:
        """Keep recovered mappings even when their effective codespace is malformed."""

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        usecmap_resolver: Callable[[str], bytes | None] | None = None,
        inheritance_depth: int = 0,
        ancestor_names: tuple[str, ...] = (),
    ) -> None:
        if inheritance_depth > 16:
            raise ValueError("ToUnicode CMap UseCMap recursion limit exceeded")
        super().__init__(
            data,
            usecmap_resolver=usecmap_resolver,
            inheritance_depth=inheritance_depth,
            ancestor_names=ancestor_names,
        )

    def resolve_parent(
        self,
        name: str,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> PdfToUnicodeCMap | None:
        data = resolver(name) if resolver is not None else None
        return self.load_parent(data, resolver, depth) if data is not None else None

    def load_parent(
        self,
        data: bytes,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> PdfToUnicodeCMap | None:
        try:
            return type(self)(data, usecmap_resolver=resolver, inheritance_depth=depth)
        except ValueError:
            return None

    def decode(self, data: bytes, *, preserve_nulls: bool = False) -> str:
        if not data:
            return ""

        mappings = self.mappings
        lengths = self.decode_lengths or (1,)
        n = len(data)
        out: list[str] = []
        out_append = out.append
        pos = 0
        mappings_get = mappings.get
        while pos < n:
            match_found = False
            for length in lengths:
                if length <= 0 or pos + length > n:
                    continue

                if length == 1:
                    chunk = bytes((data[pos],))
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


def parse_to_unicode_cmap(data: bytes) -> ParsedToUnicodeCMap:
    mappings: dict[bytes, str] = {}
    program = CMapProgram.parse(data)

    internal_parse_mapping_blocks(program, mappings)
    try:
        code_space_ranges = internal_parse_codespace_ranges(program)
    except ValueError:
        # A number of producers write a numerically ordered codespace whose
        # individual bytes are not ordered (for example ``<0083> <020c>``).
        # That is not a valid rectangular CMap codespace, but the explicit
        # bfchar/bfrange entries remain unambiguous.  PostScript CMap parsers
        # such as PDFMiner retain those entries, so recover them instead of
        # rejecting the complete ToUnicode map.  A map with no usable entries
        # still raises, preserving validation for genuinely empty corruption.
        if not mappings:
            raise
        code_space_ranges = ()

    return ParsedToUnicodeCMap(
        code_space_ranges=code_space_ranges,
        mappings=mappings,
        usecmap_name=cmap_metadata(program)[0],
    )
