from __future__ import annotations

from binascii import unhexlify
from collections.abc import Callable

from core_adobe_fonts.cmap.ranges import (
    ranges_overlap,
    validate_codespace_range,
)
from core_pdf.impl.fonts_cmap_tokenizer import (
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


def decode_utf16be_text(data: bytes) -> str:
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


def parse_codespace_ranges(program: CMapProgram) -> tuple[tuple[bytes, bytes], ...]:
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
            except ValueError, UnicodeDecodeError:
                continue
            if any(ranges_overlap((start, end), existing) for existing in code_space_ranges):
                raise ValueError("invalid ToUnicode CMap codespacerange")
            code_space_ranges.append((start, end))
            valid_range_count += 1
    if saw_codespace_block and valid_range_count == 0:
        raise ValueError("invalid ToUnicode CMap codespacerange")
    return tuple(code_space_ranges)


def parse_mapping_blocks(program: CMapProgram, mappings: dict[bytes, str]) -> None:
    invalid_range_count = 0
    valid_range_count = 0
    for block in cmap_mapping_blocks(program, include_cid_ranges=True):
        match block.operator:
            case b"beginbfchar":
                parse_bfchar_block(block, mappings)
            case b"beginbfrange":
                block_invalid, block_valid = parse_bfrange_block(block, mappings)
                invalid_range_count += block_invalid
                valid_range_count += block_valid
            case b"begincidrange":
                parse_cidrange_block(block, mappings)
    if invalid_range_count and not valid_range_count:
        raise ValueError("invalid ToUnicode CMap bfrange")


def parse_bfchar_block(block: CMapMappingBlock, mappings: dict[bytes, str]) -> None:
    # A bfchar block's operands are source, destination pairs, and records()
    # made a frozen record of each: they are paired here directly. A hex
    # string whose inside is an even run of letters and digits decodes as
    # unhexlify of that inside, which is what decode_cmap_token comes to
    # through three frames -- and raises the same ValueError when it is not
    # hex; anything else still takes decode_cmap_token.
    operands = block.operands
    for src_tok, dst_tok in zip(operands[0::2], operands[1::2], strict=False):
        try:
            raw = src_tok[1:-1]
            if src_tok[:1] == b"<" and raw.isalnum() and not len(raw) & 1:
                src = unhexlify(raw)
            else:
                src = decode_cmap_token(src_tok)
            if not src:
                continue
            raw = dst_tok[1:-1]
            if dst_tok[:1] == b"<" and raw.isalnum() and not len(raw) & 1:
                dst = decode_utf16be_text(unhexlify(raw))
            else:
                dst = decode_utf16be_text(decode_cmap_token(dst_tok))
        except ValueError, UnicodeDecodeError:
            if dst_tok.startswith(b"<") and b"<" in dst_tok[1:]:
                prefix = dst_tok[1 : dst_tok.find(b"<", 1)]
                try:
                    src = decode_cmap_token(src_tok)
                    if not src:
                        continue
                    if len(prefix) % 2:
                        prefix += b"0"
                    dst = decode_utf16be_text(bytes.fromhex(prefix.decode("ascii")))
                except ValueError, UnicodeDecodeError:
                    break
                mappings[src] = dst
                break
            continue

        mappings[src] = dst


def parse_bfrange_block(block: CMapMappingBlock, mappings: dict[bytes, str]) -> tuple[int, int]:
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
        except ValueError, UnicodeDecodeError, IndexError:
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
                    dst = decode_utf16be_text(decode_cmap_token(dst_tok))
                except ValueError, UnicodeDecodeError:
                    continue
                mappings[source_range.source_at(offset)] = dst
                added = True
            if added:
                valid_range_count += 1
            else:
                invalid_range_count += 1
        elif t3.startswith((b"<", b"(")):
            try:
                base_dst = decode_utf16be_text(decode_cmap_token(t3))
                expanded = expand_range(
                    source_range.first, source_range.last, source_range.width, base_dst
                )
            except ValueError, UnicodeDecodeError:
                invalid_range_count += 1
                continue
            mappings.update(expanded)
            valid_range_count += 1
        else:
            invalid_range_count += 1
    return invalid_range_count, valid_range_count


def parse_cidrange_block(block: CMapMappingBlock, mappings: dict[bytes, str]) -> None:
    for record in block.records():
        assert record.source_end is not None
        try:
            source_range = cmap_source_range(
                decode_cmap_hex_token(record.source), decode_cmap_hex_token(record.source_end)
            )
            destination = int(record.destination)
        except ValueError, UnicodeDecodeError:
            continue
        if source_range.count > MAX_CMAP_RANGE_SPAN:
            continue
        for offset in range(source_range.count):
            mappings[source_range.source_at(offset)] = unicode_scalar_or_replacement(
                destination + offset
            )


class ToUnicodeCMap(PdfToUnicodeCMap):
    __slots__ = ()

    max_inheritance_depth = 16

    @staticmethod
    def parse_program(data: bytes) -> ParsedToUnicodeCMap:
        return parse_to_unicode_cmap(data)

    def validate_mappings(self) -> None:
        pass

    def reject_parent(self, reason: str) -> PdfToUnicodeCMap | None:  # noqa: ARG002
        return None

    def load_parent(
        self,
        data: bytes,
        resolver: Callable[[str], bytes | None] | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> PdfToUnicodeCMap | None:
        try:
            return super().load_parent(data, resolver, depth, ancestor_names)
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

    parse_mapping_blocks(program, mappings)
    try:
        code_space_ranges = parse_codespace_ranges(program)
    except ValueError:
        if not mappings:
            raise
        code_space_ranges = ()

    return ParsedToUnicodeCMap(
        code_space_ranges=code_space_ranges,
        mappings=mappings,
        usecmap_name=cmap_metadata(program)[0],
    )


MAX_CMAP_RANGE_SPAN = 65536


def unicode_scalar_or_replacement(codepoint: int) -> str:
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return chr(codepoint)
    return "\ufffd"


def expand_range(start: int, end: int, source_hex_len: int, base_dst: str) -> dict[bytes, str]:
    mapping: dict[bytes, str] = {}
    if (
        source_hex_len <= 0
        or end >= 1 << (source_hex_len * 8)
        or end < start
        or end - start + 1 > MAX_CMAP_RANGE_SPAN
    ):
        raise ValueError("invalid ToUnicode CMap bfrange")
    prefix = "".join(unicode_scalar_or_replacement(ord(c)) for c in base_dst[:-1])
    final_scalar = ord(base_dst[-1]) if base_dst else None
    for i in range(start, end + 1):
        mapping[i.to_bytes(source_hex_len, "big")] = (
            ""
            if final_scalar is None
            else prefix + unicode_scalar_or_replacement(final_scalar + i - start)
        )
    return mapping
