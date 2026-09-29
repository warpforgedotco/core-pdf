# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from binascii import unhexlify
from collections.abc import Iterable
from functools import lru_cache

from core_adobe_fonts.cmap import decoder, tokenizer
from core_adobe_fonts.cmap.decoder import CMapResourceResolver
from core_adobe_fonts.cmap.ranges import ranges_overlap, validate_codespace_range
from core_adobe_fonts.cmap.resources import resolve_cmap_resource
from core_adobe_fonts.cmap.tokenizer import CMapBlock, CMapToken, iter_cmap_tokens
from core_pdf_cythonized import scan_to_unicode_cmap
from core_pdf_spec.s_07_syntax_primitives.scanning import read_literal_string
from core_pdf_spec.s_09_fonts import cmap_tounicode
from core_pdf_spec.s_09_fonts.cmap_tounicode import (
    CMapMappingBlock,
    ParsedToUnicodeCMap,
    cmap_mapping_blocks,
    cmap_source_range,
    decode_utf16be,
)
from core_pdf_spec.s_09_fonts.font_program_truetype import is_unicode_scalar

RESOURCE_PACKAGE = "core_adobe_fonts.cmap.data"


CMapUnicodeSource = tuple[str, str, int]


CID_COLLECTION_UNICODE_SOURCES: dict[tuple[str, str], dict[bool, tuple[CMapUnicodeSource, ...]]] = {
    ("Adobe", "GB1"): {
        False: (
            ("UniGB-UTF32-H", "utf-32-be", 3),
            ("GBK2K-H", "gb18030", 0),
            ("GBK-EUC-H", "gbk", 0),
            ("GBKp-EUC-H", "gbk", 0),
            ("GB-EUC-H", "gb2312", 0),
            ("GBpc-EUC-H", "gb2312", 0),
            ("GB-H", "gb2312_7bit", 0),
        ),
        True: (
            ("UniGB-UTF32-V", "utf-32-be", 3),
            ("GBK2K-V", "gb18030", 0),
            ("GBK-EUC-V", "gbk", 0),
            ("GBKp-EUC-V", "gbk", 0),
            ("GB-EUC-V", "gb2312", 0),
            ("GBpc-EUC-V", "gb2312", 0),
            ("GB-V", "gb2312_7bit", 0),
        ),
    },
    ("Adobe", "CNS1"): {
        False: (
            ("UniCNS-UTF32-H", "utf-32-be", 3),
            ("HKscs-B5-H", "big5hkscs", 0),
            ("B5-H", "big5", 0),
            ("B5pc-H", "big5", 0),
            ("ETen-B5-H", "cp950", 0),
            ("ETenms-B5-H", "cp950", 0),
        ),
        True: (
            ("UniCNS-UTF32-V", "utf-32-be", 3),
            ("HKscs-B5-V", "big5hkscs", 0),
            ("B5-V", "big5", 0),
            ("B5pc-V", "big5", 0),
            ("ETen-B5-V", "cp950", 0),
            ("ETenms-B5-V", "cp950", 0),
        ),
    },
    ("Adobe", "Japan1"): {
        False: (
            ("90ms-RKSJ-H", "cp932", 3),
            ("EUC-H", "euc_jp", 3),
            ("RKSJ-H", "cp932", 0),
            ("78-RKSJ-H", "cp932", 0),
            ("78ms-RKSJ-H", "cp932", 0),
            ("Add-RKSJ-H", "cp932", 0),
            ("Ext-RKSJ-H", "cp932", 0),
            ("90msp-RKSJ-H", "cp932", 0),
            ("83pv-RKSJ-H", "cp932", 0),
            ("90pv-RKSJ-H", "cp932", 0),
            ("78-EUC-H", "euc_jp", 0),
            ("H", "jis_x0208", 0),
            ("78-H", "jis_x0208", 0),
            ("Add-H", "jis_x0208", 0),
            ("Ext-H", "jis_x0208", 0),
            ("NWP-H", "jis_x0208", 0),
            ("UniJIS-UTF32-H", "utf-32-be", 1),
            ("UniJIS2004-UTF32-H", "utf-32-be", 1),
            ("UniJISX0213-UTF32-H", "utf-32-be", 1),
            ("UniJISX02132004-UTF32-H", "utf-32-be", 1),
        ),
        True: (
            ("90ms-RKSJ-V", "cp932", 3),
            ("EUC-V", "euc_jp", 3),
            ("RKSJ-V", "cp932", 0),
            ("78-RKSJ-V", "cp932", 0),
            ("78ms-RKSJ-V", "cp932", 0),
            ("Add-RKSJ-V", "cp932", 0),
            ("Ext-RKSJ-V", "cp932", 0),
            ("90msp-RKSJ-V", "cp932", 0),
            ("90pv-RKSJ-V", "cp932", 0),
            ("78-EUC-V", "euc_jp", 0),
            ("V", "jis_x0208", 0),
            ("78-V", "jis_x0208", 0),
            ("Add-V", "jis_x0208", 0),
            ("Ext-V", "jis_x0208", 0),
            ("NWP-V", "jis_x0208", 0),
            ("UniJIS-UTF32-V", "utf-32-be", 1),
            ("UniJIS2004-UTF32-V", "utf-32-be", 1),
            ("UniJISX0213-UTF32-V", "utf-32-be", 1),
            ("UniJISX02132004-UTF32-V", "utf-32-be", 1),
        ),
    },
    ("Adobe", "Japan2"): {
        False: (
            ("Hojo-EUC-H", "euc_jp", 3),
            ("UniHojo-UTF32-H", "utf-32-be", 1),
        ),
        True: (
            ("Hojo-EUC-V", "euc_jp", 3),
            ("UniHojo-UTF32-V", "utf-32-be", 1),
        ),
    },
    ("Adobe", "Manga1"): {
        False: (("UniManga-UTF32-H", "utf-32-be", 1),),
        True: (("UniManga-UTF32-V", "utf-32-be", 1),),
    },
    ("Adobe", "Korea1"): {
        False: (
            ("UniKS-UTF32-H", "utf-32-be", 3),
            ("KSCms-UHC-H", "cp949", 0),
            ("KSC-Johab-H", "johab", 0),
            ("KSC-EUC-H", "euc_kr", 0),
            ("KSCpc-EUC-H", "euc_kr", 0),
            ("KSC-H", "euc_kr_7bit", 0),
        ),
        True: (
            ("UniKS-UTF32-V", "utf-32-be", 3),
            ("KSCms-UHC-V", "cp949", 0),
            ("KSC-Johab-V", "johab", 0),
            ("KSC-EUC-V", "euc_kr", 0),
            ("KSCpc-EUC-V", "euc_kr", 0),
            ("KSC-V", "euc_kr_7bit", 0),
        ),
    },
    ("Adobe", "KR"): {
        False: (("UniAKR-UTF32-H", "utf-32-be", 3),),
        True: (),
    },
}


CID_COLLECTION_UNICODE_OVERRIDES: dict[tuple[str, str], dict[int, str]] = {
    ("Adobe", "GB1"): {
        115: "\u3008",
        116: "\u3009",
        10060: "\u2ff0",
        10061: "\u2ff1",
        10062: "\u2ff2",
        10063: "\u2ff3",
        10064: "\u2ff4",
        10065: "\u2ff5",
        10066: "\u2ff6",
        10067: "\u2ff7",
        10068: "\u2ff8",
        10069: "\u2ff9",
        10070: "\u2ffa",
        10071: "\u2ffb",
        22047: "\u2e81",
        22051: "\u2e84",
        22054: "\u2e88",
        22055: "\u2e8b",
        22060: "\u2e8c",
        22061: "\u2e97",
        22074: "\u2ea7",
        22077: "\u2eaa",
        22080: "\u2eae",
        22082: "\u2eb3",
        22083: "\u2eb6",
        22084: "\u2eb7",
        22088: "\u2ebb",
        22098: "\u2eca",
    },
    ("Adobe", "CNS1"): {
        148: "\u3008",
        149: "\u3009",
    },
    ("Adobe", "Japan1"): {
        114: "\u2012",
        127: "\u0301",
        138: "\u0336",
        226: "\u0305",
        682: "\u3008",
        683: "\u3009",
        693: "\u2212",
        8206: "\u27a1",
    },
}


PREDEFINED_CMAP_UNICODE_CODECS: dict[str, str] = {
    "GB-EUC-H": "gb2312",
    "GB-EUC-V": "gb2312",
    "GB-H": "gb2312_7bit",
    "GB-V": "gb2312_7bit",
    "GBK-EUC-H": "gbk",
    "GBK-EUC-V": "gbk",
    "GBKp-EUC-H": "gbk",
    "GBKp-EUC-V": "gbk",
    "GBK2K-H": "gb18030",
    "GBK2K-V": "gb18030",
    "GBpc-EUC-H": "gb2312",
    "GBpc-EUC-V": "gb2312",
    "B5-H": "big5",
    "B5-V": "big5",
    "B5pc-H": "big5",
    "B5pc-V": "big5",
    "ETen-B5-H": "cp950",
    "ETen-B5-V": "cp950",
    "ETenms-B5-H": "cp950",
    "ETenms-B5-V": "cp950",
    "KSC-EUC-H": "euc_kr",
    "KSC-EUC-V": "euc_kr",
    "KSC-H": "euc_kr_7bit",
    "KSC-V": "euc_kr_7bit",
    "KSC-Johab-H": "johab",
    "KSC-Johab-V": "johab",
    "KSCms-UHC-H": "cp949",
    "KSCms-UHC-V": "cp949",
    "KSCms-UHC-HW-H": "cp949",
    "KSCms-UHC-HW-V": "cp949",
    "KSCpc-EUC-H": "euc_kr",
    "KSCpc-EUC-V": "euc_kr",
    "90ms-RKSJ-H": "cp932",
    "90ms-RKSJ-V": "cp932",
    "90msp-RKSJ-H": "cp932",
    "90msp-RKSJ-V": "cp932",
    "RKSJ-H": "cp932",
    "RKSJ-V": "cp932",
    "78-RKSJ-H": "cp932",
    "78-RKSJ-V": "cp932",
    "78ms-RKSJ-H": "cp932",
    "78ms-RKSJ-V": "cp932",
    "Add-RKSJ-H": "cp932",
    "Add-RKSJ-V": "cp932",
    "Ext-RKSJ-H": "cp932",
    "Ext-RKSJ-V": "cp932",
    "EUC-H": "euc_jp",
    "EUC-V": "euc_jp",
    "78-EUC-H": "euc_jp",
    "78-EUC-V": "euc_jp",
    "H": "jis_x0208",
    "V": "jis_x0208",
    "78-H": "jis_x0208",
    "78-V": "jis_x0208",
    "Add-H": "jis_x0208",
    "Add-V": "jis_x0208",
    "Ext-H": "jis_x0208",
    "Ext-V": "jis_x0208",
    "NWP-H": "jis_x0208",
    "NWP-V": "jis_x0208",
}


@lru_cache(maxsize=64)
def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    if name in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(byte_width=2, wmode=int(name.endswith("-V")))
    if name in {"OneByteIdentityH", "OneByteIdentityV"}:
        return CMapDecoder.identity(byte_width=1, wmode=int(name.endswith("V")))
    cmap_data = resolve_cmap_resource(name)
    if cmap_data is None:
        return None
    try:
        return CMapDecoder(
            cmap_data,
            usecmap_resolver=resolve_cmap_resource,
        )
    except ValueError:
        return None


def unicode_scalar_from_cmap_code(code: bytes, codec: str) -> str | None:
    try:
        if codec in {"gb2312_7bit", "euc_kr_7bit"}:
            if len(code) == 1 and code[0] < 0x80:
                text = code.decode("ascii")
            elif len(code) == 2 and 0x21 <= code[0] <= 0x7E and 0x21 <= code[1] <= 0x7E:
                base_codec = "gb2312" if codec == "gb2312_7bit" else "euc_kr"
                text = bytes(byte | 0x80 for byte in code).decode(base_codec)
            else:
                return None
        elif codec == "jis_x0208":
            if len(code) == 1 and code[0] < 0x80:
                text = code.decode("ascii")
            elif len(code) == 2:
                text = (b"\x1b$B" + code + b"\x1b(B").decode("iso2022_jp")
            else:
                return None
        else:
            text = code.decode(codec)
    except UnicodeError:
        return None
    if len(text) != 1:
        return None
    codepoint = ord(text)
    if 0xD800 <= codepoint <= 0xDFFF:
        return None
    return text


def predefined_cmap_unicode(name: str | None, code: bytes) -> str | None:
    if name is None:
        return None
    codec = PREDEFINED_CMAP_UNICODE_CODECS.get(name)
    if codec is None and name.startswith(
        ("UniAKR", "UniCNS", "UniGB", "UniHojo", "UniJIS", "UniKS", "UniManga")
    ):
        if "-UTF8-" in name:
            codec = "utf-8"
        elif "-UTF16-" in name or "-UCS2-" in name:
            codec = "utf-16-be"
        elif "-UTF32-" in name:
            codec = "utf-32-be"
    if codec is None:
        return None
    return unicode_scalar_from_cmap_code(code, codec)


def unicode_candidate_preference(text: str) -> tuple[int, int, int, int, int]:
    codepoint = ord(text)
    is_unified_ideograph = (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    )
    is_compatibility_form = (
        0x2E80 <= codepoint <= 0x2FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0xFE10 <= codepoint <= 0xFE4F
        or 0x2F800 <= codepoint <= 0x2FA1F
    )
    is_combining = 0x0300 <= codepoint <= 0x036F
    is_private_use = (
        0xE000 <= codepoint <= 0xF8FF
        or 0xF0000 <= codepoint <= 0xFFFFD
        or 0x100000 <= codepoint <= 0x10FFFD
    )
    is_ascii = codepoint < 0x80
    return (
        int(is_unified_ideograph),
        int(not is_private_use),
        int(not is_compatibility_form),
        int(not is_combining),
        int(is_ascii),
    )


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


def parse_codespace_ranges(
    blocks: Iterable[list[bytes]],
) -> tuple[tuple[bytes, bytes], ...]:
    code_space_ranges: list[tuple[bytes, bytes]] = []
    saw_codespace_block = False
    valid_range_count = 0
    for tokens in blocks:
        saw_codespace_block = True
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


class ToUnicodeCMap(cmap_tounicode.ToUnicodeCMap):
    __slots__ = ()

    max_inheritance_depth = 16

    @staticmethod
    def parse_program(data: bytes) -> ParsedToUnicodeCMap:
        return parse_to_unicode_cmap(data)

    def validate_mappings(self) -> None:
        pass

    def reject_parent(self, reason: str) -> cmap_tounicode.ToUnicodeCMap | None:  # noqa: ARG002
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
    scanned = scan_to_unicode_cmap(data if type(data) is bytes else bytes(data))
    if scanned is None:
        return parse_to_unicode_program(CMapProgram.parse(data))
    mappings, codespace_blocks, usecmap_name = scanned
    return to_unicode_cmap(mappings, codespace_blocks, usecmap_name)


def parse_to_unicode_program(program: CMapProgram) -> ParsedToUnicodeCMap:
    mappings: dict[bytes, str] = {}
    parse_mapping_blocks(program, mappings)
    return to_unicode_cmap(
        mappings,
        (
            block.token_values()
            for block in program.blocks(b"begincodespacerange", b"endcodespacerange")
        ),
        cmap_metadata(program)[0],
    )


def to_unicode_cmap(
    mappings: dict[bytes, str],
    codespace_blocks: Iterable[list[bytes]],
    usecmap_name: str | None,
) -> ParsedToUnicodeCMap:
    try:
        code_space_ranges = parse_codespace_ranges(codespace_blocks)
    except ValueError:
        if not mappings:
            raise
        code_space_ranges = ()
    return ParsedToUnicodeCMap(
        code_space_ranges=code_space_ranges,
        mappings=mappings,
        usecmap_name=usecmap_name,
    )


MAX_CMAP_RANGE_SPAN = 65536


def unicode_scalar_or_replacement(codepoint: int) -> str:
    return chr(codepoint) if is_unicode_scalar(codepoint) else "\ufffd"


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


LEGACY_EOL_PAIR = re.compile(rb"\r\n|\n\r")


def collect_cmap_tokens(data: bytes, *, group_arrays: bool) -> tuple[CMapToken, ...]:
    tokens: list[CMapToken] = []
    try:
        for token in iter_cmap_tokens(data, group_arrays=group_arrays):
            tokens.append(token)  # noqa: PERF402
    except ValueError:
        pass
    return tuple(tokens)


def decode_cmap_hex_token(token: bytes) -> bytes:
    if not token.startswith(b"<") or not token.endswith(b">"):
        token = b"<" + token[1:-1] + b">"
    return tokenizer.decode_cmap_hex_token(token)


def decode_cmap_token(token: bytes) -> bytes:
    if token.startswith(b"<"):
        return decode_cmap_hex_token(token)
    if not token.startswith(b"("):
        return tokenizer.decode_cmap_token(token)
    raw = memoryview(LEGACY_EOL_PAIR.sub(b"\r\n", token))
    if len(raw) < 2:
        raise ValueError("invalid PDF literal string")
    value, _ = read_literal_string(raw, 0, len(raw))
    if value is None:
        raise ValueError("unterminated PDF literal string")
    return value


class CMapProgram(tokenizer.CMapProgram):
    __slots__ = ()

    @classmethod
    def read_tokens(cls, source: bytes) -> tuple[CMapToken, ...]:
        return collect_cmap_tokens(source, group_arrays=True)

    @classmethod
    def validate_program(
        cls, tokens: tuple[CMapToken, ...], scoped_tokens: tuple[CMapToken, ...]
    ) -> None:
        pass

    def block_count(self, begin_index: int) -> int:  # noqa: ARG002
        return 0

    def validate_block(
        self, begin_keyword: bytes, block_tokens: list[CMapToken], declared_count: int
    ) -> None:
        pass

    def reject_unterminated_block(self) -> None:
        pass


def cmap_metadata(data: bytes | tokenizer.CMapProgram) -> tuple[str | None, int | None]:
    program = data if isinstance(data, tokenizer.CMapProgram) else CMapProgram.parse(data)
    words = [token.value for token in program.tokens if token.kind == "word"]
    usecmap_name: str | None = None
    wmode: int | None = None
    usecmap_checked = False
    wmode_checked = False
    for index, word in enumerate(words):
        if not usecmap_checked and index > 0 and word == b"usecmap":
            usecmap_checked = True
            name = words[index - 1]
            if name.startswith(b"/"):
                try:
                    usecmap_name = name[1:].decode("latin-1")
                except UnicodeDecodeError:
                    usecmap_name = None
        if not wmode_checked and index + 2 < len(words) and word == b"/WMode":
            wmode_checked = True
            if words[index + 2] == b"def":
                try:
                    value = int(words[index + 1])
                except ValueError:
                    continue
                if value in {0, 1}:
                    wmode = value
        if usecmap_checked and wmode_checked:
            break
    return usecmap_name, wmode


def cmap_tokens(
    data: bytes, *, include_arrays: bool = False, include_words: bool = False
) -> list[bytes]:
    tokens = collect_cmap_tokens(data, group_arrays=include_arrays)
    return CMapBlock(data, tokens).token_values(
        include_arrays=include_arrays, include_words=include_words
    )


class CMapDecoder(decoder.CMapDecoder):
    __slots__ = ()

    max_inheritance_depth = 5

    @staticmethod
    def parse_program(data: bytes) -> CMapProgram:
        return CMapProgram.parse(data)

    @staticmethod
    def program_metadata(program: tokenizer.CMapProgram) -> tuple[str | None, int | None]:
        return cmap_metadata(program)

    def validate_mappings(self) -> None:
        pass

    def reject_mapping(self, reason: str, cause: BaseException | None = None) -> None:
        pass

    @staticmethod
    def decode_codespace_token(token: bytes) -> bytes:
        return decode_cmap_hex_token(token)

    @staticmethod
    def resolve_usecmap(
        name: str,
        *,
        usecmap_resolver: CMapResourceResolver | None,
        depth: int,
        ancestor_names: tuple[str, ...] = (),
    ) -> decoder.CMapDecoder | None:
        if name in {"OneByteIdentityH", "OneByteIdentityV"}:
            return CMapDecoder.identity(byte_width=1, wmode=int(name.endswith("V")))
        if name in ancestor_names:
            return None
        if usecmap_resolver is not None:
            resolved = usecmap_resolver(name)
            if resolved is not None:
                return CMapDecoder(
                    resolved,
                    usecmap_resolver=usecmap_resolver,
                    inheritance_depth=depth,
                    ancestor_names=(*ancestor_names, name),
                )
        if name in {"Identity-H", "Identity-V"}:
            return CMapDecoder.identity(byte_width=2, wmode=int(name.endswith("-V")))
        return None
