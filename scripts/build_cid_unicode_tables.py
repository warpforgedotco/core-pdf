#!/usr/bin/env python3
"""Precompute the CID-to-Unicode votes of every packaged CID collection.

Writes src/core_pdf/impl/data/cid_unicode/<Registry>-<Ordering>.bin.gz, which
core_pdf.impl.fonts_unicode.CIDUnicodeMap reads. The sources, overrides, and
voting rules live here and run only to build those tables. Rerun it after
changing them or the Adobe CMap data; tests/src/core_pdf/test_cid_unicode_tables.py
fails until it is.
"""

from __future__ import annotations

import gzip
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/core_pdf/impl/data/cid_unicode"

from core_adobe_fonts.cmap.ranges import code_in_ranges, iter_codespace_range  # noqa: E402
from core_pdf.impl import fonts_unicode  # noqa: E402
from core_pdf.impl.fonts_cmap import (  # noqa: E402
    CMapDecoder,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_scalar_from_cmap_code,
)
from core_pdf.impl.fonts_unicode import (  # noqa: E402
    CID_UNICODE_TABLE_ENTRY,
    CID_UNICODE_TABLE_HEADER,
    CID_UNICODE_TABLE_MAGIC,
)

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


CompactCMap = dict[int, tuple[bytes, ...]]


def compact_cmap_from_decoder(decoder: CMapDecoder) -> CompactCMap:
    code_space_ranges = decoder.code_space_ranges

    def code_is_decodable(code: bytes) -> bool:
        return not code_space_ranges or code_in_ranges(code, code_space_ranges)

    effective_codes_by_cid: defaultdict[int, list[bytes]] = defaultdict(list)
    seen: set[bytes] = set()
    for code, cid in decoder.cid_mappings.items():
        if not code_is_decodable(code):
            continue
        seen.add(code)
        effective_codes_by_cid[cid].append(code)
    for cid_range in reversed(decoder.cid_ranges):
        for offset, code in enumerate(iter_codespace_range(cid_range.start, cid_range.end)):
            if code in seen or not code_is_decodable(code):
                continue
            seen.add(code)
            effective_codes_by_cid[cid_range.first_cid + offset].append(code)
    return {cid: tuple(codes) for cid, codes in effective_codes_by_cid.items()}


@cache
def compact_cmap(name: str) -> CompactCMap | None:
    if resolve_cmap_resource(name) is None:
        return None
    decoder = resolve_cmap_decoder(name)
    return compact_cmap_from_decoder(decoder) if decoder is not None else None


def preferred_unicode_for_cid(cmap_name: str, codec: str, cid: int) -> str | None:
    cmap = compact_cmap(cmap_name)
    if cmap is None:
        return None
    candidates = {
        text
        for code in cmap.get(cid, ())
        if (text := unicode_scalar_from_cmap_code(code, codec)) is not None
    }
    if not candidates:
        return None
    return max(candidates, key=lambda text: (unicode_candidate_preference(text), -ord(text)))


def tally_cid_votes(
    sources: Iterable[CMapUnicodeSource], cid: int, *, unweighted: bool = False
) -> Counter[str]:
    candidates: Counter[str] = Counter()
    for cmap_name, codec, weight in sources:
        if (weight <= 0) != unweighted:
            continue
        text = preferred_unicode_for_cid(cmap_name, codec, cid)
        if text is not None:
            candidates[text] += 1 if unweighted else weight
    return candidates


def vote_from_sources(registry: str, ordering: str, vertical: bool, cid: int) -> str | None:
    override = CID_COLLECTION_UNICODE_OVERRIDES.get((registry, ordering), {}).get(cid)
    if override is not None:
        return override
    collection = CID_COLLECTION_UNICODE_SOURCES.get((registry, ordering))
    if collection is None:
        return None
    sources = collection[vertical]
    opposite_sources = collection[not vertical]
    candidates = (
        tally_cid_votes(sources, cid)
        or tally_cid_votes(opposite_sources, cid)
        or tally_cid_votes((*sources, *opposite_sources), cid, unweighted=True)
    )
    if not candidates:
        return None
    ranked = {
        text: (weight, unicode_candidate_preference(text), -ord(text))
        for text, weight in candidates.items()
    }
    return max(ranked, key=ranked.__getitem__)


def encode_cid_unicode_table(horizontal: dict[int, str], vertical: dict[int, str]) -> bytes:
    changes = {
        cid: vertical.get(cid, "")
        for cid in sorted(horizontal.keys() | vertical.keys())
        if vertical.get(cid) != horizontal.get(cid)
    }
    payload = bytearray(
        CID_UNICODE_TABLE_HEADER.pack(CID_UNICODE_TABLE_MAGIC, len(horizontal), len(changes))
    )
    for cid, text in sorted(horizontal.items()):
        payload += CID_UNICODE_TABLE_ENTRY.pack(cid, ord(text))
    for cid, text in changes.items():
        payload += CID_UNICODE_TABLE_ENTRY.pack(cid, ord(text) if text else 0)
    # The level is explicit because the packaged tables are compared byte for
    # byte, and Python 3.15 lowered gzip.compress's default from 9 to 6.
    return gzip.compress(bytes(payload), compresslevel=9, mtime=0)


def collection_cids(registry: str, ordering: str) -> set[int]:
    cids = set(CID_COLLECTION_UNICODE_OVERRIDES.get((registry, ordering), {}))
    for sources in CID_COLLECTION_UNICODE_SOURCES[(registry, ordering)].values():
        for cmap_name, _codec, _weight in sources:
            cmap = compact_cmap(cmap_name)
            if cmap is not None:
                cids.update(cmap)
    return cids


def collection_votes(registry: str, ordering: str, *, vertical: bool) -> dict[int, str]:
    votes: dict[int, str] = {}
    for cid in sorted(collection_cids(registry, ordering)):
        text = vote_from_sources(registry, ordering, vertical, cid)
        if text is None:
            continue
        if len(text) != 1:
            raise ValueError(f"{registry}-{ordering} CID {cid} votes for {text!r}, not one scalar")
        votes[cid] = text
    return votes


def build_table(registry: str, ordering: str) -> bytes:
    return encode_cid_unicode_table(
        collection_votes(registry, ordering, vertical=False),
        collection_votes(registry, ordering, vertical=True),
    )


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for registry, ordering in sorted(CID_COLLECTION_UNICODE_SOURCES):
        path = OUTPUT / fonts_unicode.cid_unicode_table_name(registry, ordering)
        path.write_bytes(build_table(registry, ordering))
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    build()
