from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from types import MappingProxyType

from core_pdf.impl.third_party.cid.cmap import CMapDecoder, iter_codespace_range

RESOURCE_PACKAGE = "core_pdf.impl.third_party.cid"
CMapUnicodeSource = tuple[str, str, int]
CID_COLLECTION_UNICODE_SOURCES: dict[
    tuple[str, str], dict[bool, tuple[CMapUnicodeSource, ...]]
] = {
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
        10060: "\u2FF0",
        10061: "\u2FF1",
        10062: "\u2FF2",
        10063: "\u2FF3",
        10064: "\u2FF4",
        10065: "\u2FF5",
        10066: "\u2FF6",
        10067: "\u2FF7",
        10068: "\u2FF8",
        10069: "\u2FF9",
        10070: "\u2FFA",
        10071: "\u2FFB",
        22047: "\u2E81",
        22051: "\u2E84",
        22054: "\u2E88",
        22055: "\u2E8B",
        22060: "\u2E8C",
        22061: "\u2E97",
        22074: "\u2EA7",
        22077: "\u2EAA",
        22080: "\u2EAE",
        22082: "\u2EB3",
        22083: "\u2EB6",
        22084: "\u2EB7",
        22088: "\u2EBB",
        22098: "\u2ECA",
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
        8206: "\u27A1",
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


@lru_cache(maxsize=1)
def cmap_resource_root() -> Traversable:
    return resources.files(RESOURCE_PACKAGE).joinpath("resources")


def normalized_cmap_name(name: str) -> str:
    return name[1:] if name.startswith("/") else name


@lru_cache(maxsize=1)
def cmap_resource_index() -> dict[str, Traversable]:
    root = cmap_resource_root()
    if not root.is_dir():
        return {}

    index: dict[str, Traversable] = {}
    candidates: list[tuple[Traversable, str | None]] = [(root, None)]
    while candidates:
        current, parent_name = candidates.pop()
        for child in current.iterdir():
            if child.is_dir():
                candidates.append((child, child.name))
                continue
            if parent_name != "CMap":
                continue
            name = child.name
            existing = index.get(name)
            if existing is None or "/deprecated/" in str(existing):
                index[name] = child
    return index


def resolve_cmap_resource(name: str) -> bytes | None:
    resource = cmap_resource_index().get(normalized_cmap_name(name))
    if resource is None:
        return None
    return resource.read_bytes()


@lru_cache(maxsize=256)
def resolve_cmap_decoder(name: str) -> CMapDecoder | None:
    normalized_name = normalized_cmap_name(name)
    if normalized_name in {"Identity-H", "Identity-V"}:
        return CMapDecoder.identity(byte_width=2, wmode=int(normalized_name.endswith("-V")))
    if normalized_name in {"OneByteIdentityH", "OneByteIdentityV"}:
        return CMapDecoder.identity(byte_width=1, wmode=int(normalized_name.endswith("V")))
    cmap_data = resolve_cmap_resource(normalized_name)
    if cmap_data is None:
        return None
    try:
        return CMapDecoder(
            cmap_data,
            usecmap_resolver=resolve_cmap_decoder,
        )
    except ValueError:
        return None


def has_cmap_resource(name: str) -> bool:
    return normalized_cmap_name(name) in cmap_resource_index()


def unicode_scalar_from_cmap_code(code: bytes, codec: str) -> str | None:
    try:
        if codec in {"gb2312_7bit", "euc_kr_7bit"}:
            if len(code) == 1 and code[0] < 0x80:
                text = code.decode("ascii")
            elif len(code) == 2 and all(0x21 <= byte <= 0x7E for byte in code):
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
    normalized_name = normalized_cmap_name(name)
    codec = PREDEFINED_CMAP_UNICODE_CODECS.get(normalized_name)
    if codec is None and normalized_name.startswith(
        ("UniAKR", "UniCNS", "UniGB", "UniHojo", "UniJIS", "UniKS", "UniManga")
    ):
        if "-UTF8-" in normalized_name:
            codec = "utf-8"
        elif "-UTF16-" in normalized_name or "-UCS2-" in normalized_name:
            codec = "utf-16-be"
        elif "-UTF32-" in normalized_name:
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


def iter_effective_cid_mappings(cmap: CMapDecoder) -> Iterator[tuple[bytes, int]]:
    seen: set[bytes] = set()
    for code in cmap.cid_mappings:
        cid = cmap.mapped_cid(code)
        if cid is not None:
            seen.add(code)
            yield code, cid
    for cid_range in reversed(cmap.cid_ranges):
        for code in iter_codespace_range(cid_range.start, cid_range.end):
            if code in seen:
                continue
            seen.add(code)
            yield code, cid_range.cid_for(code)


def preferred_unicode_by_cid(cmap: CMapDecoder, codec: str) -> dict[int, str]:
    candidates: defaultdict[int, set[str]] = defaultdict(set)
    for code, cid in iter_effective_cid_mappings(cmap):
        text = unicode_scalar_from_cmap_code(code, codec)
        if text is not None:
            candidates[cid].add(text)
    return {
        cid: max(texts, key=lambda text: (unicode_candidate_preference(text), -ord(text)))
        for cid, texts in candidates.items()
    }


@lru_cache(maxsize=32)
def resolve_cid_unicode_map(
    registry: str,
    ordering: str,
    *,
    vertical: bool = False,
) -> Mapping[int, str] | None:
    """Resolve a CID collection to Unicode using its bundled Adobe CMaps.

    PDF encoding CMaps map character codes to collection-specific CIDs.  The
    Source-encoding and UTF-32 CMaps describe the inverse relationship needed
    when a CID font omits ``/ToUnicode``.  Source encodings receive extra vote
    weight because they disambiguate compatibility glyphs and vertical forms.
    """
    collection_sources = CID_COLLECTION_UNICODE_SOURCES.get((registry, ordering))
    if collection_sources is None:
        return None
    sources = collection_sources[vertical]
    opposite_sources = collection_sources[not vertical]

    candidates: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for cmap_name, codec, weight in sources:
        if weight <= 0:
            continue
        cmap = resolve_cmap_decoder(cmap_name)
        if cmap is None:
            continue
        for cid, text in preferred_unicode_by_cid(cmap, codec).items():
            candidates[cid][text] += weight
    primary_cids = set(candidates)
    for cmap_name, codec, weight in opposite_sources:
        if weight <= 0:
            continue
        cmap = resolve_cmap_decoder(cmap_name)
        if cmap is None:
            continue
        for cid, text in preferred_unicode_by_cid(cmap, codec).items():
            if cid in primary_cids:
                continue
            candidates[cid][text] += weight
    standard_cids = set(candidates)
    for cmap_name, codec, weight in (*sources, *opposite_sources):
        if weight > 0:
            continue
        cmap = resolve_cmap_decoder(cmap_name)
        if cmap is None:
            continue
        for cid, text in preferred_unicode_by_cid(cmap, codec).items():
            if cid in standard_cids:
                continue
            candidates[cid][text] += 1
    cid_to_unicode = {
        cid: max(
            counts,
            key=lambda text: (counts[text], unicode_candidate_preference(text), -ord(text)),
        )
        for cid, counts in candidates.items()
    }
    cid_to_unicode.update(CID_COLLECTION_UNICODE_OVERRIDES.get((registry, ordering), {}))
    if not cid_to_unicode:
        return None
    return MappingProxyType(cid_to_unicode)


__all__ = (
    "cmap_resource_index",
    "cmap_resource_root",
    "has_cmap_resource",
    "normalized_cmap_name",
    "predefined_cmap_unicode",
    "resolve_cid_unicode_map",
    "resolve_cmap_decoder",
    "resolve_cmap_resource",
)
