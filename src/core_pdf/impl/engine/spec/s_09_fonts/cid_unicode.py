# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from functools import lru_cache

from core_pdf.impl.third_party.cid import resolve_cmap_decoder
from core_pdf.impl.third_party.cid.cmap import CMapDecoder, iter_codespace_range

UNICODE_CMAP_NAMES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "CNS1": (
        ("UniCNS-UTF16-H", "UniCNS-UCS2-H"),
        ("UniCNS-UTF16-V", "UniCNS-UCS2-V", "UniCNS-UTF16-H", "UniCNS-UCS2-H"),
    ),
    "GB1": (
        ("UniGB-UTF16-H", "UniGB-UCS2-H"),
        ("UniGB-UTF16-V", "UniGB-UCS2-V", "UniGB-UTF16-H", "UniGB-UCS2-H"),
    ),
    "Japan1": (
        ("UniJIS-UTF16-H", "UniJIS-UCS2-H"),
        ("UniJIS-UTF16-V", "UniJIS-UCS2-V", "UniJIS-UTF16-H", "UniJIS-UCS2-H"),
    ),
    "Korea1": (
        ("UniKS-UTF16-H", "UniKS-UCS2-H"),
        ("UniKS-UTF16-V", "UniKS-UCS2-V", "UniKS-UTF16-H", "UniKS-UCS2-H"),
    ),
    "KR": (
        ("UniAKR-UTF16-H",),
        ("UniAKR-UTF16-H",),
    ),
}


def decode_utf16_code(code: bytes) -> str | None:
    if not code or len(code) % 2 != 0:
        return None
    try:
        text = code.decode("utf-16-be")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return None
    return text


def unicode_text_score(text: str) -> tuple[int, int, int]:
    if not text:
        return (0, 0, 0)
    codepoint = ord(text[0])
    if codepoint == 0 or 0xD800 <= codepoint <= 0xDFFF or codepoint > 0x10FFFF:
        return (0, 0, 0)
    if codepoint < 32 or 0x7F <= codepoint <= 0x9F:
        return (1, -len(text), 0)
    if 0xE000 <= codepoint <= 0xF8FF:
        return (2, -len(text), 0)
    if (
        0x2E80 <= codepoint <= 0x2EFF
        or 0x2F00 <= codepoint <= 0x2FDF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0xFE10 <= codepoint <= 0xFE4F
    ):
        return (3, -len(text), -codepoint)
    if (
        0x3400 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2EBEF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return (5, -len(text), -codepoint)
    return (4, -len(text), -codepoint)


def add_cid_mapping(mapping: dict[int, str], cid: int, text: str | None) -> None:
    if cid <= 0 or text is None:
        return
    existing = mapping.get(cid)
    if existing is None or unicode_text_score(text) > unicode_text_score(existing):
        mapping[cid] = text


def invert_unicode_cmap(cmap: CMapDecoder) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for code, cid in cmap.cid_mappings.items():
        add_cid_mapping(mapping, cid, decode_utf16_code(code))
    for cid_range in cmap.cid_ranges:
        for code in iter_codespace_range(cid_range.start, cid_range.end):
            add_cid_mapping(mapping, cid_range.cid_for(code), decode_utf16_code(code))
    return mapping


@lru_cache(maxsize=32)
def predefined_cid_to_unicode(ordering: str, is_vertical: bool) -> dict[int, str]:
    cmap_name_groups = UNICODE_CMAP_NAMES.get(ordering)
    if cmap_name_groups is None:
        return {}
    cmap_names = cmap_name_groups[1] if is_vertical else cmap_name_groups[0]
    mapping: dict[int, str] = {}
    for cmap_name in cmap_names:
        cmap = resolve_cmap_decoder(cmap_name)
        if cmap is None:
            continue
        for cid, text in invert_unicode_cmap(cmap).items():
            existing = mapping.get(cid)
            if existing is None or unicode_text_score(text) > unicode_text_score(existing):
                mapping[cid] = text
    return mapping
