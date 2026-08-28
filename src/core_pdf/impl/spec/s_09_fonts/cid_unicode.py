# SPDX-License-Identifier: AGPL-3.0-only
"""Native CID-to-Unicode recovery and compact CMap helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.cmap_ranges import (
    code_in_ranges,
    iter_codespace_range,
)
from core_pdf.impl.spec.s_09_fonts.cmap_resources import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    normalized_cmap_name,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
)


@dataclass(frozen=True, slots=True)
class CompactCMap:
    effective_codes_by_cid: dict[int, tuple[bytes, ...]]

    def codes_for_cid(self, cid: int) -> tuple[bytes, ...]:
        return self.effective_codes_by_cid.get(cid, ())


def internal_compact_cmap(decoder: CMapDecoder) -> CompactCMap:
    """Invert the effective CID mappings compiled by ``CMapDecoder``."""
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
    return CompactCMap(
        {cid: tuple(codes) for cid, codes in effective_codes_by_cid.items()},
    )


@lru_cache(maxsize=128)
def compact_cmap(name: str) -> CompactCMap | None:
    normalized_name = normalized_cmap_name(name)
    if resolve_cmap_resource(normalized_name) is None:
        return None
    decoder = resolve_cmap_decoder(normalized_name)
    return internal_compact_cmap(decoder) if decoder is not None else None


def preferred_unicode_for_cid(cmap_name: str, codec: str, cid: int) -> str | None:
    cmap = compact_cmap(cmap_name)
    if cmap is None:
        return None
    candidates = {
        text
        for code in cmap.codes_for_cid(cid)
        if (text := unicode_scalar_from_cmap_code(code, codec)) is not None
    }
    if not candidates:
        return None
    return max(candidates, key=lambda text: (unicode_candidate_preference(text), -ord(text)))


class CIDUnicodeMap:
    __slots__ = ("cache", "ordering", "registry", "vertical")

    def __init__(self, registry: str, ordering: str, vertical: bool) -> None:
        self.registry = registry
        self.ordering = ordering
        self.vertical = vertical
        self.cache: dict[int, str | None] = {}

    def get(self, cid: int, default: str | None = None) -> str | None:
        if cid in self.cache:
            result = self.cache[cid]
            return default if result is None else result
        result = self.internal_resolve(cid)
        self.cache[cid] = result
        return default if result is None else result

    def internal_resolve(self, cid: int) -> str | None:
        override = CID_COLLECTION_UNICODE_OVERRIDES.get((self.registry, self.ordering), {}).get(cid)
        if override is not None:
            return override
        collection = CID_COLLECTION_UNICODE_SOURCES.get((self.registry, self.ordering))
        if collection is None:
            return None
        sources = collection[self.vertical]
        opposite_sources = collection[not self.vertical]
        candidates: Counter[str] = Counter()
        for cmap_name, codec, weight in sources:
            if weight <= 0:
                continue
            text = preferred_unicode_for_cid(cmap_name, codec, cid)
            if text is not None:
                candidates[text] += weight
        if not candidates:
            for cmap_name, codec, weight in opposite_sources:
                if weight <= 0:
                    continue
                text = preferred_unicode_for_cid(cmap_name, codec, cid)
                if text is not None:
                    candidates[text] += weight
        if not candidates:
            for cmap_name, codec, weight in (*sources, *opposite_sources):
                if weight > 0:
                    continue
                text = preferred_unicode_for_cid(cmap_name, codec, cid)
                if text is not None:
                    candidates[text] += 1
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda text: (candidates[text], unicode_candidate_preference(text), -ord(text)),
        )


@lru_cache(maxsize=32)
def resolve_cid_unicode_map(
    registry: str,
    ordering: str,
    *,
    vertical: bool = False,
) -> CIDUnicodeMap | None:
    if (registry, ordering) not in CID_COLLECTION_UNICODE_SOURCES:
        return None
    return CIDUnicodeMap(registry, ordering, vertical)


__all__ = ("CIDUnicodeMap", "resolve_cid_unicode_map")
