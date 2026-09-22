# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections import Counter, defaultdict
from functools import cache
from typing import Any, ClassVar, Self

from core_adobe_fonts.cmap.ranges import (
    code_in_ranges,
    iter_codespace_range,
)
from core_pdf.impl.fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.fonts.cmap_resources import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
)
from core_pdf.impl.records import internal_Record

internal_frozen_setattr = object.__setattr__


class CompactCMap(internal_Record):
    __slots__ = ("effective_codes_by_cid",)

    effective_codes_by_cid: dict[int, tuple[bytes, ...]]

    __fields__: ClassVar[tuple[str, ...]] = ("effective_codes_by_cid",)
    __match_args__ = ("effective_codes_by_cid",)

    def __init__(self, effective_codes_by_cid: dict[int, tuple[bytes, ...]]) -> None:
        internal_frozen_setattr(self, "effective_codes_by_cid", effective_codes_by_cid)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(effective_codes_by_cid={self.effective_codes_by_cid!r})"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.effective_codes_by_cid == other.effective_codes_by_cid

    def __hash__(self) -> int:
        return hash((self.effective_codes_by_cid,))

    def __replace__(self, /, **changes: Any) -> Self:
        effective_codes_by_cid = changes.pop("effective_codes_by_cid", self.effective_codes_by_cid)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(effective_codes_by_cid)

    def codes_for_cid(self, cid: int) -> tuple[bytes, ...]:
        return self.effective_codes_by_cid.get(cid, ())


def internal_compact_cmap(decoder: CMapDecoder) -> CompactCMap:
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


@cache
def compact_cmap(name: str) -> CompactCMap | None:
    if resolve_cmap_resource(name) is None:
        return None
    decoder = resolve_cmap_decoder(name)
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
    __slots__ = ("internal_cache", "ordering", "registry", "vertical")

    def __init__(self, registry: str, ordering: str, vertical: bool) -> None:
        self.registry = registry
        self.ordering = ordering
        self.vertical = vertical
        self.internal_cache: dict[int, str | None] = {}

    def get(self, cid: int, default: str | None = None) -> str | None:
        result = self.internal_resolve(cid)
        return default if result is None else result

    def internal_resolve(self, cid: int) -> str | None:
        cache = self.internal_cache
        try:
            return cache[cid]
        except KeyError:
            text = cache[cid] = self.internal_vote(cid)
            return text

    def internal_vote(self, cid: int) -> str | None:
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
        ranked = {
            text: (weight, unicode_candidate_preference(text), -ord(text))
            for text, weight in candidates.items()
        }
        return max(ranked, key=ranked.__getitem__)


@cache
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
