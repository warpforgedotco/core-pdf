# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from core_pdf.impl.third_party.cid.cmap import (
    CIDRange,
    cmap_tokens,
    cmap_usecmap_name,
    decode_cmap_hex_token,
    iter_codespace_range,
    validate_codespace_range,
)
from core_pdf.impl.third_party.cid.resource_loader import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    normalized_cmap_name,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
)


@dataclass(frozen=True, slots=True)
class CompactCMap:
    mappings_by_cid: dict[int, tuple[bytes, ...]]
    mapped_codes: frozenset[bytes]
    ranges: tuple[CIDRange, ...]
    effective_codes_by_cid: dict[int, tuple[bytes, ...]]

    def codes_for_cid(self, cid: int) -> tuple[bytes, ...]:
        return self.effective_codes_by_cid.get(cid, ())


def code_for_cid(cid_range: CIDRange, cid: int) -> bytes | None:
    offset = cid - cid_range.first_cid
    if offset < 0:
        return None
    widths = tuple(end - start + 1 for start, end in zip(cid_range.start, cid_range.end))
    size = 1
    for width in widths:
        size *= width
    if offset >= size:
        return None
    code = bytearray(cid_range.start)
    for index in range(len(code) - 1, -1, -1):
        offset, digit = divmod(offset, widths[index])
        code[index] += digit
    return bytes(code)


def cid_range_size(cid_range: CIDRange) -> int:
    size = 1
    for start, end in zip(cid_range.start, cid_range.end):
        size *= end - start + 1
    return size


def remove_codes_covered_by_ranges(
    mappings: dict[bytes, int], ranges: list[CIDRange]
) -> dict[bytes, int]:
    expansion_size = sum(cid_range_size(item) for item in ranges)
    if expansion_size <= 1_000_000:
        covered = {code for item in ranges for code in iter_codespace_range(item.start, item.end)}
        return {code: cid for code, cid in mappings.items() if code not in covered}
    return {
        code: cid
        for code, cid in mappings.items()
        if not any(item.contains(code) for item in ranges)
    }


def _parsed_cid_data(data: bytes) -> tuple[dict[bytes, int], list[CIDRange]]:
    mappings: dict[bytes, int] = {}
    ranges: list[CIDRange] = []
    mode = 0
    items: list[bytes] = []
    for token in cmap_tokens(data, include_words=True):
        if token == b"begincidchar":
            mode = 2
            items.clear()
            continue
        if token == b"begincidrange":
            mode = 3
            items.clear()
            continue
        if token in {b"endcidchar", b"endcidrange"}:
            mode = 0
            items.clear()
            continue
        if mode == 0:
            continue
        items.append(token)
        if len(items) < mode:
            continue
        if mode == 2:
            code_token, cid_token = items
            items.clear()
            if not (code_token.startswith(b"<") and code_token.endswith(b">")):
                continue
            try:
                mappings[decode_cmap_hex_token(code_token)] = int(cid_token)
            except (ValueError, UnicodeDecodeError):
                continue
        else:
            start_token, end_token, cid_token = items
            items.clear()
            if not (
                start_token.startswith(b"<")
                and start_token.endswith(b">")
                and end_token.startswith(b"<")
                and end_token.endswith(b">")
            ):
                continue
            try:
                start = decode_cmap_hex_token(start_token)
                end = decode_cmap_hex_token(end_token)
                validate_codespace_range(start, end)
                first_cid = int(cid_token)
            except (ValueError, UnicodeDecodeError):
                continue
            ranges.append(CIDRange(start, end, first_cid))
    return mappings, ranges


@lru_cache(maxsize=128)
def compact_cmap(name: str) -> CompactCMap | None:
    data = resolve_cmap_resource(normalized_cmap_name(name))
    if data is None:
        return None
    parent_name = cmap_usecmap_name(data)
    parent = compact_cmap(parent_name) if parent_name is not None else None
    mappings: dict[bytes, int] = {}
    ranges: list[CIDRange] = []
    if parent is not None:
        for cid, parent_codes in parent.mappings_by_cid.items():
            mappings.update((code, cid) for code in parent_codes)
        ranges.extend(parent.ranges)
    child_mappings, child_ranges = _parsed_cid_data(data)
    mappings.update(child_mappings)
    if child_ranges and mappings:
        mappings = remove_codes_covered_by_ranges(mappings, child_ranges)
    ranges.extend(child_ranges)
    mappings_by_cid: defaultdict[int, list[bytes]] = defaultdict(list)
    for code, cid in mappings.items():
        mappings_by_cid[cid].append(code)
    effective_codes_by_cid: defaultdict[int, list[bytes]] = defaultdict(list)
    for cid, codes in mappings_by_cid.items():
        effective_codes_by_cid[cid].extend(codes)
    seen = set(mappings)
    for cid_range in reversed(ranges):
        for offset, code in enumerate(iter_codespace_range(cid_range.start, cid_range.end)):
            if code in seen:
                continue
            seen.add(code)
            effective_codes_by_cid[cid_range.first_cid + offset].append(code)
    return CompactCMap(
        {cid: tuple(codes) for cid, codes in mappings_by_cid.items()},
        frozenset(mappings),
        tuple(ranges),
        {cid: tuple(codes) for cid, codes in effective_codes_by_cid.items()},
    )


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
        result = self._resolve(cid)
        self.cache[cid] = result
        return default if result is None else result

    def _resolve(self, cid: int) -> str | None:
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
