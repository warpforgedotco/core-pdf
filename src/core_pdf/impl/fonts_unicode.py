# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import cache
from typing import Any, ClassVar

from core_adobe_fonts.cmap.ranges import (
    code_in_ranges,
    iter_codespace_range,
)
from core_pdf.impl.fonts_cff_repair import CFFUnicodeRepairIndex, is_repairable_to_unicode_label
from core_pdf.impl.fonts_cmap_resources import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    CMapUnicodeSource,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
)
from core_pdf.impl.fonts_cmap_tokenizer import CMapDecoder
from core_pdf.impl.fonts_cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.fonts_helpers import slot_setter
from core_pdf.impl.fonts_program_base import GlyphProgram
from core_pdf.impl.fonts_program_cff import CFFFont
from core_pdf.impl.fonts_widths import (
    recover_descendant,
)
from core_pdf.impl.glyphs import UnicodeSource
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import (
    Record,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream


def single_code_mapping(
    to_unicode: ToUnicodeCMap, cmap: CMapDecoder | None, limit: int | None = None
) -> dict[bytes, tuple[int, str]]:
    mapping: dict[bytes, tuple[int, str]] = {}
    for code_bytes, value in to_unicode.mappings.items():
        if len(code_bytes) not in {1, 2}:
            continue
        cid = int.from_bytes(code_bytes, "big")
        if cmap is not None:
            decoded = cmap.decode_entries(code_bytes)
            if len(decoded) == 1 and decoded[0][0] == code_bytes:
                cid = decoded[0][1]
        if limit is not None and cid >= limit:
            continue
        mapping.setdefault(code_bytes, (cid, value))
    return mapping


def build_cff_unicode_repair_index(
    font: dict[str, Any],
    font_program: GlyphProgram,
    to_unicode: ToUnicodeCMap | None,
    cmap: CMapDecoder | None,
) -> CFFUnicodeRepairIndex | None:
    if to_unicode is None or not isinstance(font_program, CFFFont):
        return None
    descendant = recover_descendant(font)
    if descendant is None:
        return None
    if recover_pdf_name(descendant.get("Subtype")) != "CIDFontType0":
        return None
    descriptor = descendant.get("FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = descriptor.get("FontFile3")
    if not isinstance(font_file, PdfStream) or len(font_file.data) > 750_000:
        return None
    mapping = single_code_mapping(to_unicode, cmap)
    if not any(is_repairable_to_unicode_label(value) for _, value in mapping.values()):
        return None
    mapping_items = tuple(sorted((code, cid, value) for code, (cid, value) in mapping.items()))
    return CFFUnicodeRepairIndex(font_program, mapping_items)


UNRESOLVED_UNICODE_SOURCES = frozenset(
    {UnicodeSource.IDENTITY, UnicodeSource.REPLACEMENT, UnicodeSource.FALLBACK_NUL}
)


class UnicodeChoice(Record):
    __slots__ = ("text", "source", "alternates")

    text: str
    source: UnicodeSource
    alternates: tuple[str, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("text", "source", "alternates")
    __match_args__ = ("text", "source", "alternates")

    def __init__(self, text: str, source: UnicodeSource, alternates: tuple[str, ...] = ()) -> None:
        _unicodechoice_set_text(self, text)
        _unicodechoice_set_source(self, source)
        _unicodechoice_set_alternates(self, alternates)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.source == other.source
            and self.alternates == other.alternates
        )

    def __hash__(self) -> int:
        return hash((self.text, self.source, self.alternates))


_unicodechoice_set_text = slot_setter(UnicodeChoice, "text")


_unicodechoice_set_source = slot_setter(UnicodeChoice, "source")


_unicodechoice_set_alternates = slot_setter(UnicodeChoice, "alternates")


def dedupe_alternates(values: Iterable[str], selected: str) -> tuple[str, ...]:
    seen = {selected}
    alternates: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        alternates.append(value)
    return tuple(alternates)


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


class CIDUnicodeMap:
    __slots__ = ("_cache", "ordering", "registry", "vertical")

    def __init__(self, registry: str, ordering: str, vertical: bool) -> None:
        self.registry = registry
        self.ordering = ordering
        self.vertical = vertical
        self._cache: dict[int, str | None] = {}

    def get(self, cid: int, default: str | None = None) -> str | None:
        result = self.resolve(cid)
        return default if result is None else result

    def resolve(self, cid: int) -> str | None:
        cache = self._cache
        try:
            return cache[cid]
        except KeyError:
            text = cache[cid] = self.vote(cid)
            return text

    def vote(self, cid: int) -> str | None:
        override = CID_COLLECTION_UNICODE_OVERRIDES.get((self.registry, self.ordering), {}).get(cid)
        if override is not None:
            return override
        collection = CID_COLLECTION_UNICODE_SOURCES.get((self.registry, self.ordering))
        if collection is None:
            return None
        sources = collection[self.vertical]
        opposite_sources = collection[not self.vertical]
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


def has_untrusted_unicode_semantics(text: str) -> bool:
    if not text:
        return True
    for ch in text:
        if ch == "\ufffd":
            return True
        if unicodedata.category(ch).startswith("C"):
            return True
    return False


def has_invalid_unicode_mapping(text: str) -> bool:
    return "\ufffd" in text or "\x00" in text


def should_prefer_glyph_name_mapping(
    current: str,
    mapped: str,
    *,
    authoritative: bool,
) -> bool:
    if not mapped or current == mapped:
        return False
    if authoritative:
        return True
    if has_untrusted_unicode_semantics(current):
        return True
    return unicodedata.normalize("NFKC", mapped) == current


def replace_unicode_from_glyph_names(
    text: str,
    data: bytes,
    glyph_decode_table: tuple[str, ...],
    *,
    authoritative: bool,
    fallback_mapping: bool = False,
) -> str:
    if not text:
        if authoritative or fallback_mapping or all(glyph_decode_table[code] for code in data):
            return "".join(glyph_decode_table[code] for code in data)
        return text
    if len(text) != len(data):
        return text
    if len(data) == 1:
        mapped = glyph_decode_table[data[0]]
        if not mapped:
            return text
        if not (
            fallback_mapping
            or should_prefer_glyph_name_mapping(
                text,
                mapped,
                authoritative=authoritative,
            )
        ):
            return text
        return mapped
    out: list[str] | None = None
    for index, code in enumerate(data):
        mapped = glyph_decode_table[code]
        if not mapped:
            continue
        current = text[index]
        if current == mapped:
            continue
        if not (
            fallback_mapping
            or should_prefer_glyph_name_mapping(
                current,
                mapped,
                authoritative=authoritative,
            )
        ):
            continue
        if out is None:
            out = list(text)
        out[index] = mapped
    return text if out is None else "".join(out)
