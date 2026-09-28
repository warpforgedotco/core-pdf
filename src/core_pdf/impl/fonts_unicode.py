# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from functools import cache
from typing import Any

from core_adobe_fonts.cmap.ranges import (
    code_in_ranges,
    iter_codespace_range,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts_cff_repair import CFFUnicodeRepairIndex, is_repairable_to_unicode_label
from core_pdf.impl.fonts_cmap import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    CMapDecoder,
    CMapUnicodeSource,
    ToUnicodeCMap,
    predefined_cmap_unicode,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
    unicode_scalar_or_replacement,
)
from core_pdf.impl.fonts_encoding import FontEncoding
from core_pdf.impl.fonts_program import recover_descriptor, recover_font_file
from core_pdf.impl.fonts_program_base import GlyphProgram
from core_pdf.impl.fonts_program_cff import CFFFont
from core_pdf.impl.fonts_widths import (
    recover_descendant,
)
from core_pdf.impl.glyphs import UnicodeSource
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.types import (
    GeneratedRecord,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream


def single_code_mapping(
    to_unicode: ToUnicodeCMap, cmap: CMapDecoder | None, limit: int | None = None
) -> dict[bytes, tuple[int, str]]:
    mapping: dict[bytes, tuple[int, str]] = {}
    # An identity CMap decodes a code of its own width to its value, and a one-byte
    # code in a two-byte identity to CID 0, so it need not decode each code.
    identity = cmap.identity_width() if cmap is not None else None
    for code_bytes, value in to_unicode.mappings.items():
        width = len(code_bytes)
        if width not in {1, 2}:
            continue
        cid = int.from_bytes(code_bytes, "big")
        if identity is not None:
            if identity == 2 and width == 1:
                cid = 0
        elif cmap is not None:
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
    font_file = recover_font_file(recover_descriptor(descendant.get("FontDescriptor")), "FontFile3")
    if font_file is None or font_program.source_size > 750_000:
        return None
    mapping = single_code_mapping(to_unicode, cmap)
    if not any(is_repairable_to_unicode_label(value) for _, value in mapping.values()):
        return None
    mapping_items = tuple(sorted((code, cid, value) for code, (cid, value) in mapping.items()))
    return CFFUnicodeRepairIndex(font_program, mapping_items)


UNRESOLVED_UNICODE_SOURCES = frozenset(
    {UnicodeSource.IDENTITY, UnicodeSource.REPLACEMENT, UnicodeSource.FALLBACK_NUL}
)


class UnicodeChoice(GeneratedRecord):
    text: str
    source: UnicodeSource
    alternates: tuple[str, ...] = ()


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


def parse_to_unicode(font: dict[str, Any]) -> ToUnicodeCMap | None:
    to_unicode_obj = font.get("ToUnicode")
    if not isinstance(to_unicode_obj, PdfStream):
        return None
    try:
        return ToUnicodeCMap(
            to_unicode_obj.data,
            usecmap_resolver=resolve_cmap_resource,
        )
    except PdfParseError, ValueError:
        return None


def drop_code_entries(cache: dict[Any, Any], codes: Collection[bytes]) -> None:
    for key in [key for key in cache if key[0] in codes]:
        del cache[key]


NO_CHANGED_CODES: frozenset[bytes] = frozenset()


class UnicodeResolver:
    __slots__ = (
        "to_unicode",
        "encoding",
        "program",
        "is_cid_font",
        "is_vertical",
        "ligature_overrides",
        "cff_unicode_repair_index",
        "cff_unicode_repairs",
        "unicode_choice_cache",
    )

    to_unicode: ToUnicodeCMap | None
    encoding: FontEncoding
    program: GlyphProgram
    is_cid_font: bool
    is_vertical: bool
    ligature_overrides: dict[int, str]
    cff_unicode_repair_index: CFFUnicodeRepairIndex | None
    cff_unicode_repairs: dict[bytes, str]
    unicode_choice_cache: dict[tuple[bytes, int, int | None], UnicodeChoice]

    def __init__(
        self,
        to_unicode: ToUnicodeCMap | None,
        encoding: FontEncoding,
        program: GlyphProgram,
        *,
        is_cid_font: bool,
        is_vertical: bool,
        ligature_overrides: dict[int, str],
        cff_unicode_repair_index: CFFUnicodeRepairIndex | None,
    ) -> None:
        self.to_unicode = to_unicode
        self.encoding = encoding
        self.program = program
        self.is_cid_font = is_cid_font
        self.is_vertical = is_vertical
        self.ligature_overrides = ligature_overrides
        self.cff_unicode_repair_index = cff_unicode_repair_index
        self.cff_unicode_repairs = {}
        self.unicode_choice_cache = {}

    def observe_codes(self, entries: Iterable[tuple[bytes, int]]) -> Collection[bytes]:
        repair_index = self.cff_unicode_repair_index
        if repair_index is None:
            return NO_CHANGED_CODES
        to_unicode = self.to_unicode
        mappings = to_unicode.mappings if to_unicode is not None else {}
        repairs = repair_index.repairs_for_codes(
            code_bytes
            for code_bytes, ignored_cid in entries
            if (mapped := mappings.get(code_bytes)) is not None
            and has_invalid_unicode_mapping(mapped)
        )
        if not repairs:
            return NO_CHANGED_CODES
        current = self.cff_unicode_repairs
        changed = {code for code, text in repairs.items() if current.get(code) != text}
        if changed:
            drop_code_entries(self.unicode_choice_cache, changed)
            current.update(repairs)
        return changed

    def unicode_choice_for_code(
        self, code_bytes: bytes, fallback_code: int, gid: int | None = None
    ) -> UnicodeChoice:
        key = (code_bytes, fallback_code, gid)
        cache = self.unicode_choice_cache
        choice = cache.get(key)
        if choice is None:
            choice = cache[key] = self.resolve_unicode_choice_for_code(
                code_bytes, fallback_code, gid
            )
        return choice

    def resolve_unicode_choice_for_code(
        self, code_bytes: bytes, fallback_code: int, gid: int | None
    ) -> UnicodeChoice:
        alternates: list[str] = []
        to_unicode_text = None
        if self.to_unicode is not None:
            to_unicode_text = self.to_unicode.mappings.get(code_bytes)
            if to_unicode_text is not None:
                alternates.append(to_unicode_text)

        def choose(text: str, source: UnicodeSource) -> UnicodeChoice:
            return UnicodeChoice(text, source, dedupe_alternates(alternates, text))

        if to_unicode_text is not None and not has_invalid_unicode_mapping(to_unicode_text):
            visual_punctuation = self.visual_punctuation_for_code(
                to_unicode_text,
                fallback_code=fallback_code,
            )
            if visual_punctuation is not None:
                return choose(visual_punctuation, UnicodeSource.TRUETYPE_GLYPH_SHAPE)
            return choose(to_unicode_text, UnicodeSource.TO_UNICODE)

        replacement = self.cff_unicode_repairs.get(code_bytes)
        if replacement is not None:
            return choose(replacement, UnicodeSource.CFF_GLYPH_REPAIR)

        if gid is not None:
            tt_text = self.program.unicode_for_gid(gid)
            if tt_text == to_unicode_text == "\ufffd":
                return UnicodeChoice(to_unicode_text, UnicodeSource.TO_UNICODE)
            if tt_text and not has_untrusted_unicode_semantics(tt_text):
                return choose(tt_text, UnicodeSource.TRUETYPE_CMAP)

        encoding = self.encoding
        if fallback_code != 0:
            predefined_text = predefined_cmap_unicode(encoding.base_encoding, code_bytes)
            if predefined_text is not None:
                return choose(predefined_text, UnicodeSource.PREDEFINED_CMAP)

        if (
            not self.is_cid_font
            and len(code_bytes) == 1
            and code_bytes[0] not in encoding.differences
        ):
            encoding_table = encoding.encoding_decode_table
            encoding_text = encoding_table[code_bytes[0]]
            if encoding_text and not has_invalid_unicode_mapping(encoding_text):
                return choose(encoding_text, UnicodeSource.ENCODING)

        registry = encoding.cid_registry
        ordering = encoding.cid_ordering
        cid_unicode_map = (
            resolve_cid_unicode_map(registry, ordering, vertical=self.is_vertical)
            if registry is not None and ordering is not None
            else None
        )
        if cid_unicode_map is not None:
            cid_text = cid_unicode_map.get(fallback_code)
            if cid_text is not None:
                return choose(cid_text, UnicodeSource.CID_COLLECTION)

        if to_unicode_text is not None and "\ufffd" in to_unicode_text:
            return UnicodeChoice(to_unicode_text, UnicodeSource.TO_UNICODE)

        if fallback_code == 0:
            return choose("\u0000", UnicodeSource.FALLBACK_NUL)
        text = unicode_scalar_or_replacement(fallback_code)
        source = UnicodeSource.IDENTITY if text != "\ufffd" else UnicodeSource.REPLACEMENT
        return choose(text, source)

    def visual_punctuation_for_code(self, text: str, *, fallback_code: int) -> str | None:
        if len(text) != 1 or not unicodedata.category(text).startswith("M"):
            return None
        bbox = self.program.code_bbox(fallback_code)
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        if width < 450.0 or height <= 0.0 or width / height < 2.5:
            return None
        return "–"

    def apply_simple_unicode_overrides(
        self, choice: UnicodeChoice, code_bytes: bytes
    ) -> UnicodeChoice:
        text = choice.text
        encoding = self.encoding
        if encoding.glyph_decode_table is not None:
            glyph_decode_table = encoding.glyph_decode_table
            if len(code_bytes) == 1:
                mapped = glyph_decode_table[code_bytes[0]]
                if not text:
                    if encoding.glyph_decode_table_authoritative or mapped:
                        text = mapped
                elif (
                    len(text) == 1
                    and mapped
                    and text != mapped
                    and (
                        choice.source in UNRESOLVED_UNICODE_SOURCES
                        or should_prefer_glyph_name_mapping(
                            text,
                            mapped,
                            authoritative=encoding.glyph_decode_table_authoritative,
                        )
                    )
                ):
                    text = mapped
            else:
                text = replace_unicode_from_glyph_names(
                    text,
                    code_bytes,
                    glyph_decode_table,
                    authoritative=encoding.glyph_decode_table_authoritative,
                    fallback_mapping=choice.source in UNRESOLVED_UNICODE_SOURCES,
                )
        if self.ligature_overrides:
            lo = self.ligature_overrides
            text = "".join(lo.get(ord(ch), ch) for ch in text)
        if text == choice.text:
            return choice
        return UnicodeChoice(
            text,
            UnicodeSource.GLYPH_NAME,
            dedupe_alternates((choice.text, *choice.alternates), text),
        )

    def apply_ligature_overrides(self, choice: UnicodeChoice) -> UnicodeChoice:
        if not self.ligature_overrides:
            return choice
        lo = self.ligature_overrides
        text = "".join(lo.get(ord(ch), ch) for ch in choice.text)
        if text == choice.text:
            return choice
        return UnicodeChoice(
            text,
            UnicodeSource.LIGATURE_OVERRIDE,
            dedupe_alternates((choice.text, *choice.alternates), text),
        )


__all__ = ("CIDUnicodeMap", "UnicodeResolver", "resolve_cid_unicode_map")
