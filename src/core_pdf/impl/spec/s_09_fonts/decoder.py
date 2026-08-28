# SPDX-License-Identifier: AGPL-3.0-only
"""Decode font programs to glyphs, widths, and Unicode text."""

from __future__ import annotations

import re
import typing
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.primitives import PdfString
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import normalize_pdf_name
from core_pdf.impl.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.spec.s_09_fonts.cff import (
    build_cff_unicode_repair_index,
    cff_font_for_pdf_font,
)
from core_pdf.impl.spec.s_09_fonts.cid_unicode import (
    CIDUnicodeMap,
    resolve_cid_unicode_map,
)
from core_pdf.impl.spec.s_09_fonts.cmap_decoder import CMapDecoder
from core_pdf.impl.spec.s_09_fonts.cmap_encoding import BYTE_CACHE
from core_pdf.impl.spec.s_09_fonts.cmap_ranges import (
    code_in_ranges,
    unicode_scalar_or_replacement,
)
from core_pdf.impl.spec.s_09_fonts.cmap_resources import (
    predefined_cmap_unicode,
    resolve_cmap_decoder,
    resolve_cmap_resource,
)
from core_pdf.impl.spec.s_09_fonts.cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.spec.s_09_fonts.cmap_widths import FontWidthMap
from core_pdf.impl.spec.s_09_fonts.font_names import resolve_base_font_name
from core_pdf.impl.spec.s_09_fonts.font_program import (
    CFFFont,
    CFFUnicodeRepairIndex,
)
from core_pdf.impl.spec.s_09_fonts.font_program_opentype import (
    OpenTypeFontProgram,
    opentype_font_for_pdf_font,
)
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
    TrueTypeFontProgram,
)
from core_pdf.impl.spec.s_09_fonts.font_program_type1 import (
    Type1FontProgram,
    type1_font_for_pdf_font,
)
from core_pdf.impl.spec.s_09_fonts.glyph_decode import (
    build_glyph_decode_table,
    has_invalid_unicode_mapping,
    has_untrusted_unicode_semantics,
    replace_unicode_from_glyph_names,
    should_prefer_glyph_name_mapping,
)
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.spec.s_09_fonts.helpers import (
    base_encoding_glyph_names,
    cached_decode_table,
    parse_differences,
    unicode_for_glyph_name,
)
from core_pdf.impl.spec.s_09_fonts.metrics import (
    adjust_type3_widths,
    parse_font_metrics,
    standard_14_widths,
)
from core_pdf.impl.spec.s_09_fonts.truetype import tt_font_for_pdf_font
from core_pdf.impl.spec.s_09_fonts.widths import (
    get_descendant,
    parse_font_widths,
)
from core_pdf.impl.types import Rectangle

if typing.TYPE_CHECKING:
    from typing import Any

    from core_pdf.impl.spec.s_09_fonts.fallback import RasterFontProviderLike


FontProgram = CFFFont | TrueTypeFontProgram | Type1FontProgram | OpenTypeFontProgram


LEGITIMATE_MULTI_CHAR_GLYPHS = frozenset({"ff", "fi", "fl", "ffi", "ffl", "st"})
TYPE1_ENCODING_ENTRY_RE = re.compile(rb"\bdup\s+(\d{1,3})\s+/([A-Za-z0-9_.]+)\s+put\b")


def parse_type1_font_program_encoding(font_program: bytes | memoryview) -> dict[int, str]:
    data = bytes(font_program)
    eexec_pos = data.find(b"currentfile eexec")
    if eexec_pos >= 0:
        data = data[:eexec_pos]

    differences: dict[int, str] = {}
    for match in TYPE1_ENCODING_ENTRY_RE.finditer(data):
        code = int(match.group(1))
        if 0 <= code <= 255:
            differences[code] = match.group(2).decode("latin-1")
    return differences


def internal_font_program_for_pdf_font(font: dict[str, Any]) -> FontProgram | None:
    """Select one embedded outline implementation in format-preference order."""
    for resolver in (
        cff_font_for_pdf_font,
        tt_font_for_pdf_font,
        type1_font_for_pdf_font,
        opentype_font_for_pdf_font,
    ):
        program = resolver(font)
        if program is not None:
            return program
    return None


# Distinct CID codes memoized per decoder. A simple font's cache is bounded by
# its 256 single-byte codes; a CID font's code space is not, so this bounds the
# dictionary while still covering the few thousand distinct codes a dense CJK
# page can carry.
internal_CID_GLYPH_CACHE_LIMIT = 4096


@dataclass(frozen=True, slots=True)
class DecodedGlyph:
    code_bytes: bytes
    char_code: int
    cid: int
    gid: int | None
    unicode: str
    unicode_source: str
    alternates: tuple[str, ...]
    width_code: int
    bitmap_code: int
    split_unicode: bool = False


@dataclass(frozen=True, slots=True)
class UnicodeChoice:
    text: str
    source: str
    alternates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Type3CharProcProgram:
    """A resolved Type3 CharProc and its optional safe replay program."""

    stream: PdfStream | None
    operations: tuple[tuple[Any, tuple[Any, ...]], ...] | None


def split_code_bytes(data: bytes, cmap: CMapDecoder | ToUnicodeCMap | None) -> list[bytes]:
    if not data:
        return []
    if cmap is None:
        byte_cache = BYTE_CACHE
        return [byte_cache[byte] for byte in data]
    lengths = getattr(cmap, "decode_lengths", None) or (1,)
    ranges = getattr(cmap, "code_space_ranges", None) or ()
    chunks: list[bytes] = []
    pos = 0
    n = len(data)
    while pos < n:
        matched = False
        for length in lengths:
            if pos + length > n:
                continue
            chunk = BYTE_CACHE[data[pos]] if length == 1 else data[pos : pos + length]
            if ranges and not code_in_ranges(chunk, ranges):
                continue
            chunks.append(chunk)
            pos += length
            matched = True
            break
        if not matched:
            chunks.append(BYTE_CACHE[data[pos]])
            pos += 1
    return chunks


@dataclass(init=False, repr=False, eq=False, slots=True, match_args=False)
class FontDecoder:
    font: dict[str, Any]
    ligature_overrides: dict[int, str]
    to_unicode: ToUnicodeCMap | None
    cmap: CMapDecoder | None
    cid_unicode_map: Mapping[int, str] | CIDUnicodeMap | None
    cid_unicode_map_resolved: bool
    cid_registry: str | None
    cid_ordering: str | None
    base_encoding: str | None
    differences: dict[int, str]
    encoding_differences: dict[int, str]
    simple_encoding_glyph_names: tuple[str, ...]
    encoding_decode_table: tuple[str, ...]
    is_cid_font: bool
    is_type3: bool
    byte_decode_table: tuple[str, ...] | None
    widths: FontWidthMap
    default_width: float
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]
    is_vertical: bool
    ascent: float
    descent: float
    decode_cache: dict[bytes, str]
    glyphs_cache: dict[bytes, tuple[DecodedGlyph, ...]]
    fast_widths_cache: tuple[float, ...] | None
    simple_glyph_cache: dict[int, DecodedGlyph]
    cid_glyph_cache: dict[bytes, DecodedGlyph]
    glyph_bbox_cache: dict[int, Rectangle | None]
    glyph_bitmap_cache: dict[tuple[int, int, int], tuple[int, ...]]
    font_name: str | None
    glyph_decode_table: tuple[str, ...] | None
    glyph_decode_table_authoritative: bool
    type3_glyph_names: dict[int, str] | None
    type3_charproc_cache: list[Type3CharProcProgram | None]
    type3_charproc_cache_hits: int
    type3_charproc_cache_misses: int
    type3_charproc_compiled_programs: int
    type3_charproc_compiled_operations: int
    type3_charproc_unsafe_fallbacks: int
    cff_unicode_repair_index: CFFUnicodeRepairIndex | None
    cff_unicode_repairs: dict[bytes, str]
    font_program: FontProgram | None
    raster_font_provider: RasterFontProviderLike | None

    def __init__(
        self,
        font: dict[str, Any],
        ligature_overrides: dict[int, str] | None = None,
        raster_font_provider: RasterFontProviderLike | None = None,
    ) -> None:
        self.font = font
        self.ligature_overrides = ligature_overrides if ligature_overrides is not None else {}
        self.raster_font_provider = raster_font_provider
        self.decode_cache: dict[bytes, str] = {}
        self.glyphs_cache: dict[bytes, tuple[DecodedGlyph, ...]] = {}
        self.simple_glyph_cache = {}
        self.cid_glyph_cache = {}
        self.fast_widths_cache = None
        self.glyph_bbox_cache = {}
        self.glyph_bitmap_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}
        self.type3_glyph_names = None
        self.type3_charproc_cache = [None] * 256
        self.type3_charproc_cache_hits = 0
        self.type3_charproc_cache_misses = 0
        self.type3_charproc_compiled_programs = 0
        self.type3_charproc_compiled_operations = 0
        self.type3_charproc_unsafe_fallbacks = 0
        self.internal_initialize()

    def internal_initialize(self) -> None:
        font = self.font
        subtype = lookup_dict_key(font, "Subtype")
        if subtype is not None:
            subtype = normalize_pdf_name(subtype)

        # The embedded program's built-in encoding participates in simple-font
        # decoding, so select the one canonical backend before normalizing the
        # PDF font dictionary's /Encoding.
        self.font_program = internal_font_program_for_pdf_font(font)

        to_unicode_obj = lookup_dict_key(font, "ToUnicode")
        to_unicode = None
        if isinstance(to_unicode_obj, PdfStream):
            try:
                to_unicode = ToUnicodeCMap(
                    to_unicode_obj.data,
                    usecmap_resolver=resolve_cmap_resource,
                )
            except (PdfParseError, ValueError):
                to_unicode = None

        (
            cmap,
            base_encoding,
            differences,
            builtin_encoding,
            builtin_encoding_authoritative,
        ) = self.parse_encoding(font)
        font_metrics = parse_font_widths(font, subtype)
        widths = font_metrics.widths
        default_width = font_metrics.default_width
        is_vertical = font_metrics.is_vertical
        is_cid_font = subtype == "Type0" and get_descendant(font) is not None

        base_font_name = resolve_base_font_name(font, subtype)
        if (
            base_encoding == "V"
            or (base_encoding and base_encoding.endswith("-V"))
            or (base_font_name and base_font_name.endswith("-V"))
            or (cmap is not None and getattr(cmap, "wmode", 0) == 1)
        ):
            is_vertical = True

        ascent, descent = parse_font_metrics(font, subtype, base_font_name, widths)

        is_type3 = subtype == "Type3"
        if is_type3:
            widths = adjust_type3_widths(font, widths)

        simple_encoding_glyph_names = self.internal_simple_encoding_glyph_names(
            base_encoding,
            builtin_encoding,
            differences,
            authoritative_builtin=builtin_encoding_authoritative,
        )
        if builtin_encoding_authoritative:
            encoding_decode_table = tuple(
                unicode_for_glyph_name(name) or "" for name in simple_encoding_glyph_names
            )
        else:
            key = base_encoding or ("Type3" if is_type3 else "")
            encoding_decode_table = cached_decode_table(key, tuple(sorted(differences.items())))

        byte_decode_table: tuple[str, ...] | None = None
        if to_unicode is None and not is_cid_font:
            byte_decode_table = encoding_decode_table

        if not widths and not is_cid_font and not is_type3:
            # A standard 14 font may legally omit /Widths (9.6.2.2); supply the
            # built-in metrics rather than advancing every glyph by MissingWidth.
            builtin = standard_14_widths(base_font_name, encoding_decode_table)
            if builtin is not None:
                widths = builtin
                # Table 122 defaults MissingWidth to 0, and a code this font
                # does not encode should not advance by a full em.
                if lookup_dict_key(font, "MissingWidth") is None:
                    default_width = 0.0

        self.to_unicode = to_unicode
        self.cmap = cmap
        self.cid_unicode_map = None
        self.cid_unicode_map_resolved = not is_cid_font
        self.cid_registry, self.cid_ordering = self.internal_cid_system_info(font)
        self.base_encoding = base_encoding
        self.differences = differences
        # Compatibility facades inspect the sparse overrides directly. Keep
        # that view alongside the complete glyph-name and Unicode tables used
        # by native decoding and outline selection.
        self.encoding_differences = (
            {**builtin_encoding, **differences} if builtin_encoding else differences
        )
        self.simple_encoding_glyph_names = simple_encoding_glyph_names
        self.encoding_decode_table = encoding_decode_table
        self.is_cid_font = is_cid_font
        self.is_type3 = is_type3
        self.byte_decode_table = byte_decode_table
        self.widths = widths
        self.default_width = default_width
        self.default_vertical_displacement_y = font_metrics.default_vertical_displacement_y
        self.default_vertical_origin_y = font_metrics.default_vertical_origin_y
        self.vertical_metrics = font_metrics.vertical_metrics
        self.is_vertical = is_vertical
        self.ascent = ascent
        self.descent = descent
        self.font_name = base_font_name
        glyph_decode = build_glyph_decode_table(base_font_name, differences)
        if glyph_decode is None:
            self.glyph_decode_table = None
            self.glyph_decode_table_authoritative = False
        else:
            self.glyph_decode_table, self.glyph_decode_table_authoritative = glyph_decode
        self.cff_unicode_repair_index = build_cff_unicode_repair_index(font, to_unicode, cmap)
        self.cff_unicode_repairs = {}

    @staticmethod
    def internal_cid_system_info_string(value: object) -> str | None:
        if isinstance(value, PdfString):
            return value.data.decode("latin-1")
        normalized = normalize_pdf_name(value)
        if normalized is not None:
            return normalized
        if isinstance(value, bytes):
            return value.decode("latin-1")
        return None

    @classmethod
    def internal_cid_system_info(
        cls,
        font: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        descendant = get_descendant(font)
        system_info = lookup_dict_key(descendant, "CIDSystemInfo") if descendant else None
        if not isinstance(system_info, dict):
            system_info = lookup_dict_key(font, "CIDSystemInfo")
        if not isinstance(system_info, dict):
            return None, None
        registry = cls.internal_cid_system_info_string(lookup_dict_key(system_info, "Registry"))
        ordering = cls.internal_cid_system_info_string(lookup_dict_key(system_info, "Ordering"))
        return registry, ordering

    @classmethod
    def internal_cid_unicode_map(
        cls,
        font: dict[str, Any],
        *,
        vertical: bool,
    ) -> Mapping[int, str] | CIDUnicodeMap | None:
        registry, ordering = cls.internal_cid_system_info(font)
        if registry is None or ordering is None:
            return None
        return resolve_cid_unicode_map(registry, ordering, vertical=vertical)

    @property
    def fast_widths(self) -> tuple[float, ...]:
        widths = self.fast_widths_cache
        if widths is None:
            widths = self.widths.fast_256(self.default_width)
            self.fast_widths_cache = widths
        return widths

    def parse_encoding(
        self, font: dict[str, Any]
    ) -> tuple[CMapDecoder | None, str | None, dict[int, str], dict[int, str], bool]:
        cmap = None
        base_encoding = None
        base_encoding_explicit = False
        differences: dict[int, str] = {}
        subtype = normalize_pdf_name(lookup_dict_key(font, "Subtype"))
        encoding_obj = lookup_dict_key(font, "Encoding")
        match encoding_obj:
            case str():
                base_encoding = normalize_pdf_name(encoding_obj)
                base_encoding_explicit = base_encoding is not None
                cmap = self.internal_named_cmap(base_encoding)
            case PdfStream():
                try:
                    cmap = CMapDecoder(
                        encoding_obj.data,
                        usecmap_resolver=resolve_cmap_decoder,
                    )
                except (PdfParseError, ValueError):
                    cmap = None
            case dict():
                base_encoding = normalize_pdf_name(lookup_dict_key(encoding_obj, "BaseEncoding"))
                if base_encoding is None:
                    base_encoding = (
                        "WinAnsiEncoding" if subtype == "TrueType" else "StandardEncoding"
                    )
                else:
                    base_encoding_explicit = True
                differences_obj = lookup_dict_key(encoding_obj, "Differences")
                if differences_obj is not None and not isinstance(differences_obj, (list, tuple)):
                    differences_obj = None
                differences = parse_differences(
                    list(differences_obj)
                    if isinstance(differences_obj, tuple)
                    else differences_obj,
                    normalize_pdf_name,
                )
            case _:
                base_encoding = normalize_pdf_name(encoding_obj)
                base_encoding_explicit = base_encoding is not None
                cmap = self.internal_named_cmap(base_encoding)
        if base_encoding is None and subtype == "Type3":
            base_encoding = "StandardEncoding"
        builtin: dict[int, str] = {}
        builtin_authoritative = False
        if subtype in ("Type1", "MMType1") and not base_encoding_explicit:
            # Table 114: when /BaseEncoding is absent, the implicit base for an
            # embedded font program is the program's own built-in encoding, and
            # /Differences describes changes from that. An explicit base
            # encoding still wins, so this only fills the implicit case.
            builtin, builtin_authoritative = self.internal_builtin_font_encoding(font)
        if base_encoding is None and subtype in ("Type1", "MMType1"):
            # 9.6.6.1: every font program bar Type 3 carries a built-in
            # encoding, which governs when the font dictionary supplies none.
            # Where we cannot read it back out of the program, the Latin text
            # default in Annex D is the right stand-in -- PDFDocEncoding, the
            # previous fallback, encodes text strings such as metadata and
            # bookmark titles and has no business decoding glyphs.
            base_encoding = "StandardEncoding"
        return cmap, base_encoding, differences, builtin, builtin_authoritative

    def internal_builtin_font_encoding(self, font: dict[str, Any]) -> tuple[dict[int, str], bool]:
        """Read the code to glyph-name encoding out of an embedded program."""
        match self.font_program:
            case CFFFont() as program:
                try:
                    return (
                        program.builtin_encoding(),
                        program.builtin_encoding_is_authoritative(),
                    )
                except (PdfParseError, ValueError):
                    return {}, False
            case _:
                pass
        descriptor = lookup_dict_key(font, "FontDescriptor")
        if not isinstance(descriptor, dict):
            return {}, False
        font_file = lookup_dict_key(descriptor, "FontFile")
        if isinstance(font_file, PdfStream):
            try:
                encoding = parse_type1_font_program_encoding(font_file.data)
                return encoding, bool(encoding)
            except (PdfParseError, ValueError):
                return {}, False
        return {}, False

    @staticmethod
    def internal_simple_encoding_glyph_names(
        base_encoding: str | None,
        builtin_encoding: Mapping[int, str],
        differences: Mapping[int, str],
        *,
        authoritative_builtin: bool,
    ) -> tuple[str, ...]:
        """Layer one complete simple-font code-to-glyph-name encoding.

        Custom and Expert CFF encodings are sparse and authoritative: an
        absent code denotes ``.notdef`` rather than falling through to
        StandardEncoding. Explicit PDF /Differences are always the final layer.
        """
        names = (
            [".notdef"] * 256
            if authoritative_builtin
            else list(base_encoding_glyph_names(base_encoding))
        )
        for mapping in (builtin_encoding, differences):
            for code, name in mapping.items():
                if 0 <= code < 256:
                    names[code] = name or ".notdef"
        return tuple(names)

    def internal_named_cmap(self, base_encoding: str | None) -> CMapDecoder | None:
        if base_encoding is None:
            return None
        return resolve_cmap_decoder(base_encoding)

    def decode(self, data: bytes) -> str:
        if not data:
            return ""
        use_cache = len(data) <= 16
        decode_cache = self.decode_cache
        if use_cache:
            cached = decode_cache.get(data)
            if cached is not None:
                return cached
        table = self.byte_decode_table
        if (
            table is not None
            and not self.is_cid_font
            and self.to_unicode is None
            and self.glyph_decode_table is None
            and not self.ligature_overrides
        ):
            result = "".join(table[byte] for byte in data)
        else:
            result = "".join(glyph.unicode for glyph in self.decode_glyphs(data))
        if use_cache and len(decode_cache) < 512:
            decode_cache[data] = result
        return result

    def decode_glyphs(self, data: bytes | bytearray | memoryview) -> tuple[DecodedGlyph, ...]:
        if not data:
            return ()
        use_cache = len(data) <= 16
        cache_key = bytes(data) if use_cache else None
        if cache_key is not None:
            cached = self.glyphs_cache.get(cache_key)
            if cached is not None:
                return cached

        if self.is_cid_font:
            glyphs = self.internal_decode_cid_glyphs(bytes(data))
        else:
            glyphs = self.internal_decode_simple_glyphs(data)
        result = tuple(glyphs)
        if cache_key is not None and len(self.glyphs_cache) < 512:
            self.glyphs_cache[cache_key] = result
        return result

    def internal_unicode_choice_for_code(
        self, code_bytes: bytes, fallback_code: int, gid: int | None = None
    ) -> UnicodeChoice:
        alternates: list[str] = []
        to_unicode_text = None
        if self.to_unicode is not None:
            to_unicode_text = self.to_unicode.mappings.get(code_bytes)
            if to_unicode_text is not None:
                alternates.append(to_unicode_text)

        if to_unicode_text is not None and not has_invalid_unicode_mapping(to_unicode_text):
            visual_punctuation = self.internal_visual_punctuation_for_code(
                to_unicode_text,
                fallback_code=fallback_code,
            )
            if visual_punctuation is not None:
                return UnicodeChoice(
                    visual_punctuation,
                    "truetype_glyph_shape",
                    dedupe_alternates(alternates, visual_punctuation),
                )
            return UnicodeChoice(
                to_unicode_text,
                "to_unicode",
                dedupe_alternates(alternates, to_unicode_text),
            )

        replacement = self.cff_unicode_repairs.get(code_bytes)
        if replacement is not None:
            return UnicodeChoice(
                replacement,
                "cff_glyph_repair",
                dedupe_alternates(alternates, replacement),
            )

        if gid is not None:
            tt_text = self.internal_true_type_unicode_for_gid(gid)
            if tt_text and not has_untrusted_unicode_semantics(tt_text):
                return UnicodeChoice(
                    tt_text,
                    "truetype_cmap",
                    dedupe_alternates(alternates, tt_text),
                )

        if fallback_code != 0:
            predefined_text = predefined_cmap_unicode(self.base_encoding, code_bytes)
            if predefined_text is not None:
                return UnicodeChoice(
                    predefined_text,
                    "predefined_cmap",
                    dedupe_alternates(alternates, predefined_text),
                )

        if not self.is_cid_font and len(code_bytes) == 1 and code_bytes[0] not in self.differences:
            encoding_table = self.encoding_decode_table
            encoding_text = encoding_table[code_bytes[0]]
            if encoding_text and not has_invalid_unicode_mapping(encoding_text):
                return UnicodeChoice(
                    encoding_text,
                    "encoding",
                    dedupe_alternates(alternates, encoding_text),
                )

        cid_unicode_map = self.internal_resolved_cid_unicode_map()
        if cid_unicode_map is not None:
            cid_text = cid_unicode_map.get(fallback_code)
            if cid_text is not None:
                return UnicodeChoice(
                    cid_text,
                    "cid_collection",
                    dedupe_alternates(alternates, cid_text),
                )

        if fallback_code == 0:
            return UnicodeChoice("\u0000", "fallback_nul", dedupe_alternates(alternates, "\u0000"))
        text = unicode_scalar_or_replacement(fallback_code)
        source = "identity" if text != "\ufffd" else "replacement"
        return UnicodeChoice(text, source, dedupe_alternates(alternates, text))

    def internal_clear_unicode_caches(self) -> None:
        """Clear every memo whose value can depend on a Unicode mapping."""
        self.internal_clear_cid_unicode_caches()
        self.simple_glyph_cache.clear()

    def internal_clear_cid_unicode_caches(self) -> None:
        """Clear text and glyph memos derived from CID Unicode mappings."""
        self.decode_cache.clear()
        self.glyphs_cache.clear()
        self.cid_glyph_cache.clear()

    def internal_update_cff_unicode_repairs(self, repairs: Mapping[bytes, str]) -> bool:
        """Install changed CFF repairs and invalidate CID glyphs built without them."""
        changed = any(self.cff_unicode_repairs.get(code) != text for code, text in repairs.items())
        if not changed:
            return False
        self.cff_unicode_repairs.update(repairs)
        self.internal_clear_cid_unicode_caches()
        return True

    def internal_resolved_cid_unicode_map(self) -> Mapping[int, str] | CIDUnicodeMap | None:
        if not self.cid_unicode_map_resolved:
            self.cid_unicode_map = self.internal_cid_unicode_map(
                self.font, vertical=self.is_vertical
            )
            self.cid_unicode_map_resolved = True
        return self.cid_unicode_map

    def internal_true_type_unicode_for_gid(self, gid: int) -> str:
        match self.font_program:
            case TrueTypeFontProgram() as program:
                return program.unicode_for_gid(gid)
            case _:
                return ""

    def internal_visual_punctuation_for_code(self, text: str, *, fallback_code: int) -> str | None:
        """Recover a horizontal punctuation glyph from a misleading ToUnicode map."""
        if len(text) != 1 or not unicodedata.category(text).startswith("M"):
            return None
        match self.font_program:
            case TrueTypeFontProgram() as program:
                bbox = program.glyph_bbox(fallback_code)
            case _:
                return None
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        if width < 450.0 or height <= 0.0 or width / height < 2.5:
            return None
        return "–"

    def internal_apply_simple_unicode_overrides(
        self, choice: UnicodeChoice, code_bytes: bytes
    ) -> UnicodeChoice:
        text = choice.text
        if self.glyph_decode_table is not None:
            glyph_decode_table = self.glyph_decode_table
            if len(code_bytes) == 1:
                mapped = glyph_decode_table[code_bytes[0]]
                if not text:
                    if self.glyph_decode_table_authoritative or mapped:
                        text = mapped
                elif (
                    len(text) == 1
                    and mapped
                    and text != mapped
                    and (
                        choice.source in {"identity", "replacement", "fallback_nul"}
                        or should_prefer_glyph_name_mapping(
                            text,
                            mapped,
                            authoritative=self.glyph_decode_table_authoritative,
                        )
                    )
                ):
                    text = mapped
            else:
                text = replace_unicode_from_glyph_names(
                    text,
                    code_bytes,
                    glyph_decode_table,
                    authoritative=self.glyph_decode_table_authoritative,
                    fallback_mapping=choice.source in {"identity", "replacement", "fallback_nul"},
                )
        if self.ligature_overrides:
            lo = self.ligature_overrides
            text = "".join(lo.get(ord(ch), ch) for ch in text)
        if text == choice.text:
            return choice
        return UnicodeChoice(
            text,
            "glyph_name",
            dedupe_alternates((choice.text, *choice.alternates), text),
        )

    def internal_decode_simple_glyphs(
        self, data: bytes | bytearray | memoryview
    ) -> list[DecodedGlyph]:
        glyphs: list[DecodedGlyph] = []
        glyph_cache = self.simple_glyph_cache
        table = self.byte_decode_table
        if table is None and self.to_unicode is None:
            table = self.encoding_decode_table
        byte_cache = BYTE_CACHE
        for code in data:
            cached_glyph = glyph_cache.get(code)
            if cached_glyph is not None:
                glyphs.append(cached_glyph)
                continue
            chunk = byte_cache[code]
            gid = self.glyph_id_for_code(code)
            if self.to_unicode is not None:
                choice = self.internal_unicode_choice_for_code(chunk, code, gid)
            elif table is not None:
                text = table[code]
                # Preserve the glyph and its geometry even when neither the
                # base encoding nor /Differences defines Unicode for this code.
                # Facades can project this source as their native unknown-glyph
                # spelling (pdfminer uses ``(cid:N)``); the engine retains the
                # standard Unicode replacement character instead of silently
                # losing painted content.
                # The predefined encodings intentionally preserve their raw
                # C0 slots.  Those are legitimate byte-to-text mappings (for
                # example form feed in WinAnsi), unlike an unknown numeric
                # /Differences glyph name which happens to contain a control
                # code.
                undefined = not text or (
                    code in self.differences and len(text) == 1 and ord(text) < 32
                )
                choice = UnicodeChoice(
                    "\ufffd" if undefined else text,
                    "undefined" if undefined else "encoding",
                )
            else:
                choice = self.internal_unicode_choice_for_code(chunk, code, gid)
            choice = self.internal_apply_simple_unicode_overrides(choice, chunk)
            glyph = DecodedGlyph(
                code_bytes=chunk,
                char_code=code,
                cid=code,
                gid=gid,
                unicode=choice.text,
                unicode_source=choice.source,
                alternates=choice.alternates,
                width_code=code,
                bitmap_code=code,
                split_unicode=choice.text in LEGITIMATE_MULTI_CHAR_GLYPHS,
            )
            glyph_cache[code] = glyph
            glyphs.append(glyph)
        return glyphs

    def internal_decode_cid_glyphs(self, data: bytes) -> list[DecodedGlyph]:
        glyph_cache = self.cid_glyph_cache
        entries = self.cmap.decode_entries(data) if self.cmap is not None else []
        if not entries:
            chunks = split_code_bytes(data, self.to_unicode)
            entries = [(chunk, int.from_bytes(chunk, "big") if chunk else 0) for chunk in chunks]
        repair_index = self.cff_unicode_repair_index
        if repair_index is not None:
            to_unicode = self.to_unicode
            mappings = to_unicode.mappings if to_unicode is not None else {}
            repairs = repair_index.repairs_for_codes(
                code_bytes
                for code_bytes, ignored_cid in entries
                if (mapped := mappings.get(code_bytes)) is not None
                and has_invalid_unicode_mapping(mapped)
            )
            if repairs:
                # The same invalid mappings are recomputed on every call, so
                # only a repair that actually moves the table can invalidate
                # glyphs already decoded under the old one.
                self.internal_update_cff_unicode_repairs(repairs)
        glyphs: list[DecodedGlyph] = []
        for code_bytes, cid in entries:
            cached_glyph = glyph_cache.get(code_bytes)
            if cached_glyph is not None:
                glyphs.append(cached_glyph)
                continue
            glyph = self.internal_build_cid_glyph(code_bytes, cid)
            if len(glyph_cache) < internal_CID_GLYPH_CACHE_LIMIT:
                glyph_cache[code_bytes] = glyph
            glyphs.append(glyph)
        return glyphs

    def internal_build_cid_glyph(self, code_bytes: bytes, cid: int) -> DecodedGlyph:
        """Build one CID glyph after the caller has missed the per-code cache."""
        char_code = int.from_bytes(code_bytes, "big") if code_bytes else 0
        gid = self.glyph_id_for_code(cid)
        if gid is not None and gid != 0 and not self.internal_glyph_exists(gid):
            notdef_cid = self.cmap.mapped_notdef(code_bytes) if self.cmap else None
            if notdef_cid is not None:
                cid = notdef_cid
                gid = self.glyph_id_for_code(cid)
        choice = self.internal_unicode_choice_for_code(code_bytes, cid, gid)
        if self.ligature_overrides:
            lo = self.ligature_overrides
            text = "".join(lo.get(ord(ch), ch) for ch in choice.text)
            if text != choice.text:
                choice = UnicodeChoice(
                    text,
                    "ligature_override",
                    dedupe_alternates((choice.text, *choice.alternates), text),
                )
        return DecodedGlyph(
            code_bytes=code_bytes,
            char_code=char_code,
            cid=cid,
            gid=gid,
            unicode=choice.text,
            unicode_source=choice.source,
            alternates=choice.alternates,
            width_code=cid,
            bitmap_code=cid,
            split_unicode=choice.text in LEGITIMATE_MULTI_CHAR_GLYPHS,
        )

    def internal_glyph_exists(self, gid: int) -> bool:
        program = self.font_program
        return program.has_glyph_id(gid) if program is not None else True

    def glyph_id_for_code(self, code: int) -> int | None:
        match self.font_program:
            case CFFFont() as program:
                if self.is_cid_font:
                    return program.glyph_id_for_cid(code)
                return program.glyph_id_for_name(self.internal_simple_glyph_name(code))
            case TrueTypeFontProgram() as program:
                if not self.is_cid_font and 0 <= code < 256 and program.cmap:
                    glyph_text = glyph_name_to_unicode(self.internal_simple_glyph_name(code))
                    if len(glyph_text) == 1:
                        return program.glyph_id_for_unicode(ord(glyph_text))
                return program.glyph_id_for_code(code)
            case Type1FontProgram() as program:
                return program.glyph_id_for_name(self.internal_simple_glyph_name(code))
            case OpenTypeFontProgram() as program:
                if self.is_cid_font:
                    return code
                return program.glyph_id_for_name(self.internal_simple_glyph_name(code))
            case _:
                return code

    def internal_simple_glyph_name(self, code: int) -> str:
        if 0 <= code < 256:
            return self.simple_encoding_glyph_names[code]
        return ".notdef"

    def glyph_bbox(self, code: int) -> Rectangle | None:
        if code < 0:
            return None
        cache = self.glyph_bbox_cache
        bbox = cache.get(code)
        if bbox is not None:
            return bbox
        if code in cache:
            return None
        program = self.font_program
        if program is None:
            width = self.glyph_width(code)
            bbox = None if width <= 0 else (0.0, self.descent, width, self.ascent)
        else:
            glyph_id = self.glyph_id_for_code(code)
            bbox = program.glyph_bbox_for_gid(glyph_id) if glyph_id is not None else None
        if len(cache) >= 4096:
            cache.clear()
        cache[code] = bbox
        return bbox

    def vertical_glyph_metric(self, code: int) -> tuple[float, float, float]:
        """Return the explicit W2 metric or the DW2/width-derived fallback."""
        metric = self.vertical_metrics.get(code)
        if metric is None:
            metric = (
                self.default_vertical_displacement_y,
                self.glyph_width(code) / 2.0,
                self.default_vertical_origin_y,
            )
        return metric

    def vertical_glyph_position(self, code: int, *, font_size: float) -> tuple[float, float]:
        metric = self.vertical_glyph_metric(code)
        scale = font_size / 1000.0
        return (-metric[1] * scale, -metric[2] * scale)

    def glyph_bitmap(self, code: int, *, width: int = 24, height: int = 32) -> tuple[int, ...]:
        if code < 0:
            return ()
        cache_key = (code, width, height)
        cached = self.glyph_bitmap_cache.get(cache_key)
        if cached is not None:
            return cached
        glyph_id = self.glyph_id_for_code(code)
        program = self.font_program
        bitmap = (
            program.glyph_bitmap_for_gid(glyph_id, width=width, height=height)
            if program is not None and glyph_id is not None
            else ()
        )
        if len(self.glyph_bitmap_cache) >= 512:
            self.glyph_bitmap_cache.clear()
        self.glyph_bitmap_cache[cache_key] = bitmap
        return bitmap

    def glyph_outline(
        self, code: int, gid: int | None = None, text: str = ""
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Resolve an embedded glyph outline without rasterizing it.

        The returned points use PDF's conventional 1000-unit glyph space so
        capture can apply the exact text matrix at composition time. Missing or
        malformed font programs deliberately return no outline.
        """
        if code < 0:
            return ()
        glyph_id = gid if gid is not None else self.glyph_id_for_code(code)
        if glyph_id is None:
            return ()
        program = self.font_program
        if program is not None:
            return program.normalized_glyph_contours(glyph_id)
        from core_pdf.impl.spec.s_09_fonts.fallback import fallback_glyph_outline

        return fallback_glyph_outline(
            self.font_name,
            text,
            is_cid_font=self.is_cid_font,
            is_vertical=self.is_vertical,
            cid_registry=self.cid_registry,
            cid_ordering=self.cid_ordering,
            provider=self.raster_font_provider,
        )

    def glyph_width(self, code: int) -> float:
        default_width = self.default_width
        fallback = default_width if default_width > 0.0 else 250.0 if code == 32 else 1000.0
        return self.widths.width_for(code, fallback)

    def glyph_advance_vector(
        self,
        code: int,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        encoded_space: bool,
    ) -> tuple[float, float]:
        """Return one glyph's signed text-space displacement.

        PDF 32000-1 9.4.4 defines vertical ``w1`` as a signed y
        displacement. Keeping that sign here gives text-matrix updates and
        geometry capture one canonical arithmetic path.
        """
        spacing = char_space + (word_space if encoded_space else 0.0)
        if self.is_vertical:
            displacement_y = self.vertical_glyph_metric(code)[0] * font_size / 1000.0
            return (0.0, displacement_y + spacing)
        displacement_x = self.glyph_width(code) * font_size / 1000.0 + spacing
        return (displacement_x * horizontal_scale / 100.0, 0.0)

    def text_advance_vector(
        self,
        data: bytes | bytearray | memoryview,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        glyphs: tuple[DecodedGlyph, ...] | None = None,
    ) -> tuple[float, float]:
        data = bytes(data)
        if not data:
            return (0.0, 0.0)
        if glyphs is None:
            glyphs = self.decode_glyphs(data)

        total_x = 0.0
        total_y = 0.0
        for glyph in glyphs:
            advance_x, advance_y = self.glyph_advance_vector(
                glyph.width_code,
                font_size=font_size,
                char_space=char_space,
                word_space=word_space,
                horizontal_scale=horizontal_scale,
                encoded_space=glyph.code_bytes == b" ",
            )
            total_x += advance_x
            total_y += advance_y
        return (total_x, total_y)


def dedupe_alternates(values: Iterable[str], selected: str) -> tuple[str, ...]:
    seen = {selected}
    alternates: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        alternates.append(value)
    return tuple(alternates)
