# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

from core_pdf.impl.engine.spec.s_07_objects.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.cff import (
    build_cff_unicode_repairs,
    cff_font_for_pdf_font,
)
from core_pdf.impl.engine.spec.s_09_fonts.cid_unicode import (
    CIDUnicodeMap,
    resolve_cid_unicode_map,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import BYTE_CACHE
from core_pdf.impl.engine.spec.s_09_fonts.font_names import resolve_base_font_name
from core_pdf.impl.engine.spec.s_09_fonts.glyph_decode import (
    build_glyph_decode_table,
    replace_unicode_from_glyph_names,
    should_prefer_glyph_name_mapping,
)
from core_pdf.impl.engine.spec.s_09_fonts.helpers import (
    cached_decode_table,
    parse_differences,
)
from core_pdf.impl.engine.spec.s_09_fonts.metrics import (
    adjust_type3_widths,
    parse_font_metrics,
)
from core_pdf.impl.engine.spec.s_09_fonts.truetype import tt_font_for_pdf_font
from core_pdf.impl.engine.spec.s_09_fonts.widths import (
    get_descendant,
    parse_font_widths,
)
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.primitives import PdfString
from core_pdf.impl.third_party._vendor.fontTools.agl import UV2AGL
from core_pdf.impl.third_party._vendor.fontTools.encodings.StandardEncoding import StandardEncoding
from core_pdf.impl.third_party.cff import CFFFont
from core_pdf.impl.third_party.cid.cmap import (
    CMapDecoder,
    ToUnicodeCMap,
    code_in_ranges,
)
from core_pdf.impl.third_party.cid.resource_loader import (
    predefined_cmap_unicode,
    resolve_cmap_decoder,
    resolve_cmap_resource,
)
from core_pdf.impl.third_party.cid.widths import FontWidthMap
from core_pdf.impl.third_party.truetype import TrueTypeFontProgram

if typing.TYPE_CHECKING:
    from typing import Any


LEGITIMATE_MULTI_CHAR_GLYPHS = frozenset({"ff", "fi", "fl", "ffi", "ffl", "st"})
TYPE1_ENCODING_ENTRY_RE = re.compile(rb"\bdup\s+(\d{1,3})\s+/([A-Za-z0-9_.]+)\s+put\b")


def unicode_scalar_or_replacement(codepoint: int) -> str:
    if 0 <= codepoint < 0x110000 and not 0xD800 <= codepoint <= 0xDFFF:
        return chr(codepoint)
    return "\ufffd"


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


def uses_fixed_two_byte_codes(cmap: CMapDecoder | ToUnicodeCMap | None, data: bytes) -> bool:
    if cmap is None:
        return False
    if getattr(cmap, "decode_lengths", None) == (2,):
        return True
    ranges = getattr(cmap, "code_space_ranges", None)
    if not ranges:
        return False
    first = data[:2]
    first_in_range = False
    for start, end in ranges:
        if len(start) != 2 or len(end) != 2:
            return False
        if not first_in_range and code_in_ranges(first, ((start, end),)):
            first_in_range = True
    return first_in_range


class FontDecoder:
    __slots__ = (
        "font",
        "ligature_overrides",
        "to_unicode",
        "cmap",
        "cid_unicode_map",
        "cid_unicode_map_resolved",
        "base_encoding",
        "differences",
        "is_cid_font",
        "is_type3",
        "byte_decode_table",
        "widths",
        "default_width",
        "vertical_widths",
        "default_vertical_width",
        "vertical_metrics",
        "is_vertical",
        "ascent",
        "descent",
        "decode_cache",
        "glyphs_cache",
        "fast_widths_cache",
        "glyph_bbox_cache",
        "glyph_bitmap_cache",
        "type3_glyph_names",
        "fast_widths_cid",
        "fast_widths_cid_unavailable",
        "font_name",
        "glyph_decode_table",
        "glyph_decode_table_authoritative",
        "cff_unicode_repairs",
        "cff_font",
        "tt_font",
        "lazy_initialized",
    )

    font: dict[str, Any]
    ligature_overrides: dict[int, str]
    to_unicode: ToUnicodeCMap | None
    cmap: CMapDecoder | None
    cid_unicode_map: Mapping[int, str] | CIDUnicodeMap | None
    cid_unicode_map_resolved: bool
    base_encoding: str | None
    differences: dict[int, str]
    is_cid_font: bool
    is_type3: bool
    byte_decode_table: tuple[str, ...] | None
    widths: FontWidthMap
    default_width: float
    vertical_widths: FontWidthMap
    default_vertical_width: float
    vertical_metrics: dict[int, tuple[float, float, float]]
    is_vertical: bool
    ascent: float
    descent: float
    fast_widths_cache: tuple[float, ...] | None
    glyph_bbox_cache: dict[int, tuple[float, float, float, float] | None]
    fast_widths_cid: list[float] | None
    fast_widths_cid_unavailable: bool
    font_name: str | None
    glyph_decode_table: tuple[str, ...] | None
    glyph_decode_table_authoritative: bool
    type3_glyph_names: dict[int, str] | None
    cff_unicode_repairs: dict[bytes, str]
    cff_font: CFFFont | None
    tt_font: TrueTypeFontProgram | None

    def __init__(
        self,
        font: dict[str, Any],
        ligature_overrides: dict[int, str] | None = None,
    ) -> None:
        self.font = font
        self.ligature_overrides = ligature_overrides if ligature_overrides is not None else {}
        self.decode_cache: dict[bytes, str] = {}
        self.glyphs_cache: dict[bytes, tuple[DecodedGlyph, ...]] = {}
        self.fast_widths_cache = None
        self.glyph_bbox_cache = {}
        self.glyph_bitmap_cache: dict[tuple[int, int, int], tuple[int, ...]] = {}
        self.type3_glyph_names = None
        # Flag to track whether full decoder initialization has run.
        # Defers expensive metrics / CMap / encoding parsing until first use.
        self.lazy_initialized = False

    def __getattr__(self, name: str) -> Any:
        # Trigger full parser initialization on first access to any unset slot attribute
        if not self.lazy_initialized:
            self.lazy_initialized = True
            self.__post_init__()
            return getattr(self, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __post_init__(self) -> None:
        font = self.font
        subtype = lookup_dict_key(font, "Subtype")
        if subtype is not None:
            subtype = normalize_pdf_name(subtype)

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

        cmap, base_encoding, differences = self.parse_encoding(font)
        (
            widths,
            default_width,
            is_vertical,
            vertical_widths,
            default_vertical_width,
            vertical_metrics,
        ) = parse_font_widths(font, subtype)
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

        byte_decode_table: tuple[str, ...] | None = None
        if to_unicode is None and not is_cid_font:
            key = base_encoding or ("Type3" if is_type3 else "")
            byte_decode_table = cached_decode_table(key, tuple(sorted(differences.items())))

        self.to_unicode = to_unicode
        self.cmap = cmap
        self.cid_unicode_map = None
        self.cid_unicode_map_resolved = not is_cid_font
        self.base_encoding = base_encoding
        self.differences = differences
        self.is_cid_font = is_cid_font
        self.is_type3 = is_type3
        self.byte_decode_table = byte_decode_table
        self.widths = widths
        self.default_width = default_width
        self.vertical_widths = vertical_widths
        self.default_vertical_width = default_vertical_width
        self.vertical_metrics = vertical_metrics
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
        self.cff_font = cff_font_for_pdf_font(font)
        self.tt_font = tt_font_for_pdf_font(font)
        self.cff_unicode_repairs = build_cff_unicode_repairs(font, to_unicode, cmap)

        self.fast_widths_cid = None
        self.fast_widths_cid_unavailable = False

    @staticmethod
    def _cid_system_info_string(value: object) -> str | None:
        if isinstance(value, PdfString):
            return value.data.decode("latin-1")
        normalized = normalize_pdf_name(value)
        if normalized is not None:
            return normalized
        if isinstance(value, bytes):
            return value.decode("latin-1")
        return None

    @classmethod
    def _cid_unicode_map(
        cls,
        font: dict[str, Any],
        *,
        vertical: bool,
    ) -> Mapping[int, str] | CIDUnicodeMap | None:
        descendant = get_descendant(font)
        if descendant is None:
            return None
        system_info = lookup_dict_key(descendant, "CIDSystemInfo")
        if not isinstance(system_info, dict):
            system_info = lookup_dict_key(font, "CIDSystemInfo")
        if not isinstance(system_info, dict):
            return None
        registry = cls._cid_system_info_string(lookup_dict_key(system_info, "Registry"))
        ordering = cls._cid_system_info_string(lookup_dict_key(system_info, "Ordering"))
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
    ) -> tuple[CMapDecoder | None, str | None, dict[int, str]]:
        cmap = None
        base_encoding = None
        differences: dict[int, str] = {}
        subtype = normalize_pdf_name(lookup_dict_key(font, "Subtype"))
        encoding_obj = lookup_dict_key(font, "Encoding")
        if isinstance(encoding_obj, str):
            base_encoding = normalize_pdf_name(encoding_obj)
            cmap = self._named_cmap(base_encoding)
        elif isinstance(encoding_obj, PdfStream):
            try:
                cmap = CMapDecoder(
                    encoding_obj.data,
                    usecmap_resolver=resolve_cmap_decoder,
                )
            except (PdfParseError, ValueError):
                cmap = None
        elif isinstance(encoding_obj, dict):
            base_encoding = normalize_pdf_name(lookup_dict_key(encoding_obj, "BaseEncoding"))
            if base_encoding is None:
                base_encoding = "WinAnsiEncoding" if subtype == "TrueType" else "StandardEncoding"
            differences_obj = lookup_dict_key(encoding_obj, "Differences")
            if differences_obj is not None and not isinstance(differences_obj, (list, tuple)):
                differences_obj = None
            differences = parse_differences(
                list(differences_obj) if isinstance(differences_obj, tuple) else differences_obj,
                normalize_pdf_name,
            )
        else:
            base_encoding = normalize_pdf_name(encoding_obj)
            cmap = self._named_cmap(base_encoding)
        if not differences and subtype == "Type1":
            descriptor = lookup_dict_key(font, "FontDescriptor")
            if isinstance(descriptor, dict):
                font_file = lookup_dict_key(descriptor, "FontFile")
                if isinstance(font_file, PdfStream):
                    differences = parse_type1_font_program_encoding(font_file.data)
        return cmap, base_encoding, differences

    def _named_cmap(self, base_encoding: str | None) -> CMapDecoder | None:
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
        data = bytes(data)
        if not data:
            return ()
        use_cache = len(data) <= 16
        if use_cache:
            cached = self.glyphs_cache.get(data)
            if cached is not None:
                return cached

        if self.is_cid_font:
            glyphs = self._decode_cid_glyphs(data)
        else:
            glyphs = self._decode_simple_glyphs(data)
        result = tuple(glyphs)
        if use_cache and len(self.glyphs_cache) < 512:
            self.glyphs_cache[data] = result
        return result

    def _unicode_choice_for_code(
        self, code_bytes: bytes, fallback_code: int, gid: int | None = None
    ) -> UnicodeChoice:
        alternates: list[str] = []
        to_unicode_text = None
        if self.to_unicode is not None:
            to_unicode_text = self.to_unicode.mappings.get(code_bytes)
            if to_unicode_text is not None:
                alternates.append(to_unicode_text)

        if to_unicode_text is not None:
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
            tt_text = self._true_type_unicode_for_gid(gid)
            if tt_text:
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

        cid_unicode_map = self._resolved_cid_unicode_map()
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

    def _resolved_cid_unicode_map(self) -> Mapping[int, str] | CIDUnicodeMap | None:
        if not self.cid_unicode_map_resolved:
            self.cid_unicode_map = self._cid_unicode_map(self.font, vertical=self.is_vertical)
            self.cid_unicode_map_resolved = True
        return self.cid_unicode_map

    def _true_type_unicode_for_gid(self, gid: int) -> str:
        tt_font = self.tt_font
        if tt_font is None:
            return ""
        return tt_font.unicode_for_gid(gid)

    def _apply_simple_unicode_overrides(
        self, choice: UnicodeChoice, code_bytes: bytes
    ) -> UnicodeChoice:
        if choice.source == "to_unicode":
            return choice
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
                    and should_prefer_glyph_name_mapping(
                        text,
                        mapped,
                        authoritative=self.glyph_decode_table_authoritative,
                    )
                ):
                    text = mapped
            else:
                text = replace_unicode_from_glyph_names(
                    text,
                    code_bytes,
                    glyph_decode_table,
                    authoritative=self.glyph_decode_table_authoritative,
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

    def _decode_simple_glyphs(self, data: bytes) -> list[DecodedGlyph]:
        glyphs: list[DecodedGlyph] = []
        table = self.byte_decode_table
        if table is None and self.to_unicode is None:
            key = self.base_encoding or ("Type3" if self.is_type3 else "")
            table = cached_decode_table(key, tuple(sorted(self.differences.items())))
        byte_cache = BYTE_CACHE
        for code in data:
            chunk = byte_cache[code]
            gid = self.glyph_id_for_code(code)
            if self.to_unicode is not None:
                choice = self._unicode_choice_for_code(chunk, code, gid)
            elif table is not None:
                text = table[code]
                choice = UnicodeChoice(text, "encoding")
            else:
                choice = self._unicode_choice_for_code(chunk, code, gid)
            choice = self._apply_simple_unicode_overrides(choice, chunk)
            glyphs.append(
                DecodedGlyph(
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
            )
        return glyphs

    def _decode_cid_glyphs(self, data: bytes) -> list[DecodedGlyph]:
        entries = self.cmap.decode(data) if self.cmap is not None else []
        if not entries:
            chunks = split_code_bytes(data, self.to_unicode)
            entries = [(chunk, int.from_bytes(chunk, "big") if chunk else 0) for chunk in chunks]
        glyphs: list[DecodedGlyph] = []
        for code_bytes, cid in entries:
            char_code = int.from_bytes(code_bytes, "big") if code_bytes else 0
            gid = self.glyph_id_for_code(cid)
            if gid is not None and gid != 0 and not self._glyph_exists(gid):
                notdef_cid = self.cmap.mapped_notdef(code_bytes) if self.cmap else None
                if notdef_cid is not None:
                    cid = notdef_cid
                    gid = self.glyph_id_for_code(cid)
            choice = self._unicode_choice_for_code(code_bytes, cid, gid)
            if self.ligature_overrides:
                lo = self.ligature_overrides
                text = "".join(lo.get(ord(ch), ch) for ch in choice.text)
                if text != choice.text:
                    choice = UnicodeChoice(
                        text,
                        "ligature_override",
                        dedupe_alternates((choice.text, *choice.alternates), text),
                    )
            glyphs.append(
                DecodedGlyph(
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
            )
        return glyphs

    def _glyph_exists(self, gid: int) -> bool:
        if self.cff_font is not None:
            return self.cff_font.has_glyph_id(gid)
        if self.tt_font is not None:
            return self.tt_font.has_glyph_id(gid)
        return True

    def get_fast_widths_cid(self) -> list[float] | None:
        if self.fast_widths_cid is not None:
            return self.fast_widths_cid
        if self.fast_widths_cid_unavailable:
            return None
        explicit_count = self.widths.explicit_count
        if not self.is_cid_font and explicit_count <= 20:
            self.fast_widths_cid_unavailable = True
            return None
        if explicit_count > 4096:
            self.fast_widths_cid_unavailable = True
            return None
        dw = self.default_width
        dw_pos = dw if dw > 0.0 else 1000.0
        space_w = dw if dw > 0.0 else 250.0
        table = [dw_pos] * 65536
        table[32] = space_w
        for k, v in self.widths.iter_explicit_widths():
            if 0 <= k < 65536:
                table[k] = v
        self.fast_widths_cid = table
        return table

    def glyph_id_for_code(self, code: int) -> int | None:
        cff_font = self.cff_font
        if cff_font is not None:
            if self.is_cid_font:
                return cff_font.glyph_id_for_cid(code)
            return cff_font.glyph_id_for_name(self._simple_glyph_name(code))
        tt_font = self.tt_font
        if tt_font is not None:
            if not self.is_cid_font and self.byte_decode_table is not None:
                text = self.byte_decode_table[code] if 0 <= code < 256 else ""
                if len(text) == 1:
                    return tt_font.glyph_id_for_unicode(ord(text))
            return tt_font.glyph_id_for_code(code)
        return code

    def _simple_glyph_name(self, code: int) -> str:
        name = self.differences.get(code)
        if name is not None:
            return name
        if self.base_encoding in {None, "StandardEncoding"} and 0 <= code < 256:
            return StandardEncoding[code]
        table = self.byte_decode_table
        text = table[code] if table is not None and 0 <= code < len(table) else ""
        if len(text) != 1:
            return ".notdef"
        return UV2AGL.get(ord(text), f"uni{ord(text):04X}")

    def glyph_bbox(self, code: int) -> tuple[float, float, float, float] | None:
        if code < 0:
            return None
        cache = self.glyph_bbox_cache
        if code in cache:
            return cache[code]
        cff_font = self.cff_font
        if cff_font is not None:
            if self.is_cid_font:
                bbox = cff_font.glyph_bbox(code)
            else:
                glyph_id = self.glyph_id_for_code(code)
                bbox = cff_font.glyph_bbox_for_gid(glyph_id or 0)
        else:
            tt_font = self.tt_font
            if tt_font is not None:
                bbox = tt_font.glyph_bbox(code)
            else:
                width = self.glyph_width(code)
                bbox = None if width <= 0 else (0.0, self.descent, width, self.ascent)
        if len(cache) >= 4096:
            cache.clear()
        cache[code] = bbox
        return bbox

    def vertical_glyph_position(self, code: int, *, font_size: float) -> tuple[float, float]:
        metric = self.vertical_metrics.get(
            code, (self.default_vertical_width, self.glyph_width(code) / 2.0, 0.0)
        )
        scale = font_size / 100000.0
        return (metric[1] * scale, metric[2] * scale)

    def glyph_bitmap(self, code: int, *, width: int = 24, height: int = 32) -> tuple[int, ...]:
        if code < 0:
            return ()
        cache_key = (code, width, height)
        cached = self.glyph_bitmap_cache.get(cache_key)
        if cached is not None:
            return cached
        bitmap: tuple[int, ...] = ()
        cff_font = self.cff_font
        if cff_font is not None:
            if self.is_cid_font:
                bitmap = cff_font.glyph_bitmap(code, width=width, height=height)
            else:
                glyph_id = self.glyph_id_for_code(code)
                bitmap = cff_font.glyph_bitmap_for_gid(
                    glyph_id or 0,
                    width=width,
                    height=height,
                )
            if bitmap:
                self.glyph_bitmap_cache[cache_key] = bitmap
                return bitmap
        tt_font = self.tt_font
        if tt_font is not None:
            bitmap = tt_font.glyph_bitmap(code, width=width, height=height)
        if len(self.glyph_bitmap_cache) >= 512:
            self.glyph_bitmap_cache.clear()
        self.glyph_bitmap_cache[cache_key] = bitmap
        return bitmap

    def glyph_width(self, code: int) -> float:
        if self.fast_widths_cid is not None:
            if 0 <= code < 65536:
                return self.fast_widths_cid[code]
        elif self.is_cid_font:
            widths_cid = self.get_fast_widths_cid()
            if widths_cid is not None and 0 <= code < 65536:
                return widths_cid[code]
        if 0 <= code < 256:
            return self.fast_widths[code]
        return self.widths.width_for(code, self.default_width)

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

        cs = char_space * 1000.0 / font_size if font_size else 0.0
        ws = word_space * 1000.0 / font_size if font_size else 0.0
        scale = font_size * horizontal_scale / 100000.0

        if self.is_vertical:
            if glyphs is None:
                glyphs = self.decode_glyphs(data)
            total = 0.0
            for glyph in glyphs:
                metric = self.vertical_metrics.get(
                    glyph.cid,
                    (self.default_vertical_width, self.glyph_width(glyph.cid) / 2.0, 0.0),
                )
                total += metric[0] * font_size / 100000.0 + char_space
                if glyph.char_code == 32:
                    total += word_space
            return (0.0, -total)

        if (
            glyphs is None
            and not self.is_cid_font
            and self.to_unicode is None
            and self.cmap is None
        ):
            widths = self.fast_widths
            total = 0.0
            space_count = 0
            for b in data:
                total += widths[b]
                if b == 32:
                    space_count += 1
            total += len(data) * cs + space_count * ws
            return (total * scale, 0.0)

        n = len(data)
        dw = self.default_width
        dw_pos = dw if dw > 0.0 else 1000.0
        space_w = dw if dw > 0.0 else 250.0
        if glyphs is None and n == 1:
            code = data[0]
            fwc_single = self.fast_widths_cid
            if fwc_single is not None:
                total = fwc_single[code]
            else:
                total = self.widths.width_for(code, space_w if code == 32 else dw_pos)
            total += cs
            if code == 32:
                total += ws
            if self.is_vertical:
                return (0.0, -total * scale)
            return (total * scale, 0.0)

        cmap_has_explicit_mapping = bool(
            self.cmap and (self.cmap.cid_mappings or self.cmap.cid_ranges)
        )
        if glyphs is None and self.is_cid_font and n % 2 == 0 and not cmap_has_explicit_mapping:
            fwc = self.fast_widths_cid
            if fwc is None:
                fwc = self.get_fast_widths_cid()
            if fwc is not None:
                total = 0.0
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    total += fwc[code]
                    if code == 32:
                        total += ws
                total += (n >> 1) * cs

                if self.is_vertical:
                    return (0.0, -total * scale)
                return (total * scale, 0.0)

        cmap = self.to_unicode or self.cmap
        if (
            glyphs is None
            and cmap is not None
            and n % 2 == 0
            and uses_fixed_two_byte_codes(cmap, data)
            and not cmap_has_explicit_mapping
        ):
            total = 0.0
            fwc = self.fast_widths_cid
            if fwc is None and (self.is_cid_font or self.widths.explicit_count > 20):
                fwc = self.get_fast_widths_cid()
            if fwc is not None:
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    total += fwc[code] if 0 <= code < 65536 else dw_pos
                    if code == 32:
                        total += ws
            else:
                width_map = self.widths
                for i in range(0, n, 2):
                    code = (data[i] << 8) | data[i + 1]
                    w = width_map.width_for(code, space_w if code == 32 else dw_pos)
                    total += w
                    if code == 32:
                        total += ws
            total += (n >> 1) * cs

            if self.is_vertical:
                return (0.0, -total * scale)
            return (total * scale, 0.0)

        if glyphs is None:
            glyphs = self.decode_glyphs(data)

        total = 0.0
        fwc = self.fast_widths_cid

        for glyph in glyphs:
            code = glyph.width_code
            if fwc is not None:
                w = fwc[code] if 0 <= code < 65536 else dw_pos
            else:
                w = self.widths.width_for(code, space_w if code == 32 else dw_pos)
            total += w + cs
            if code == 32:
                total += ws

        if self.is_vertical:
            return (0.0, -total * scale)
        return (total * scale, 0.0)


def dedupe_alternates(values: Iterable[str], selected: str) -> tuple[str, ...]:
    seen = {selected}
    alternates: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        alternates.append(value)
    return tuple(alternates)
