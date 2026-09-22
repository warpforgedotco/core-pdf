# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
import typing
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import suppress
from copy import replace
from functools import cache
from io import BytesIO
from typing import Any, ClassVar

import numpy

from core_adobe_fonts.cmap.ranges import (
    code_in_ranges,
    iter_codespace_range,
)
from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.fonts.cmap_resources import (
    CID_COLLECTION_UNICODE_OVERRIDES,
    CID_COLLECTION_UNICODE_SOURCES,
    predefined_cmap_unicode,
    resolve_cmap_decoder,
    resolve_cmap_resource,
    unicode_candidate_preference,
    unicode_scalar_from_cmap_code,
)
from core_pdf.impl.fonts.cmap_tokenizer import CMapDecoder
from core_pdf.impl.fonts.cmap_tounicode import ToUnicodeCMap, unicode_scalar_or_replacement
from core_pdf.impl.fonts.fallback import fallback_glyph_outline
from core_pdf.impl.fonts.font_program import (
    FONT_PROGRAM_ERRORS,
    LEGITIMATE_MULTI_CHAR_GLYPHS,
    CFFFont,
    CFFUnicodeRepairIndex,
    OpenTypeFontProgram,
    TrueTypeFontProgram,
    Type1FontProgram,
    cached_truetype_program,
    is_repairable_to_unicode_label,
    parse_type1_font_program_encoding,
)
from core_pdf.impl.fonts.glyphs import glyph_name_to_unicode
from core_pdf.impl.fonts.helpers import (
    build_decode_table,
    build_simple_encoding_glyph_names,
    parse_differences,
    strip_subset_tag,
    unicode_for_glyph_name,
)
from core_pdf.impl.fonts.metrics import (
    adjust_type3_widths,
    parse_font_metrics,
    standard_14_widths,
)
from core_pdf.impl.fonts.widths import (
    get_descendant,
    parse_font_widths,
)
from core_pdf.impl.model.glyphs import UnicodeSource
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.records import Record, ReplaceFields, ReprFields
from core_pdf.impl.types import PdfString, Rectangle
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_int_strict
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.dictionaries import (
    FontProgramInputs,
    prepare_font_program_inputs,
)
from core_pdf_spec.s_09_fonts.helpers import (
    BASE_ENCODING_GLYPH_NAMES,
)
from core_pdf_spec.s_09_fonts.metrics import glyph_advance_vector as pdf_glyph_advance_vector
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.standards import SemanticContext

if typing.TYPE_CHECKING:
    from core_pdf.impl.fonts.fallback import RasterFontProviderLike

frozen_setattr = object.__setattr__


FontProgram = CFFFont | TrueTypeFontProgram | Type1FontProgram | OpenTypeFontProgram


TYPE1_ENCODING_ENTRY_RE = re.compile(rb"\bdup\s+(\d{1,3})\s+/([A-Za-z0-9_.]+)\s+put\b")


def descriptor_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    descriptor = font.get("FontDescriptor")
    if subtype == "Type0":
        descendant = get_descendant(font)
        if isinstance(descendant, dict):
            descendant_descriptor = descendant.get("FontDescriptor")
            descriptor = descendant_descriptor or descriptor
    if not isinstance(descriptor, dict):
        return None
    return recover_pdf_name(descriptor.get("FontName"))


def resolve_base_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    base_font_name = recover_pdf_name(font.get("BaseFont"))
    if base_font_name is not None:
        return base_font_name
    return descriptor_font_name(font, subtype)


def tt_font(inputs: FontProgramInputs) -> TrueTypeFontProgram | None:
    if inputs.subtype not in {"CIDFontType2", "TrueType"}:
        return None
    font_file = inputs.font_file2
    if font_file is None:
        return None
    cid_to_gid = None
    if inputs.descendant is not None:
        cid_to_gid_obj = inputs.descendant.get("CIDToGIDMap")
        if isinstance(cid_to_gid_obj, PdfStream):
            cid_to_gid = cid_to_gid_obj.data
    try:
        return cached_truetype_program(
            font_file.data, cid_to_gid, use_cmap=inputs.descendant is None
        )
    except ValueError:
        return None


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


def cff_font(inputs: FontProgramInputs) -> CFFFont | None:
    if inputs.descendant is not None:
        if inputs.subtype != "CIDFontType0":
            return None
    elif inputs.subtype not in {"Type1", "MMType1"}:
        return None
    font_file = inputs.font_file3
    if font_file is None:
        return None
    subtype = recover_pdf_name(font_file.dictionary.get("Subtype"))
    if inputs.descendant is None and subtype not in {"Type1C", "OpenType"}:
        return None
    font_data: bytes | None = font_file.data
    if subtype == "OpenType":
        if font_data is None:
            return None
        font_data = extract_cff_table(font_data)
        if font_data is None:
            return None
    try:
        return CFFFont(font_data)
    except ValueError:
        return None


def build_cff_unicode_repair_index(
    font: dict[str, Any],
    font_program: FontProgram | None,
    to_unicode: ToUnicodeCMap | None,
    cmap: CMapDecoder | None,
) -> CFFUnicodeRepairIndex | None:
    if to_unicode is None or not isinstance(font_program, CFFFont):
        return None
    descendant = get_descendant(font)
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
    if not any(is_repairable_to_unicode_label(value) for cid_value, value in mapping.values()):
        return None
    mapping_items = tuple(sorted((code, cid, value) for code, (cid, value) in mapping.items()))
    return CFFUnicodeRepairIndex(font_program, mapping_items)


def extract_cff_table(data: bytes) -> bytes | None:
    font: TTFont | None = None
    try:
        font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
        reader = font.reader
        if reader is None:
            return None
        table = reader.tables.get("CFF ")
        return table.loadData(reader.file) if table is not None else None
    except FONT_PROGRAM_ERRORS:
        return None
    finally:
        if font is not None:
            with suppress(AttributeError):
                font.close()


def type1_font(inputs: FontProgramInputs) -> Type1FontProgram | None:
    if inputs.original_subtype not in {"Type1", "MMType1"}:
        return None
    font_file = inputs.font_file
    if font_file is None:
        return None
    length1_value = font_file.dictionary.get("Length1")
    try:
        length1 = int(length1_value) if isinstance(length1_value, (int, float)) else None
        return Type1FontProgram(font_file.data, length1=length1)
    except TypeError, ValueError, OverflowError:
        return None


def opentype_font(inputs: FontProgramInputs) -> OpenTypeFontProgram | None:
    font_file = inputs.font_file3
    if font_file is None:
        return None
    if recover_pdf_name(font_file.dictionary.get("Subtype")) != "OpenType":
        return None
    try:
        return OpenTypeFontProgram(font_file.data)
    except ValueError:
        return None


def font_program_for_pdf_font(font: dict[str, Any]) -> FontProgram | None:
    try:
        inputs = prepare_font_program_inputs(font)
    except ValueError:
        descendant = get_descendant(font)
        font_dict = descendant if descendant is not None else font
        descriptor = font_dict.get("FontDescriptor")
        original_descriptor = font.get("FontDescriptor")
        streams = (
            original_descriptor.get("FontFile") if isinstance(original_descriptor, dict) else None,
            descriptor.get("FontFile2") if isinstance(descriptor, dict) else None,
            descriptor.get("FontFile3") if isinstance(descriptor, dict) else None,
        )
        first, second, third = (
            value if isinstance(value, PdfStream) else None for value in streams
        )
        inputs = FontProgramInputs(
            recover_pdf_name(font_dict.get("Subtype")),
            recover_pdf_name(font.get("Subtype")),
            descendant,
            first,
            second,
            third,
        )
    else:
        font_dict = inputs.descendant if inputs.descendant is not None else font
        subtype = recover_pdf_name(font_dict.get("Subtype"))
        original_subtype = recover_pdf_name(font.get("Subtype"))
        if (subtype, original_subtype) != (inputs.subtype, inputs.original_subtype):
            inputs = replace(inputs, subtype=subtype, original_subtype=original_subtype)
    for resolver in (
        cff_font,
        tt_font,
        type1_font,
        opentype_font,
    ):
        program = resolver(inputs)
        if program is not None:
            return program
    return None


class DecodedGlyph(DecodedFontGlyph, ReplaceFields, ReprFields):
    __slots__ = ("unicode_source", "alternates", "bitmap_code", "split_unicode")

    unicode_source: str
    alternates: tuple[str, ...]
    bitmap_code: int
    split_unicode: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "unicode",
        "width_code",
        "unicode_source",
        "alternates",
        "bitmap_code",
        "split_unicode",
    )
    __match_args__ = (
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "unicode",
        "width_code",
        "unicode_source",
        "alternates",
        "bitmap_code",
        "split_unicode",
    )

    def __init__(
        self,
        code_bytes: bytes,
        char_code: int,
        cid: int,
        gid: int | None,
        unicode: str,
        width_code: int,
        unicode_source: str,
        alternates: tuple[str, ...],
        bitmap_code: int,
        split_unicode: bool = False,
    ) -> None:
        frozen_setattr(self, "code_bytes", code_bytes)
        frozen_setattr(self, "char_code", char_code)
        frozen_setattr(self, "cid", cid)
        frozen_setattr(self, "gid", gid)
        frozen_setattr(self, "unicode", unicode)
        frozen_setattr(self, "width_code", width_code)
        frozen_setattr(self, "unicode_source", unicode_source)
        frozen_setattr(self, "alternates", alternates)
        frozen_setattr(self, "bitmap_code", bitmap_code)
        frozen_setattr(self, "split_unicode", split_unicode)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code_bytes == other.code_bytes
            and self.char_code == other.char_code
            and self.cid == other.cid
            and self.gid == other.gid
            and self.unicode == other.unicode
            and self.width_code == other.width_code
            and self.unicode_source == other.unicode_source
            and self.alternates == other.alternates
            and self.bitmap_code == other.bitmap_code
            and self.split_unicode == other.split_unicode
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.code_bytes,
                self.char_code,
                self.cid,
                self.gid,
                self.unicode,
                self.width_code,
                self.unicode_source,
                self.alternates,
                self.bitmap_code,
                self.split_unicode,
            )
        )


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
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "source", source)
        frozen_setattr(self, "alternates", alternates)

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


SINGLE_BYTES = tuple(bytes((value,)) for value in range(256))


def split_code_bytes(data: bytes, cmap: CMapDecoder | ToUnicodeCMap | None) -> list[bytes]:
    if not data:
        return []
    if cmap is None:
        return [SINGLE_BYTES[byte] for byte in data]
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
            chunk = SINGLE_BYTES[data[pos]] if length == 1 else data[pos : pos + length]
            if ranges and not code_in_ranges(chunk, ranges):
                continue
            chunks.append(chunk)
            pos += length
            matched = True
            break
        if not matched:
            chunks.append(SINGLE_BYTES[data[pos]])
            pos += 1
    return chunks


def font_is_vertical(
    font: dict[str, Any],
    subtype: str | None,
    base_encoding: str | None,
    base_font_name: str | None,
    cmap: CMapDecoder | None,
) -> bool:
    if (
        base_encoding == "V"
        or (base_encoding and base_encoding.endswith("-V"))
        or (base_font_name and base_font_name.endswith("-V"))
        or (cmap is not None and cmap.wmode == 1)
    ):
        return True
    descendant = get_descendant(font) if subtype == "Type0" else None
    if descendant is None:
        return False
    wmode = descendant.get("WMode")
    if wmode is None:
        wmode = font.get("WMode", 0)
    try:
        return parse_int_strict(wmode, "invalid font WMode") == 1
    except ValueError:
        return False


class GlyphOutlineArrays:
    __slots__ = ("linear", "spans", "xs", "ys")

    def __init__(
        self,
        xs: numpy.ndarray[Any, Any],
        ys: numpy.ndarray[Any, Any],
        spans: tuple[tuple[int, int], ...],
    ) -> None:
        self.xs = xs
        self.ys = ys
        self.spans = spans
        self.linear: dict[
            tuple[float, float, float, float],
            tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]],
        ] = {}

    def linear_columns(
        self, a: float, b: float, c: float, d: float
    ) -> tuple[numpy.ndarray[Any, Any], numpy.ndarray[Any, Any]]:
        key = (a, b, c, d)
        columns = self.linear.get(key)
        if columns is None:
            if len(self.linear) >= LINEAR_CACHE_LIMIT:
                self.linear.clear()
            xs = self.xs
            ys = self.ys
            columns = self.linear[key] = (xs * a + ys * c, xs * b + ys * d)
        return columns


LINEAR_CACHE_LIMIT = 256


def outline_arrays(
    contours: tuple[tuple[tuple[float, float], ...], ...],
) -> GlyphOutlineArrays | None:
    xs: list[float] = []
    ys: list[float] = []
    spans: list[tuple[int, int]] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        start = len(xs)
        for x, y in contour:
            xs.append(x)
            ys.append(y)
        spans.append((start, len(xs)))
    if not spans:
        return None
    return GlyphOutlineArrays(
        numpy.asarray(xs, dtype=numpy.float64), numpy.asarray(ys, dtype=numpy.float64), tuple(spans)
    )


class FontDecoder:
    __slots__ = (
        "font",
        "semantic_context",
        "ligature_overrides",
        "to_unicode",
        "cmap",
        "cid_registry",
        "cid_ordering",
        "base_encoding",
        "differences",
        "encoding_differences",
        "simple_encoding_glyph_names",
        "encoding_decode_table",
        "is_cid_font",
        "is_type3",
        "byte_decode_table",
        "widths",
        "default_width",
        "width_fallback_value",
        "space_width_fallback",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
        "is_vertical",
        "ascent",
        "descent",
        "font_name",
        "glyph_decode_table",
        "glyph_decode_table_authoritative",
        "cff_unicode_repair_index",
        "cff_unicode_repairs",
        "font_program",
        "raster_font_provider",
        "glyph_bbox_cache",
        "glyph_outline_cache",
        "glyph_outline_array_cache",
        "glyph_id_cache",
        "unicode_choice_cache",
    )

    font: dict[str, Any]
    semantic_context: SemanticContext | None
    ligature_overrides: dict[int, str]
    to_unicode: ToUnicodeCMap | None
    cmap: CMapDecoder | None
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
    widths: Mapping[int, float]
    default_width: float
    width_fallback_value: float
    space_width_fallback: float
    default_vertical_displacement_y: float
    default_vertical_origin_y: float
    vertical_metrics: dict[int, tuple[float, float, float]]
    is_vertical: bool
    ascent: float
    descent: float
    font_name: str | None
    glyph_decode_table: tuple[str, ...] | None
    glyph_decode_table_authoritative: bool
    cff_unicode_repair_index: CFFUnicodeRepairIndex | None
    cff_unicode_repairs: dict[bytes, str]
    font_program: FontProgram | None
    raster_font_provider: RasterFontProviderLike | None
    glyph_bbox_cache: dict[int, Rectangle | None]
    glyph_outline_cache: dict[
        tuple[int, int | None, str], tuple[tuple[tuple[float, float], ...], ...]
    ]
    glyph_outline_array_cache: dict[tuple[int, int | None, str], GlyphOutlineArrays | None]
    glyph_id_cache: dict[int, int | None]
    unicode_choice_cache: dict[tuple[bytes, int, int | None], UnicodeChoice]

    __fields__: ClassVar[tuple[str, ...]] = (
        "font",
        "semantic_context",
        "ligature_overrides",
        "to_unicode",
        "cmap",
        "cid_registry",
        "cid_ordering",
        "base_encoding",
        "differences",
        "encoding_differences",
        "simple_encoding_glyph_names",
        "encoding_decode_table",
        "is_cid_font",
        "is_type3",
        "byte_decode_table",
        "widths",
        "default_width",
        "width_fallback_value",
        "space_width_fallback",
        "default_vertical_displacement_y",
        "default_vertical_origin_y",
        "vertical_metrics",
        "is_vertical",
        "ascent",
        "descent",
        "font_name",
        "glyph_decode_table",
        "glyph_decode_table_authoritative",
        "cff_unicode_repair_index",
        "cff_unicode_repairs",
        "font_program",
        "raster_font_provider",
        "glyph_bbox_cache",
        "glyph_outline_cache",
        "glyph_outline_array_cache",
        "glyph_id_cache",
        "unicode_choice_cache",
    )

    def __init__(
        self,
        font: dict[str, Any],
        ligature_overrides: dict[int, str] | None = None,
        raster_font_provider: RasterFontProviderLike | None = None,
        *,
        semantic_context: SemanticContext | None = None,
    ) -> None:
        self.font = font
        self.semantic_context = semantic_context
        self.ligature_overrides = ligature_overrides if ligature_overrides is not None else {}
        self.raster_font_provider = raster_font_provider
        self.initialize()

    def initialize(self) -> None:
        self.glyph_bbox_cache = {}
        self.glyph_outline_cache = {}
        self.glyph_outline_array_cache = {}
        self.glyph_id_cache = {}
        self.unicode_choice_cache = {}
        font = self.font
        subtype = font.get("Subtype")
        if subtype is not None:
            subtype = recover_pdf_name(subtype)

        self.font_program = font_program_for_pdf_font(font)

        to_unicode_obj = font.get("ToUnicode")
        to_unicode = None
        if isinstance(to_unicode_obj, PdfStream):
            try:
                to_unicode = ToUnicodeCMap(
                    to_unicode_obj.data,
                    usecmap_resolver=resolve_cmap_resource,
                )
            except PdfParseError, ValueError:
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
        default_width_explicit = font_metrics.default_width_explicit
        is_cid_font = subtype == "Type0" and get_descendant(font) is not None

        base_font_name = resolve_base_font_name(font, subtype)
        is_vertical = font_is_vertical(font, subtype, base_encoding, base_font_name, cmap)

        ascent, descent = parse_font_metrics(font, subtype, base_font_name, widths)

        is_type3 = subtype == "Type3"
        if is_type3:
            widths = adjust_type3_widths(font, widths)

        simple_encoding_glyph_names = build_simple_encoding_glyph_names(
            base_encoding if base_encoding in BASE_ENCODING_GLYPH_NAMES else "StandardEncoding",
            builtin_encoding,
            differences,
            authoritative_builtin=builtin_encoding_authoritative,
            context=self.semantic_context,
        )
        if builtin_encoding_authoritative:
            encoding_decode_table = tuple(
                unicode_for_glyph_name(name) or "" for name in simple_encoding_glyph_names
            )
        else:
            key = base_encoding or ("Type3" if is_type3 else "")
            encoding_decode_table = build_decode_table(
                key, differences, context=self.semantic_context
            )

        byte_decode_table: tuple[str, ...] | None = None
        if to_unicode is None and not is_cid_font:
            byte_decode_table = encoding_decode_table

        if not widths and not is_cid_font and not is_type3:
            builtin = standard_14_widths(base_font_name, encoding_decode_table)
            if builtin is not None:
                widths = builtin
                if font.get("MissingWidth") is None:
                    default_width = 0.0
                    default_width_explicit = True

        self.to_unicode = to_unicode
        self.cmap = cmap
        self.cid_registry, self.cid_ordering = self.cid_system_info(font)
        self.base_encoding = base_encoding
        self.differences = differences
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
        if default_width_explicit:
            self.width_fallback_value = default_width
            self.space_width_fallback = default_width
        else:
            self.width_fallback_value = default_width if default_width > 0.0 else 1000.0
            self.space_width_fallback = default_width if default_width > 0.0 else 250.0
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
        self.cff_unicode_repair_index = build_cff_unicode_repair_index(
            font, self.font_program, to_unicode, cmap
        )
        self.cff_unicode_repairs = {}

    @staticmethod
    def cid_system_info_string(value: object) -> str | None:
        if isinstance(value, PdfString):
            return value.data.decode("latin-1")
        normalized = recover_pdf_name(value)
        if normalized is not None:
            return normalized
        if isinstance(value, bytes):
            return value.decode("latin-1")
        return None

    @classmethod
    def cid_system_info(
        cls,
        font: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        descendant = get_descendant(font)
        system_info = descendant.get("CIDSystemInfo") if descendant else None
        if not isinstance(system_info, dict):
            system_info = font.get("CIDSystemInfo")
        if not isinstance(system_info, dict):
            return None, None
        registry = cls.cid_system_info_string(system_info.get("Registry"))
        ordering = cls.cid_system_info_string(system_info.get("Ordering"))
        return registry, ordering

    def parse_encoding(
        self, font: dict[str, Any]
    ) -> tuple[CMapDecoder | None, str | None, dict[int, str], dict[int, str], bool]:
        cmap = None
        base_encoding = None
        base_encoding_explicit = False
        differences: dict[int, str] = {}
        subtype = recover_pdf_name(font.get("Subtype"))
        encoding_obj = font.get("Encoding")
        match encoding_obj:
            case PdfStream():
                try:
                    cmap = CMapDecoder(
                        encoding_obj.data,
                        usecmap_resolver=resolve_cmap_resource,
                    )
                except PdfParseError, ValueError:
                    cmap = None
            case dict():
                base_encoding = recover_pdf_name(encoding_obj.get("BaseEncoding"))
                if base_encoding is None:
                    base_encoding = (
                        "WinAnsiEncoding" if subtype == "TrueType" else "StandardEncoding"
                    )
                else:
                    base_encoding_explicit = True
                differences_obj = encoding_obj.get("Differences")
                if differences_obj is not None and not isinstance(differences_obj, (list, tuple)):
                    differences_obj = None
                differences = parse_differences(
                    list(differences_obj)
                    if isinstance(differences_obj, tuple)
                    else differences_obj,
                    recover_pdf_name,
                )
            case _:
                base_encoding = recover_pdf_name(encoding_obj)
                base_encoding_explicit = base_encoding is not None
                cmap = self.named_cmap(base_encoding)
        if base_encoding is None and subtype == "Type3":
            base_encoding = "StandardEncoding"
        builtin: dict[int, str] = {}
        builtin_authoritative = False
        if subtype in ("Type1", "MMType1") and not base_encoding_explicit:
            builtin, builtin_authoritative = self.builtin_font_encoding(font)
        if base_encoding is None and subtype in ("Type1", "MMType1"):
            base_encoding = "StandardEncoding"
        return cmap, base_encoding, differences, builtin, builtin_authoritative

    def builtin_font_encoding(self, font: dict[str, Any]) -> tuple[dict[int, str], bool]:
        match self.font_program:
            case CFFFont() as program:
                try:
                    return (
                        program.builtin_encoding(),
                        program.builtin_encoding_is_authoritative(),
                    )
                except PdfParseError, ValueError:
                    return {}, False
            case _:
                pass
        descriptor = font.get("FontDescriptor")
        if not isinstance(descriptor, dict):
            return {}, False
        font_file = descriptor.get("FontFile")
        if isinstance(font_file, PdfStream):
            try:
                encoding = parse_type1_font_program_encoding(font_file.data)
                return encoding, bool(encoding)
            except PdfParseError, ValueError:
                return {}, False
        return {}, False

    def named_cmap(self, base_encoding: str | None) -> CMapDecoder | None:
        if base_encoding is None:
            return None
        return resolve_cmap_decoder(base_encoding)

    def decode(self, data: bytes) -> str:
        if not data:
            return ""
        table = self.byte_decode_table
        if (
            table is not None
            and not self.is_cid_font
            and self.to_unicode is None
            and self.glyph_decode_table is None
            and not self.ligature_overrides
        ):
            return "".join(table[byte] for byte in data)
        return "".join(glyph.unicode for glyph in self.decode_glyphs(data))

    def decode_glyphs(self, data: bytes | bytearray | memoryview) -> tuple[DecodedGlyph, ...]:
        if not data:
            return ()
        if self.is_cid_font:
            glyphs = self.decode_cid_glyphs(bytes(data))
        else:
            glyphs = self.decode_simple_glyphs(data)
        return tuple(glyphs)

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

        if to_unicode_text is not None and not has_invalid_unicode_mapping(to_unicode_text):
            visual_punctuation = self.visual_punctuation_for_code(
                to_unicode_text,
                fallback_code=fallback_code,
            )
            if visual_punctuation is not None:
                return UnicodeChoice(
                    visual_punctuation,
                    UnicodeSource.TRUETYPE_GLYPH_SHAPE,
                    dedupe_alternates(alternates, visual_punctuation),
                )
            return UnicodeChoice(
                to_unicode_text,
                UnicodeSource.TO_UNICODE,
                dedupe_alternates(alternates, to_unicode_text),
            )

        replacement = self.cff_unicode_repairs.get(code_bytes)
        if replacement is not None:
            return UnicodeChoice(
                replacement,
                UnicodeSource.CFF_GLYPH_REPAIR,
                dedupe_alternates(alternates, replacement),
            )

        if gid is not None:
            tt_text = self.true_type_unicode_for_gid(gid)
            if tt_text == to_unicode_text == "\ufffd":
                return UnicodeChoice(to_unicode_text, UnicodeSource.TO_UNICODE)
            if tt_text and not has_untrusted_unicode_semantics(tt_text):
                return UnicodeChoice(
                    tt_text,
                    UnicodeSource.TRUETYPE_CMAP,
                    dedupe_alternates(alternates, tt_text),
                )

        if fallback_code != 0:
            predefined_text = predefined_cmap_unicode(self.base_encoding, code_bytes)
            if predefined_text is not None:
                return UnicodeChoice(
                    predefined_text,
                    UnicodeSource.PREDEFINED_CMAP,
                    dedupe_alternates(alternates, predefined_text),
                )

        if not self.is_cid_font and len(code_bytes) == 1 and code_bytes[0] not in self.differences:
            encoding_table = self.encoding_decode_table
            encoding_text = encoding_table[code_bytes[0]]
            if encoding_text and not has_invalid_unicode_mapping(encoding_text):
                return UnicodeChoice(
                    encoding_text,
                    UnicodeSource.ENCODING,
                    dedupe_alternates(alternates, encoding_text),
                )

        registry = self.cid_registry
        ordering = self.cid_ordering
        cid_unicode_map = (
            resolve_cid_unicode_map(registry, ordering, vertical=self.is_vertical)
            if registry is not None and ordering is not None
            else None
        )
        if cid_unicode_map is not None:
            cid_text = cid_unicode_map.get(fallback_code)
            if cid_text is not None:
                return UnicodeChoice(
                    cid_text,
                    UnicodeSource.CID_COLLECTION,
                    dedupe_alternates(alternates, cid_text),
                )

        if to_unicode_text is not None and "\ufffd" in to_unicode_text:
            return UnicodeChoice(to_unicode_text, UnicodeSource.TO_UNICODE)

        if fallback_code == 0:
            return UnicodeChoice(
                "\u0000", UnicodeSource.FALLBACK_NUL, dedupe_alternates(alternates, "\u0000")
            )
        text = unicode_scalar_or_replacement(fallback_code)
        source = UnicodeSource.IDENTITY if text != "\ufffd" else UnicodeSource.REPLACEMENT
        return UnicodeChoice(text, source, dedupe_alternates(alternates, text))

    def true_type_unicode_for_gid(self, gid: int) -> str:
        match self.font_program:
            case TrueTypeFontProgram() as program:
                return program.unicode_for_gid(gid)
            case _:
                return ""

    def visual_punctuation_for_code(self, text: str, *, fallback_code: int) -> str | None:
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

    def apply_simple_unicode_overrides(
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
                        choice.source in UNRESOLVED_UNICODE_SOURCES
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

    def decode_simple_glyphs(self, data: bytes | bytearray | memoryview) -> list[DecodedGlyph]:
        glyphs: list[DecodedGlyph] = []
        table = self.byte_decode_table
        if table is None and self.to_unicode is None:
            table = self.encoding_decode_table
        for code in data:
            chunk = SINGLE_BYTES[code]
            gid = self.glyph_id_for_code(code)
            if self.to_unicode is not None:
                choice = self.unicode_choice_for_code(chunk, code, gid)
            elif table is not None:
                text = table[code]
                undefined = not text or (
                    code in self.differences and len(text) == 1 and ord(text) < 32
                )
                choice = UnicodeChoice(
                    "\ufffd" if undefined else text,
                    UnicodeSource.UNDEFINED if undefined else UnicodeSource.ENCODING,
                )
            else:
                choice = self.unicode_choice_for_code(chunk, code, gid)
            choice = self.apply_simple_unicode_overrides(choice, chunk)
            glyphs.append(
                DecodedGlyph(
                    chunk,
                    code,
                    code,
                    gid,
                    choice.text,
                    code,
                    choice.source,
                    choice.alternates,
                    code,
                    choice.text in LEGITIMATE_MULTI_CHAR_GLYPHS,
                )
            )
        return glyphs

    def decode_cid_glyphs(self, data: bytes) -> list[DecodedGlyph]:
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
                current = self.cff_unicode_repairs
                changed = {code for code, text in repairs.items() if current.get(code) != text}
                if changed:
                    cache = self.unicode_choice_cache
                    for key in [key for key in cache if key[0] in changed]:
                        del cache[key]
                    current.update(repairs)
        return [self.build_cid_glyph(code_bytes, cid) for code_bytes, cid in entries]

    def build_cid_glyph(self, code_bytes: bytes, cid: int) -> DecodedGlyph:
        char_code = int.from_bytes(code_bytes, "big") if code_bytes else 0
        gid = self.glyph_id_for_code(cid)
        if gid is not None and gid != 0 and not self.glyph_exists(gid):
            notdef_cid = self.cmap.mapped_notdef(code_bytes) if self.cmap else None
            if notdef_cid is not None:
                cid = notdef_cid
                gid = self.glyph_id_for_code(cid)
        choice = self.unicode_choice_for_code(code_bytes, cid, gid)
        if self.ligature_overrides:
            lo = self.ligature_overrides
            text = "".join(lo.get(ord(ch), ch) for ch in choice.text)
            if text != choice.text:
                choice = UnicodeChoice(
                    text,
                    UnicodeSource.LIGATURE_OVERRIDE,
                    dedupe_alternates((choice.text, *choice.alternates), text),
                )
        return DecodedGlyph(
            code_bytes,
            char_code,
            cid,
            gid,
            choice.text,
            cid,
            choice.source,
            choice.alternates,
            cid,
            choice.text in LEGITIMATE_MULTI_CHAR_GLYPHS,
        )

    def glyph_exists(self, gid: int) -> bool:
        program = self.font_program
        return program.has_glyph_id(gid) if program is not None else True

    def glyph_id_for_code(self, code: int) -> int | None:
        cache = self.glyph_id_cache
        try:
            return cache[code]
        except KeyError:
            gid = cache[code] = self.resolve_glyph_id_for_code(code)
            return gid

    def resolve_glyph_id_for_code(self, code: int) -> int | None:
        match self.font_program:
            case CFFFont() as program:
                if self.is_cid_font:
                    return program.glyph_id_for_cid(code)
                return program.glyph_id_for_name(self.glyph_name(code))
            case TrueTypeFontProgram() as program:
                if not self.is_cid_font and 0 <= code < 256 and program.cmap:
                    glyph_text = glyph_name_to_unicode(self.glyph_name(code))
                    if len(glyph_text) == 1:
                        return program.glyph_id_for_unicode(ord(glyph_text))
                return program.glyph_id_for_code(code)
            case Type1FontProgram() as program:
                return program.glyph_id_for_name(self.glyph_name(code))
            case OpenTypeFontProgram() as program:
                if self.is_cid_font:
                    return code
                return program.glyph_id_for_name(self.glyph_name(code))
            case _:
                return code

    @property
    def font_matrix(self) -> Matrix:
        try:
            return Matrix.from_operand(self.font.get("FontMatrix"))
        except ValueError:
            return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)

    def glyph_name(self, code: int) -> str:
        if 0 <= code < 256:
            return self.simple_encoding_glyph_names[code]
        return ".notdef"

    def glyph_bbox(self, code: int) -> Rectangle | None:
        cache = self.glyph_bbox_cache
        try:
            return cache[code]
        except KeyError:
            box = cache[code] = self.glyph_bbox_uncached(code)
            return box

    def glyph_bbox_uncached(self, code: int) -> Rectangle | None:
        if code < 0:
            return None
        program = self.font_program
        if program is None:
            width = self.glyph_width(code)
            return None if width <= 0 else (0.0, self.descent, width, self.ascent)
        glyph_id = self.glyph_id_for_code(code)
        return program.glyph_bbox_for_gid(glyph_id) if glyph_id is not None else None

    def vertical_glyph_metric(self, code: int) -> tuple[float, float, float]:
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
        glyph_id = self.glyph_id_for_code(code)
        program = self.font_program
        return (
            program.glyph_bitmap_for_gid(glyph_id, width=width, height=height)
            if program is not None and glyph_id is not None
            else ()
        )

    def glyph_outline(
        self, code: int, gid: int | None = None, text: str = ""
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        if code < 0:
            return ()
        cache = self.glyph_outline_cache
        key = (code, gid, text)
        contours = cache.get(key)
        if contours is None:
            contours = cache[key] = self.glyph_outline_uncached(code, gid, text)
        return contours

    def glyph_outline_arrays(
        self, code: int, gid: int | None = None, text: str = ""
    ) -> GlyphOutlineArrays | None:
        if code < 0:
            return None
        cache = self.glyph_outline_array_cache
        key = (code, gid, text)
        try:
            return cache[key]
        except KeyError:
            arrays = cache[key] = outline_arrays(self.glyph_outline(code, gid, text))
            return arrays

    def glyph_outline_uncached(
        self, code: int, gid: int | None, text: str
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        glyph_id = gid if gid is not None else self.glyph_id_for_code(code)
        if glyph_id is None:
            return ()
        program = self.font_program
        if program is not None:
            return program.normalized_glyph_contours(glyph_id)
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
        fallback = self.space_width_fallback if code == 32 else self.width_fallback_value
        return self.widths.get(code, fallback)

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
        width = self.vertical_glyph_metric(code)[0] if self.is_vertical else self.glyph_width(code)
        return pdf_glyph_advance_vector(
            width,
            vertical=self.is_vertical,
            font_size=font_size,
            char_space=char_space,
            word_space=word_space,
            horizontal_scale=horizontal_scale,
            encoded_space=encoded_space,
        )

    def text_advance_vector(
        self,
        data: bytes | bytearray | memoryview,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        glyphs: tuple[DecodedFontGlyph, ...] | None = None,
    ) -> tuple[float, float]:
        if not data:
            return (0.0, 0.0)
        if glyphs is None:
            glyphs = self.decode_glyphs(bytes(data))

        if self.is_vertical:
            total_y = 0.0
            vertical_glyph_metric = self.vertical_glyph_metric
            for glyph in glyphs:
                spacing = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
                total_y += vertical_glyph_metric(glyph.width_code)[0] * font_size / 1000.0 + spacing
            return (0.0, total_y)

        total_x = 0.0
        width_fallback = self.width_fallback_value
        space_fallback = self.space_width_fallback
        width_for = self.widths.get
        for glyph in glyphs:
            code = glyph.width_code
            fallback = space_fallback if code == 32 else width_fallback
            spacing = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
            displacement_x = width_for(code, fallback) * font_size / 1000.0 + spacing
            total_x += displacement_x * horizontal_scale / 100.0
        return (total_x, 0.0)


def dedupe_alternates(values: Iterable[str], selected: str) -> tuple[str, ...]:
    seen = {selected}
    alternates: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        alternates.append(value)
    return tuple(alternates)


frozen_setattr = object.__setattr__


class CompactCMap(Record):
    __slots__ = ("effective_codes_by_cid",)

    effective_codes_by_cid: dict[int, tuple[bytes, ...]]

    __fields__: ClassVar[tuple[str, ...]] = ("effective_codes_by_cid",)
    __match_args__ = ("effective_codes_by_cid",)

    def __init__(self, effective_codes_by_cid: dict[int, tuple[bytes, ...]]) -> None:
        frozen_setattr(self, "effective_codes_by_cid", effective_codes_by_cid)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.effective_codes_by_cid == other.effective_codes_by_cid

    def __hash__(self) -> int:
        return hash((self.effective_codes_by_cid,))

    def codes_for_cid(self, cid: int) -> tuple[bytes, ...]:
        return self.effective_codes_by_cid.get(cid, ())


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
    return CompactCMap(
        {cid: tuple(codes) for cid, codes in effective_codes_by_cid.items()},
    )


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
        for code in cmap.codes_for_cid(cid)
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


TEX_MATH_GLYPH_OVERRIDES: dict[str, dict[str, str]] = {
    "TeX_Times_Math_Italic": {
        "C14": "δ",
    },
    "TeX_Times_Math_Symbol": {
        "C14": "°",
    },
}
COMPUTER_MODERN_MATH_PREFIXES = (
    "CMEX",
    "CMMI",
    "CMMIB",
    "CMSY",
)
LIGATURE_GLYPH_TEXT = {
    "ff": "ff",
    "fi": "fi",
    "fl": "fl",
    "ffi": "ffi",
    "ffl": "ffl",
    "f_f": "ff",
    "f_i": "fi",
    "f_l": "fl",
    "f_f_i": "ffi",
    "f_f_l": "ffl",
}


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


def normalized_base_font_name(base_font_name: str | None) -> str | None:
    if base_font_name is None:
        return None
    return strip_subset_tag(base_font_name)


def build_glyph_decode_table(
    base_font_name: str | None, differences: dict[int, str]
) -> tuple[tuple[str, ...], bool] | None:
    normalized = normalized_base_font_name(base_font_name)
    if normalized is None:
        return None
    overrides = TEX_MATH_GLYPH_OVERRIDES.get(normalized, {})
    is_computer_modern_math = normalized.startswith(COMPUTER_MODERN_MATH_PREFIXES)
    if not overrides and not is_computer_modern_math and not differences:
        return None
    table: list[str | None] = [None] * 256
    has_mapping = False
    for code, glyph_name in differences.items():
        mapped = overrides.get(glyph_name)
        if mapped is None:
            mapped = LIGATURE_GLYPH_TEXT.get(glyph_name)
        if mapped is None:
            mapped = glyph_name_to_unicode(glyph_name)
            if mapped == glyph_name:
                mapped = None
        if mapped is not None and 0 <= code <= 255:
            table[code] = mapped
            has_mapping = True
    if not has_mapping:
        return None
    return tuple(ch or "" for ch in table), bool(overrides or is_computer_modern_math)


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
