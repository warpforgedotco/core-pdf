# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any

from core_adobe_fonts.cmap.ranges import (
    code_in_ranges,
)
from core_pdf.impl.caches import BoundedDict
from core_pdf.impl.fonts_cmap_tokenizer import CMapDecoder
from core_pdf.impl.fonts_cmap_tounicode import ToUnicodeCMap
from core_pdf.impl.fonts_encoding import FontEncoding
from core_pdf.impl.fonts_glyph_geometry import GlyphGeometry, GlyphOutlineArrays
from core_pdf.impl.fonts_helpers import LEGITIMATE_MULTI_CHAR_GLYPHS
from core_pdf.impl.fonts_metrics import FontMetricsModel
from core_pdf.impl.fonts_program import load_glyph_program
from core_pdf.impl.fonts_unicode import (
    UnicodeChoice,
    UnicodeResolver,
    build_cff_unicode_repair_index,
    drop_code_entries,
    parse_to_unicode,
)
from core_pdf.impl.fonts_widths import effective_descriptor, recover_descendant
from core_pdf.impl.glyphs import UnicodeSource
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.types import RecordType, Rectangle
from core_pdf_spec.s_08_graphics.matrix import Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph
from core_pdf_spec.standards import SemanticContext

if typing.TYPE_CHECKING:
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike


def descriptor_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    descriptor = effective_descriptor(font, subtype)
    if not isinstance(descriptor, dict):
        return None
    return recover_pdf_name(descriptor.get("FontName"))


def resolve_base_font_name(font: dict[str, Any], subtype: str | None) -> str | None:
    base_font_name = recover_pdf_name(font.get("BaseFont"))
    if base_font_name is not None:
        return base_font_name
    return descriptor_font_name(font, subtype)


STRING_GLYPH_CACHE_MAX_BYTES = 8
STRING_GLYPH_CACHE_MAX_ENTRIES = 8192


class DecodedGlyph(DecodedFontGlyph, metaclass=RecordType):
    unicode_source: str
    alternates: tuple[str, ...]
    bitmap_code: int
    split_unicode: bool = False


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


class FontDecoder:
    __slots__ = (
        "font",
        "semantic_context",
        "ligature_overrides",
        "raster_font_provider",
        "is_cid_font",
        "is_type3",
        "font_name",
        "is_vertical",
        "ascent",
        "descent",
        "encoding",
        "metrics",
        "unicode",
        "geometry",
        "byte_decode_table",
        "simple_glyph_cache",
        "cid_glyph_cache",
        "string_glyph_cache",
        "glyph_width",
        "glyph_name",
        "glyph_id_for_code",
        "glyph_advance_vector",
        "glyph_bbox",
        "glyph_bitmap",
        "glyph_outline_arrays",
        "vertical_glyph_metric",
        "vertical_glyph_position",
    )

    font: dict[str, Any]
    semantic_context: SemanticContext | None
    ligature_overrides: dict[int, str]
    raster_font_provider: RasterFontProviderLike | None
    is_cid_font: bool
    is_type3: bool
    font_name: str | None
    is_vertical: bool
    ascent: float
    descent: float
    encoding: FontEncoding
    metrics: FontMetricsModel
    unicode: UnicodeResolver
    geometry: GlyphGeometry
    byte_decode_table: tuple[str, ...] | None
    simple_glyph_cache: dict[int, DecodedGlyph]
    cid_glyph_cache: dict[tuple[bytes, int], DecodedGlyph]
    string_glyph_cache: BoundedDict[bytes, tuple[DecodedGlyph, ...]]
    glyph_width: Callable[[int], float]
    glyph_name: Callable[..., str]
    glyph_id_for_code: Callable[[int], int | None]
    glyph_advance_vector: Callable[..., tuple[float, float]]
    glyph_bbox: Callable[[int], Rectangle | None]
    glyph_bitmap: Callable[..., tuple[int, ...]]
    glyph_outline_arrays: Callable[..., GlyphOutlineArrays | None]
    vertical_glyph_metric: Callable[[int], tuple[float, float, float]]
    vertical_glyph_position: Callable[..., tuple[float, float]]

    def __init__(
        self,
        font: dict[Any, Any],
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
        font = self.font
        subtype = font.get("Subtype")
        if subtype is not None:
            subtype = recover_pdf_name(subtype)
        program = load_glyph_program(font)
        to_unicode = parse_to_unicode(font)
        is_cid_font = subtype == "Type0" and recover_descendant(font) is not None
        is_type3 = subtype == "Type3"
        base_font_name = resolve_base_font_name(font, subtype)
        encoding = FontEncoding(
            font,
            program,
            is_cid_font=is_cid_font,
            is_type3=is_type3,
            base_font_name=base_font_name,
            semantic_context=self.semantic_context,
        )
        metrics = FontMetricsModel(
            font,
            subtype,
            base_encoding=encoding.base_encoding,
            cmap=encoding.cmap,
            encoding_decode_table=encoding.encoding_decode_table,
            base_font_name=base_font_name,
            is_cid_font=is_cid_font,
            is_type3=is_type3,
        )
        geometry = GlyphGeometry(
            program,
            encoding,
            metrics,
            font_name=base_font_name,
            raster_font_provider=self.raster_font_provider,
        )
        self.is_cid_font = is_cid_font
        self.is_type3 = is_type3
        self.font_name = base_font_name
        self.is_vertical = metrics.is_vertical
        self.ascent = metrics.ascent
        self.descent = metrics.descent
        self.encoding = encoding
        self.metrics = metrics
        self.geometry = geometry
        self.unicode = UnicodeResolver(
            to_unicode,
            encoding,
            program,
            is_cid_font=is_cid_font,
            is_vertical=metrics.is_vertical,
            ligature_overrides=self.ligature_overrides,
            cff_unicode_repair_index=build_cff_unicode_repair_index(
                font, program, to_unicode, encoding.cmap
            ),
        )
        self.byte_decode_table = (
            encoding.encoding_decode_table if to_unicode is None and not is_cid_font else None
        )
        self.simple_glyph_cache = {}
        self.cid_glyph_cache = {}
        self.string_glyph_cache = BoundedDict(STRING_GLYPH_CACHE_MAX_ENTRIES)
        self.glyph_width = metrics.glyph_width
        self.glyph_name = encoding.glyph_name
        self.glyph_id_for_code = geometry.glyph_id_for_code
        self.glyph_advance_vector = metrics.glyph_advance_vector
        self.vertical_glyph_metric = metrics.vertical_glyph_metric
        self.vertical_glyph_position = metrics.vertical_glyph_position
        self.glyph_bbox = geometry.glyph_bbox
        self.glyph_bitmap = geometry.glyph_bitmap
        self.glyph_outline_arrays = geometry.glyph_outline_arrays

    @property
    def font_matrix(self) -> Matrix:
        try:
            return Matrix.from_operand(self.font.get("FontMatrix"))
        except ValueError:
            return Matrix(0.001, 0.0, 0.0, 0.001, 0.0, 0.0)

    def decode(self, data: bytes) -> str:
        if not data:
            return ""
        table = self.byte_decode_table
        if (
            table is not None
            and not self.is_cid_font
            and self.unicode.to_unicode is None
            and self.encoding.glyph_decode_table is None
            and not self.ligature_overrides
        ):
            return "".join(table[byte] for byte in data)
        return "".join(glyph.unicode for glyph in self.decode_glyphs(data))

    def decode_glyphs(self, data: bytes | bytearray | memoryview) -> tuple[DecodedGlyph, ...]:
        if not data:
            return ()
        if len(data) > STRING_GLYPH_CACHE_MAX_BYTES:
            return self.decode_glyphs_uncached(data)
        key = bytes(data)
        cache = self.string_glyph_cache
        glyphs = cache.get(key)
        if glyphs is None:
            glyphs = cache.put(key, self.decode_glyphs_uncached(key))
        return glyphs

    def decode_glyphs_uncached(
        self, data: bytes | bytearray | memoryview
    ) -> tuple[DecodedGlyph, ...]:
        if self.is_cid_font:
            glyphs = self.decode_cid_glyphs(bytes(data))
        else:
            glyphs = self.decode_simple_glyphs(data)
        return tuple(glyphs)

    def decode_simple_glyphs(self, data: bytes | bytearray | memoryview) -> list[DecodedGlyph]:
        glyphs: list[DecodedGlyph] = []
        cache = self.simple_glyph_cache
        append = glyphs.append
        unicode = self.unicode
        to_unicode = unicode.to_unicode
        differences = self.encoding.differences
        glyph_id_for_code = self.geometry.glyph_id_for_code
        unicode_choice_for_code = unicode.unicode_choice_for_code
        apply_simple_unicode_overrides = unicode.apply_simple_unicode_overrides
        table = self.byte_decode_table
        if table is None and to_unicode is None:
            table = self.encoding.encoding_decode_table
        for code in data:
            cached = cache.get(code)
            if cached is not None:
                append(cached)
                continue
            chunk = SINGLE_BYTES[code]
            gid = glyph_id_for_code(code)
            if to_unicode is None and table is not None:
                text = table[code]
                undefined = not text or (code in differences and len(text) == 1 and ord(text) < 32)
                choice = UnicodeChoice(
                    "\ufffd" if undefined else text,
                    UnicodeSource.UNDEFINED if undefined else UnicodeSource.ENCODING,
                )
            else:
                choice = unicode_choice_for_code(chunk, code, gid)
            choice = apply_simple_unicode_overrides(choice, chunk)
            glyph = DecodedGlyph(
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
            cache[code] = glyph
            append(glyph)
        return glyphs

    def decode_cid_glyphs(self, data: bytes) -> list[DecodedGlyph]:
        cmap = self.encoding.cmap
        entries = cmap.decode_entries(data) if cmap is not None else []
        unicode = self.unicode
        if not entries:
            chunks = split_code_bytes(data, unicode.to_unicode)
            entries = [(chunk, int.from_bytes(chunk, "big") if chunk else 0) for chunk in chunks]
        if unicode.cff_unicode_repair_index is not None:
            changed = unicode.observe_codes(entries)
            if changed:
                drop_code_entries(self.cid_glyph_cache, changed)
                self.string_glyph_cache.clear()
        decoded_cache = self.cid_glyph_cache
        glyphs: list[DecodedGlyph] = []
        append = glyphs.append
        for code_bytes, cid in entries:
            decoded_key = (code_bytes, cid)
            decoded = decoded_cache.get(decoded_key)
            if decoded is None:
                decoded = decoded_cache[decoded_key] = self.build_cid_glyph(code_bytes, cid)
            append(decoded)
        return glyphs

    def build_cid_glyph(self, code_bytes: bytes, cid: int) -> DecodedGlyph:
        char_code = int.from_bytes(code_bytes, "big") if code_bytes else 0
        geometry = self.geometry
        gid = geometry.glyph_id_for_code(cid)
        if gid is not None and gid != 0 and not geometry.glyph_exists(gid):
            cmap = self.encoding.cmap
            notdef_cid = cmap.mapped_notdef(code_bytes) if cmap else None
            if notdef_cid is not None:
                cid = notdef_cid
                gid = geometry.glyph_id_for_code(cid)
        unicode = self.unicode
        choice = unicode.apply_ligature_overrides(
            unicode.unicode_choice_for_code(code_bytes, cid, gid)
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

        metrics = self.metrics
        if metrics.is_vertical:
            total_y = 0.0
            vertical_glyph_metric = metrics.vertical_glyph_metric
            for glyph in glyphs:
                spacing = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
                total_y += vertical_glyph_metric(glyph.width_code)[0] * font_size / 1000.0 + spacing
            return (0.0, total_y)

        total_x = 0.0
        default_width = metrics.default_width
        width_for = metrics.widths.get
        for glyph in glyphs:
            width = width_for(glyph.width_code, default_width)
            spacing = char_space + (word_space if glyph.code_bytes == b" " else 0.0)
            displacement_x = width * font_size / 1000.0 + spacing
            total_x += displacement_x * horizontal_scale / 100.0
        return (total_x, 0.0)
