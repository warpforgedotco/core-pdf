# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.fonts_encoding import FontEncoding
from core_pdf.impl.fonts_fallback import fallback_glyph_outline
from core_pdf.impl.fonts_metrics import FontMetricsModel
from core_pdf.impl.fonts_program_base import NULL_PROGRAM, GlyphProgram
from core_pdf.impl.fonts_raster_kernel import Contours
from core_pdf.impl.glyph_outlines import GlyphOutlineArrays, outline_arrays
from core_pdf.impl.types import Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.fonts_fallback import RasterFontProviderLike


class GlyphGeometry:
    __slots__ = (
        "program",
        "encoding",
        "metrics",
        "font_name",
        "raster_font_provider",
        "glyph_id_cache",
        "glyph_bbox_cache",
        "glyph_outline_array_cache",
    )

    program: GlyphProgram
    encoding: FontEncoding
    metrics: FontMetricsModel
    font_name: str | None
    raster_font_provider: RasterFontProviderLike | None
    glyph_id_cache: dict[int, int | None]
    glyph_bbox_cache: dict[int, Rectangle | None]
    glyph_outline_array_cache: dict[tuple[int, int | None, str], GlyphOutlineArrays | None]

    def __init__(
        self,
        program: GlyphProgram,
        encoding: FontEncoding,
        metrics: FontMetricsModel,
        *,
        font_name: str | None,
        raster_font_provider: RasterFontProviderLike | None,
    ) -> None:
        self.program = program
        self.encoding = encoding
        self.metrics = metrics
        self.font_name = font_name
        self.raster_font_provider = raster_font_provider
        self.glyph_id_cache = {}
        self.glyph_bbox_cache = {}
        self.glyph_outline_array_cache = {}

    def glyph_exists(self, gid: int) -> bool:
        return self.program.has_glyph_id(gid)

    def glyph_id_for_code(self, code: int) -> int | None:
        cache = self.glyph_id_cache
        try:
            return cache[code]
        except KeyError:
            gid = cache[code] = self.resolve_glyph_id_for_code(code)
            return gid

    def resolve_glyph_id_for_code(self, code: int) -> int | None:
        return self.program.glyph_id_for_code(code, self.encoding)

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
        program = self.program
        if program is NULL_PROGRAM:
            metrics = self.metrics
            width = metrics.glyph_width(code)
            return None if width <= 0 else (0.0, metrics.descent, width, metrics.ascent)
        glyph_id = self.glyph_id_for_code(code)
        return program.glyph_bbox_for_gid(glyph_id) if glyph_id is not None else None

    def glyph_bitmap(self, code: int, *, width: int = 24, height: int = 32) -> tuple[int, ...]:
        if code < 0:
            return ()
        glyph_id = self.glyph_id_for_code(code)
        return (
            self.program.glyph_bitmap_for_gid(glyph_id, width=width, height=height)
            if glyph_id is not None
            else ()
        )

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
            arrays = cache[key] = outline_arrays(self.glyph_outline_uncached(code, gid, text))
            return arrays

    def glyph_outline_uncached(self, code: int, gid: int | None, text: str) -> Contours:
        glyph_id = gid if gid is not None else self.glyph_id_for_code(code)
        if glyph_id is None:
            return ()
        program = self.program
        if program is NULL_PROGRAM:
            return self.fallback_glyph_outline(text)
        return program.normalized_glyph_contours(glyph_id)

    def fallback_glyph_outline(self, text: str) -> Contours:
        encoding = self.encoding
        return fallback_glyph_outline(
            self.font_name,
            text,
            is_cid_font=encoding.is_cid_font,
            is_vertical=self.metrics.is_vertical,
            cid_registry=encoding.cid_registry,
            cid_ordering=encoding.cid_ordering,
            provider=self.raster_font_provider,
        )
