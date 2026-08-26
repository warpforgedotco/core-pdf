"""OpenType CFF/CFF2 outline access through the vendored fontTools subset."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from core_pdf._vendor.fontTools.pens.boundsPen import BoundsPen
from core_pdf._vendor.fontTools.pens.recordingPen import DecomposingRecordingPen
from core_pdf._vendor.fontTools.pens.transformPen import TransformPen
from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.engine.spec.s_07_syntax.coercion import normalize_pdf_name
from core_pdf.impl.engine.spec.s_07_syntax.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.font_program_truetype import (
    FONT_PROGRAM_ERRORS,
    internal_recording_to_contours,
)
from core_pdf.impl.engine.spec.s_09_fonts.raster_kernel import Point, rasterize_contours
from core_pdf.impl.engine.spec.s_09_fonts.widths import get_descendant
from core_pdf.impl.objects import PdfStream


class OpenTypeFontProgram:
    """Expose normalized outlines from an OpenType CFF or CFF2 wrapper."""

    __slots__ = (
        "font",
        "glyph_count",
        "internal_contour_cache",
        "internal_glyph_set",
        "reverse_glyph_map",
        "units_per_em",
    )

    def __init__(self, data: bytes) -> None:
        try:
            self.font = TTFont(
                BytesIO(data),
                lazy=True,
                recalcBBoxes=False,
                recalcTimestamp=False,
            )
            if not ({"CFF ", "CFF2"} & set(self.font.keys())):
                raise ValueError("OpenType font has no CFF outline table")
            self.reverse_glyph_map = self.font.getReverseGlyphMap()
            self.glyph_count = len(self.font.getGlyphOrder())
            self.units_per_em = float(getattr(self.font["head"], "unitsPerEm", 1000) or 1000)
            # The renderer resolves variable CFF2 fonts at their default
            # instance. A default glyph set does not need axis metadata, and
            # dropping the lazily registered table avoids loading variation
            # machinery that is intentionally absent from the vendored subset.
            if "CFF2" in self.font and "fvar" in self.font:
                del self.font["fvar"]
        except FONT_PROGRAM_ERRORS as exc:
            raise ValueError("invalid OpenType CFF font program") from exc
        self.internal_glyph_set: Any | None = None
        self.internal_contour_cache: dict[int, tuple[tuple[Point, ...], ...]] = {}

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.reverse_glyph_map.get(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        cached = self.internal_contour_cache.get(glyph_id)
        if cached is not None:
            return cached
        try:
            glyph_name = self.font.getGlyphName(glyph_id)
            glyph_set = self.internal_glyph_set
            if glyph_set is None:
                glyph_set = self.font.getGlyphSet()
                self.internal_glyph_set = glyph_set
            pen = DecomposingRecordingPen(glyph_set, skipMissingComponents=True)
            glyph_set[glyph_name].draw(pen)
            contours = internal_recording_to_contours(pen.value)
            scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
            result = tuple(
                tuple((x * scale, y * scale) for x, y in contour) for contour in contours
            )
        except FONT_PROGRAM_ERRORS:
            result = ()
        if len(self.internal_contour_cache) >= 512:
            self.internal_contour_cache.clear()
        self.internal_contour_cache[glyph_id] = result
        return result

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        try:
            glyph_name = self.font.getGlyphName(glyph_id)
            glyph_set = self.internal_glyph_set
            if glyph_set is None:
                glyph_set = self.font.getGlyphSet()
                self.internal_glyph_set = glyph_set
            bounds_pen = BoundsPen(glyph_set)
            scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
            normalized_pen = TransformPen(bounds_pen, (scale, 0.0, 0.0, scale, 0.0, 0.0))
            glyph_set[glyph_name].draw(normalized_pen)
        except FONT_PROGRAM_ERRORS:
            return None
        bounds = bounds_pen.bounds
        if bounds is None:
            return None
        x_min, y_min, x_max, y_max = bounds
        return (float(x_min), float(y_min), float(x_max), float(y_max))

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        contours = self.normalized_glyph_contours(glyph_id)
        return rasterize_contours(contours, width=width, height=height) if contours else ()


def opentype_font_for_pdf_font(font: dict[str, object]) -> OpenTypeFontProgram | None:
    descendant = get_descendant(font)
    font_dict = descendant if descendant is not None else font
    descriptor = lookup_dict_key(font_dict, "FontDescriptor")
    if not isinstance(descriptor, dict):
        return None
    font_file = lookup_dict_key(descriptor, "FontFile3")
    if not isinstance(font_file, PdfStream):
        return None
    if normalize_pdf_name(lookup_dict_key(font_file.dictionary, "Subtype")) != "OpenType":
        return None
    try:
        return OpenTypeFontProgram(font_file.data)
    except ValueError:
        return None


__all__ = ["OpenTypeFontProgram", "opentype_font_for_pdf_font"]
