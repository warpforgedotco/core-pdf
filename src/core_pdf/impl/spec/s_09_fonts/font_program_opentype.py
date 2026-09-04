"""OpenType CFF/CFF2 outline access through the vendored fontTools subset."""

from __future__ import annotations

from io import BytesIO

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
    FONT_PROGRAM_ERRORS,
    internal_fonttools_bbox,
    internal_fonttools_contours,
)
from core_pdf.impl.spec.s_09_fonts.raster_kernel import Point, rasterize_contours, scale_contours


class OpenTypeFontProgram:
    """Expose normalized outlines from an OpenType CFF or CFF2 wrapper."""

    __slots__ = (
        "font",
        "glyph_count",
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

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.reverse_glyph_map.get(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return 0 <= glyph_id < self.glyph_count

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        try:
            contours = internal_fonttools_contours(self.font, glyph_id)
            scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
            return scale_contours(contours, scale)
        except FONT_PROGRAM_ERRORS:
            return ()

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        try:
            scale = 1000.0 / self.units_per_em if self.units_per_em else 1.0
            return internal_fonttools_bbox(self.font, glyph_id, scale)
        except FONT_PROGRAM_ERRORS:
            return None

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        contours = self.normalized_glyph_contours(glyph_id)
        return rasterize_contours(contours, width=width, height=height) if contours else ()


__all__ = ["OpenTypeFontProgram"]
