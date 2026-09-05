"""OpenType CFF/CFF2 outline access through the vendored fontTools subset."""

from __future__ import annotations

from io import BytesIO

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl.spec.s_09_fonts.font_program_truetype import (
    FONT_PROGRAM_ERRORS,
    internal_FontToolsOutlineAccess,
)
from core_pdf.impl.spec.s_09_fonts.raster_kernel import Point


class OpenTypeFontProgram:
    """Expose normalized outlines from an OpenType CFF or CFF2 wrapper."""

    __slots__ = (
        "font",
        "outlines",
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
            self.outlines = internal_FontToolsOutlineAccess(self.font)
            # The renderer resolves variable CFF2 fonts at their default
            # instance. A default glyph set does not need axis metadata, and
            # dropping the lazily registered table avoids loading variation
            # machinery that is intentionally absent from the vendored subset.
            if "CFF2" in self.font and "fvar" in self.font:
                del self.font["fvar"]
        except FONT_PROGRAM_ERRORS as exc:
            raise ValueError("invalid OpenType CFF font program") from exc

    def glyph_id_for_name(self, glyph_name: str) -> int | None:
        return self.outlines.glyph_id_for_name(glyph_name)

    def has_glyph_id(self, glyph_id: int) -> bool:
        return self.outlines.has_glyph_id(glyph_id)

    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]:
        return self.outlines.normalized_glyph_contours(glyph_id)

    def glyph_bbox_for_gid(self, glyph_id: int) -> tuple[float, float, float, float] | None:
        return self.outlines.glyph_bbox_for_gid(glyph_id)

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        return self.outlines.glyph_bitmap_for_gid(glyph_id, width=width, height=height)


__all__ = ["OpenTypeFontProgram"]
