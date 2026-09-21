from __future__ import annotations

from io import BytesIO

from core_pdf._vendor.fontTools.ttLib import TTFont
from core_pdf.impl._impl.fonts.font_program_truetype import (
    FONT_PROGRAM_ERRORS,
    internal_FontToolsOutlineAccess,
)
from core_pdf.impl._impl.fonts.raster_kernel import Point


def parse_opentype_program(data: bytes) -> TTFont:
    font = TTFont(BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    if not ({"CFF ", "CFF2"} & set(font.keys())):
        raise ValueError("OpenType font has no CFF outline table")
    return font


class OpenTypeFontProgram:
    __slots__ = (
        "font",
        "outlines",
    )

    def __init__(self, data: bytes) -> None:
        try:
            self.font = parse_opentype_program(data)
            self.outlines = internal_FontToolsOutlineAccess(self.font)
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
