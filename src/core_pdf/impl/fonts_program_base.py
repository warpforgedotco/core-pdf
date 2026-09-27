from __future__ import annotations

from abc import abstractmethod

from core_pdf.impl.fonts_raster_kernel import (
    Point,
    rasterize_contours,
)


class BitmapFromContours:
    __slots__ = ()

    @abstractmethod
    def normalized_glyph_contours(self, glyph_id: int) -> tuple[tuple[Point, ...], ...]: ...

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        return rasterize_contours(
            self.normalized_glyph_contours(glyph_id), width=width, height=height
        )
