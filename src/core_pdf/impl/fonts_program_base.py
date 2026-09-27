from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

from core_pdf.impl.fonts_raster_kernel import (
    Point,
    rasterize_contours,
)

GlyphBox = tuple[float, float, float, float]
GlyphContours = tuple[tuple[Point, ...], ...]


class GlyphNaming(Protocol):
    @property
    def is_cid_font(self) -> bool: ...

    def glyph_name(self, code: int) -> str: ...


class GlyphBoxMetrics(Protocol):
    @property
    def ascent(self) -> float: ...

    @property
    def descent(self) -> float: ...

    def glyph_width(self, code: int) -> float: ...


class GlyphOutlineFallback(Protocol):
    def fallback_glyph_outline(self, text: str) -> GlyphContours: ...


class GlyphProgram:
    __slots__ = ()

    @abstractmethod
    def has_glyph_id(self, glyph_id: int, /) -> bool: ...

    @abstractmethod
    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None: ...

    @abstractmethod
    def glyph_contours_uncached(self, glyph_id: int) -> GlyphContours: ...

    @abstractmethod
    def glyph_bbox_uncached(self, glyph_id: int) -> GlyphBox | None: ...

    def normalized_glyph_contours(self, glyph_id: int) -> GlyphContours:
        return self.glyph_contours_uncached(glyph_id)

    def glyph_bbox_for_gid(self, glyph_id: int) -> GlyphBox | None:
        return self.glyph_bbox_uncached(glyph_id)

    def glyph_bitmap_for_gid(
        self, glyph_id: int, *, width: int = 24, height: int = 32
    ) -> tuple[int, ...]:
        return rasterize_contours(
            self.normalized_glyph_contours(glyph_id), width=width, height=height
        )

    def unicode_for_gid(self, gid: int) -> str:
        return ""

    def font_builtin_encoding(self) -> tuple[dict[int, str], bool] | None:
        return None

    def code_bbox(self, code: int) -> GlyphBox | None:
        return None

    def glyph_bbox_for_code(
        self, code: int, glyph_id: int | None, metrics: GlyphBoxMetrics
    ) -> GlyphBox | None:
        return self.glyph_bbox_for_gid(glyph_id) if glyph_id is not None else None

    def glyph_outline_for(
        self, glyph_id: int, text: str, fallback: GlyphOutlineFallback
    ) -> GlyphContours:
        return self.normalized_glyph_contours(glyph_id)


class NullProgram(GlyphProgram):
    __slots__ = ()

    def has_glyph_id(self, glyph_id: int) -> bool:
        return True

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        return code

    def glyph_contours_uncached(self, glyph_id: int) -> GlyphContours:
        return ()

    def glyph_bbox_uncached(self, glyph_id: int) -> GlyphBox | None:
        return None

    def glyph_bbox_for_code(
        self, code: int, glyph_id: int | None, metrics: GlyphBoxMetrics
    ) -> GlyphBox | None:
        width = metrics.glyph_width(code)
        return None if width <= 0 else (0.0, metrics.descent, width, metrics.ascent)

    def glyph_outline_for(
        self, glyph_id: int, text: str, fallback: GlyphOutlineFallback
    ) -> GlyphContours:
        return fallback.fallback_glyph_outline(text)


NULL_PROGRAM = NullProgram()
