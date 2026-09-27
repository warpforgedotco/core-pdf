from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from core_pdf.impl.fonts_raster_kernel import Contours, rasterize_contours
from core_pdf.impl.types import Rectangle


class GlyphNaming(Protocol):
    @property
    def is_cid_font(self) -> bool: ...

    def glyph_name(self, code: int) -> str: ...


class GlyphProgram(ABC):
    __slots__ = ()

    @abstractmethod
    def has_glyph_id(self, glyph_id: int, /) -> bool: ...

    @abstractmethod
    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None: ...

    @abstractmethod
    def normalized_glyph_contours(self, glyph_id: int) -> Contours: ...

    @abstractmethod
    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None: ...

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

    def code_bbox(self, code: int) -> Rectangle | None:
        return None


class NullProgram(GlyphProgram):
    __slots__ = ()

    def has_glyph_id(self, glyph_id: int) -> bool:
        return True

    def glyph_id_for_code(self, code: int, naming: GlyphNaming) -> int | None:
        return code

    def normalized_glyph_contours(self, glyph_id: int) -> Contours:
        return ()

    def glyph_bbox_for_gid(self, glyph_id: int) -> Rectangle | None:
        return None


NULL_PROGRAM = NullProgram()
