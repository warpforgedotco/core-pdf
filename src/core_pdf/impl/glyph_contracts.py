# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core_pdf.impl.glyph_outlines import GlyphOutlineArrays
from core_pdf.impl.types import Rectangle
from core_pdf_spec.s_09_fonts.service import FontService


@runtime_checkable
class GlyphGeometrySource(Protocol):
    @property
    def font_name(self) -> str | None: ...

    @property
    def is_vertical(self) -> bool: ...

    @property
    def is_type3(self) -> bool: ...

    @property
    def ascent(self) -> float: ...

    @property
    def descent(self) -> float: ...

    def glyph_width(self, code: int, /) -> float: ...

    def glyph_bbox(self, code: int, /) -> Rectangle | None: ...

    def vertical_glyph_position(self, code: int, *, font_size: float) -> tuple[float, float]: ...

    def glyph_bitmap(self, code: int, *, width: int = 24, height: int = 32) -> tuple[int, ...]: ...

    def glyph_outline_arrays(
        self, code: int, gid: int | None = None, text: str = ""
    ) -> GlyphOutlineArrays | None: ...


@runtime_checkable
class CaptureFont(FontService, GlyphGeometrySource, Protocol):
    pass


__all__ = ("CaptureFont", "GlyphGeometrySource")
