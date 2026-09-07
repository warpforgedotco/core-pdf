"""Font contracts consumed by PDF text-state execution.

Font selection and Unicode recovery are provided by the caller. The execution
contract contains character evidence and PDF metrics, never captured runs or rasters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core_pdf.impl.spec.s_08_graphics.matrix import Matrix


@dataclass(frozen=True, slots=True)
class DecodedFontGlyph:
    code_bytes: bytes
    char_code: int
    cid: int
    gid: int | None
    unicode: str
    width_code: int


class FontService(Protocol):
    @property
    def is_type3(self) -> bool: ...
    @property
    def is_vertical(self) -> bool: ...
    @property
    def font(self) -> dict[str, Any]: ...
    @property
    def fast_widths(self) -> tuple[float, ...]: ...
    @property
    def ascent(self) -> float: ...
    @property
    def descent(self) -> float: ...
    @property
    def font_name(self) -> str | None: ...

    @property
    def font_matrix(self) -> Matrix: ...

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]: ...
    def glyph_name(self, code: int) -> str: ...
    def glyph_width(self, code: int) -> float: ...
    def vertical_glyph_position(self, code: int, *, font_size: float) -> tuple[float, float]: ...
    def glyph_advance_vector(
        self,
        code: int,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        encoded_space: bool,
    ) -> tuple[float, float]: ...
    def text_advance_vector(
        self,
        data: bytes | bytearray | memoryview,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        glyphs: tuple[DecodedFontGlyph, ...] | None = None,
    ) -> tuple[float, float]: ...


FontProvider = Callable[[dict[str, Any], dict[str, Any]], FontService]
