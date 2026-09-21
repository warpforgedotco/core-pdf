# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, TypeAlias

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.color_rendering import (
    BlackPointCompensation,
    ColorRendering,
    parse_rendering_intent,
)
from core_pdf_spec.s_08_graphics.color_spec import DEVICE_GRAY, ColorSpace
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph, FontService
from core_pdf_spec.types import Rectangle

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.inline_images import InlineImage
    from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
    from core_pdf_spec.s_07_content.streams import ContentStreamFrame
    from core_pdf_spec.s_11_transparency.soft_masks import SoftMask

NON_PAINTING_RENDER_MODES = frozenset({3, 7})


@dataclass(frozen=True, slots=True)
class PathCommand:
    operator: str
    operands: tuple[float, ...]
    ctm: Matrix = IDENTITY_MATRIX
    flatness: float = 0.0


@dataclass(slots=True)
class PdfPath:
    commands: list[PathCommand] = field(default_factory=list)

    def move_to(self, x: float, y: float) -> None:
        self.commands.append(PathCommand("m", (x, y)))

    def line_to(self, x: float, y: float) -> None:
        self.commands.append(PathCommand("l", (x, y)))

    def close(self) -> None:
        self.commands.append(PathCommand("h", ()))

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.commands.append(PathCommand("re", (x, y, w, h)))

    def cubic_to(self, points: tuple[float, ...], ctm: Matrix, flatness: float) -> None:
        self.commands.append(PathCommand("c", points, ctm, flatness))


@dataclass(frozen=True, slots=True)
class ShadingPattern:
    dictionary: PdfDict
    extgstate: PdfDict | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class TilingPattern:
    bbox: Rectangle
    x_step: float
    y_step: float
    stream: PdfStream
    resources: PdfDict
    matrix: Matrix
    paint_type: int
    base_color: tuple[float, ...] | None
    base_color_spec: ColorSpace | None = field(default=None, kw_only=True)
    alpha_is_shape: bool = field(default=False, kw_only=True)
    text_knockout: bool = field(default=True, kw_only=True)


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


@dataclass(slots=True)
class MarkedContentEntry:
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None


@dataclass(slots=True)
class GraphicsState:
    ctm: Matrix = IDENTITY_MATRIX
    fill_color: tuple[float, ...] | None = (0.0,)
    fill_pattern: PatternPaint | None = None
    fill_opacity: float = 1.0
    stroke_color: tuple[float, ...] | None = (0.0,)
    stroke_pattern: PatternPaint | None = None
    stroke_opacity: float = 1.0
    fill_space: ColorSpace = DEVICE_GRAY
    stroke_space: ColorSpace = DEVICE_GRAY
    blend_mode: str | None = None
    flatness: float = 1.0
    render_intent: str | None = None
    black_point_compensation: BlackPointCompensation = "Default"
    line_width: float = 1.0
    line_cap: int = 0
    line_join: int = 0
    miter_limit: float = 10.0
    dash_pattern: tuple[tuple[float, ...], float] = ((), 0.0)
    font_size: float = 0.0
    horizontal_scale: float = 100.0
    char_space: float = 0.0
    word_space: float = 0.0
    rise: float = 0.0
    leading: float = 0.0
    render_mode: int = 0
    current_font: str | None = None
    current_decoder: FontService | None = None
    decoder_resources: PdfDict | None = None
    alpha_is_shape: bool = field(default=False, kw_only=True)
    text_knockout: bool = field(default=True, kw_only=True)
    soft_mask: SoftMask | None = field(default=None, kw_only=True)

    @property
    def color_rendering(self) -> ColorRendering:
        return internal_color_rendering(self.render_intent, self.black_point_compensation)

    def __copy__(self) -> GraphicsState:
        new = object.__new__(GraphicsState)
        for name in internal_GRAPHICS_STATE_FIELD_NAMES:
            setattr(new, name, getattr(self, name))
        return new


internal_GRAPHICS_STATE_FIELD_NAMES = tuple(item.name for item in fields(GraphicsState))


@lru_cache(maxsize=64)
def internal_color_rendering(
    intent: str | None, black_point: BlackPointCompensation
) -> ColorRendering:
    return ColorRendering(parse_rendering_intent(intent or "RelativeColorimetric"), black_point)


class ContentSink(Protocol):
    def show_text(
        self,
        state: ContentInterpreter,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[DecodedFontGlyph, ...],
        decoder: FontService,
        adv_x: float,
        adv_y: float,
        /,
    ) -> None: ...

    def text_boundary(self, state: ContentInterpreter, kind: str, /) -> None: ...

    def paint_path(
        self, state: ContentInterpreter, path: PdfPath, kind: str, fill_rule: str, /
    ) -> None: ...

    def clip_path(self, state: ContentInterpreter, path: PdfPath, fill_rule: str, /) -> None: ...

    def paint_image(self, state: ContentInterpreter, source: PdfStream, /) -> None: ...

    def paint_inline_image(self, state: ContentInterpreter, image: InlineImage, /) -> None: ...

    def paint_shading(self, state: ContentInterpreter, shading: PdfDict, /) -> None: ...

    def end_marked_content(
        self, state: ContentInterpreter, entry: MarkedContentEntry, /
    ) -> None: ...

    def save_graphics(self, state: ContentInterpreter, /) -> None: ...

    def restore_graphics(self, state: ContentInterpreter, /) -> None: ...

    def enter_stream(self, state: ContentInterpreter, frame: ContentStreamFrame, /) -> None: ...

    def exit_stream(self, state: ContentInterpreter, frame: ContentStreamFrame, /) -> None: ...


__all__ = (
    "NON_PAINTING_RENDER_MODES",
    "ContentSink",
    "GraphicsState",
    "MarkedContentEntry",
    "PathCommand",
    "PatternPaint",
    "PdfPath",
    "ShadingPattern",
    "TilingPattern",
)
