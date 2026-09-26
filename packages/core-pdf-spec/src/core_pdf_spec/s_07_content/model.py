# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Protocol, Self, TypeAlias

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

frozen_setattr = object.__setattr__


NON_PAINTING_RENDER_MODES = frozenset({3, 7})


# One byte per path construction operator, and how many numbers each keeps in
# PdfPath.coords: a curve keeps its start point, its three points, the linear
# part of the CTM it was drawn under (a, b, c, d) and the flatness.
PATH_MOVE = ord("m")
PATH_LINE = ord("l")
PATH_CLOSE = ord("h")
PATH_RECT = ord("r")
PATH_CURVE = ord("c")
PATH_OPERAND_COUNTS = {PATH_MOVE: 2, PATH_LINE: 2, PATH_CLOSE: 0, PATH_RECT: 4, PATH_CURVE: 13}


class PdfPath:
    """A path under construction, as flat storage rather than an object per operator.

    ``ops`` holds one byte per operator and ``coords`` its numbers, in order,
    as PATH_OPERAND_COUNTS lays them out: content streams build paths of
    thousands of segments, and a command object with an operand tuple for
    each was most of what building one cost.
    """

    __slots__ = ("ops", "coords")

    ops: bytearray
    coords: array[float]

    __fields__: ClassVar[tuple[str, ...]] = ("ops", "coords")
    __match_args__ = ("ops", "coords")

    def __init__(self, ops: bytearray | None = None, coords: array[float] | None = None) -> None:
        self.ops = bytearray() if ops is None else ops
        self.coords = array("d") if coords is None else coords

    def __repr__(self) -> str:
        name = self.__class__.__qualname__
        return f"{name}(ops={bytes(self.ops)!r}, coords={self.coords.tolist()!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.ops == other.ops and self.coords == other.coords

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        return bool(self.ops)

    def __replace__(self, /, **changes: Any) -> Self:
        ops = changes.pop("ops", self.ops)
        coords = changes.pop("coords", self.coords)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(ops, coords)

    def move_to(self, x: float, y: float) -> None:
        self.ops.append(PATH_MOVE)
        self.coords.append(x)
        self.coords.append(y)

    def line_to(self, x: float, y: float) -> None:
        self.ops.append(PATH_LINE)
        self.coords.append(x)
        self.coords.append(y)

    def close(self) -> None:
        self.ops.append(PATH_CLOSE)

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        self.ops.append(PATH_RECT)
        self.coords.extend((x, y, w, h))

    def cubic_to(self, points: tuple[float, ...], ctm: Matrix, flatness: float) -> None:
        """A curve from its eight numbers: start point, two controls, end point."""
        if len(points) != 8:
            raise ValueError(f"expected 8 values to unpack, got {len(points)}")
        self.ops.append(PATH_CURVE)
        self.coords.extend(points)
        self.coords.extend((ctm.a, ctm.b, ctm.c, ctm.d, flatness))

    def operators(self) -> list[str]:
        """The operators in order, as content-stream names (re for a rectangle)."""
        return ["re" if op == PATH_RECT else chr(op) for op in self.ops]


class ShadingPattern:
    __slots__ = ("dictionary", "extgstate")

    dictionary: PdfDict
    extgstate: PdfDict | None

    __fields__: ClassVar[tuple[str, ...]] = ("dictionary", "extgstate")
    __match_args__ = ("dictionary",)

    def __init__(self, dictionary: PdfDict, *, extgstate: PdfDict | None = None) -> None:
        frozen_setattr(self, "dictionary", dictionary)
        frozen_setattr(self, "extgstate", extgstate)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"dictionary={self.dictionary!r}, "
            f"extgstate={self.extgstate!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.dictionary == other.dictionary and self.extgstate == other.extgstate

    def __hash__(self) -> int:
        return hash((self.dictionary, self.extgstate))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        dictionary = changes.pop("dictionary", self.dictionary)
        extgstate = changes.pop("extgstate", self.extgstate)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(dictionary, extgstate=extgstate)


class TilingPattern:
    __slots__ = (
        "bbox",
        "x_step",
        "y_step",
        "stream",
        "resources",
        "matrix",
        "paint_type",
        "base_color",
        "base_color_spec",
        "alpha_is_shape",
        "text_knockout",
    )

    bbox: Rectangle
    x_step: float
    y_step: float
    stream: PdfStream
    resources: PdfDict
    matrix: Matrix
    paint_type: int
    base_color: tuple[float, ...] | None
    base_color_spec: ColorSpace | None
    alpha_is_shape: bool
    text_knockout: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "bbox",
        "x_step",
        "y_step",
        "stream",
        "resources",
        "matrix",
        "paint_type",
        "base_color",
        "base_color_spec",
        "alpha_is_shape",
        "text_knockout",
    )
    __match_args__ = (
        "bbox",
        "x_step",
        "y_step",
        "stream",
        "resources",
        "matrix",
        "paint_type",
        "base_color",
    )

    def __init__(
        self,
        bbox: Rectangle,
        x_step: float,
        y_step: float,
        stream: PdfStream,
        resources: PdfDict,
        matrix: Matrix,
        paint_type: int,
        base_color: tuple[float, ...] | None,
        *,
        base_color_spec: ColorSpace | None = None,
        alpha_is_shape: bool = False,
        text_knockout: bool = True,
    ) -> None:
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "x_step", x_step)
        frozen_setattr(self, "y_step", y_step)
        frozen_setattr(self, "stream", stream)
        frozen_setattr(self, "resources", resources)
        frozen_setattr(self, "matrix", matrix)
        frozen_setattr(self, "paint_type", paint_type)
        frozen_setattr(self, "base_color", base_color)
        frozen_setattr(self, "base_color_spec", base_color_spec)
        frozen_setattr(self, "alpha_is_shape", alpha_is_shape)
        frozen_setattr(self, "text_knockout", text_knockout)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"bbox={self.bbox!r}, "
            f"x_step={self.x_step!r}, "
            f"y_step={self.y_step!r}, "
            f"stream={self.stream!r}, "
            f"resources={self.resources!r}, "
            f"matrix={self.matrix!r}, "
            f"paint_type={self.paint_type!r}, "
            f"base_color={self.base_color!r}, "
            f"base_color_spec={self.base_color_spec!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"text_knockout={self.text_knockout!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.x_step == other.x_step
            and self.y_step == other.y_step
            and self.stream == other.stream
            and self.resources == other.resources
            and self.matrix == other.matrix
            and self.paint_type == other.paint_type
            and self.base_color == other.base_color
            and self.base_color_spec == other.base_color_spec
            and self.alpha_is_shape == other.alpha_is_shape
            and self.text_knockout == other.text_knockout
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.bbox,
                self.x_step,
                self.y_step,
                self.stream,
                self.resources,
                self.matrix,
                self.paint_type,
                self.base_color,
                self.base_color_spec,
                self.alpha_is_shape,
                self.text_knockout,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        bbox = changes.pop("bbox", self.bbox)
        x_step = changes.pop("x_step", self.x_step)
        y_step = changes.pop("y_step", self.y_step)
        stream = changes.pop("stream", self.stream)
        resources = changes.pop("resources", self.resources)
        matrix = changes.pop("matrix", self.matrix)
        paint_type = changes.pop("paint_type", self.paint_type)
        base_color = changes.pop("base_color", self.base_color)
        base_color_spec = changes.pop("base_color_spec", self.base_color_spec)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        text_knockout = changes.pop("text_knockout", self.text_knockout)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            bbox,
            x_step,
            y_step,
            stream,
            resources,
            matrix,
            paint_type,
            base_color,
            base_color_spec=base_color_spec,
            alpha_is_shape=alpha_is_shape,
            text_knockout=text_knockout,
        )


PatternPaint: TypeAlias = ShadingPattern | TilingPattern


class MarkedContentEntry:
    __slots__ = ("layer", "actual_text", "mcid")

    layer: str | None
    actual_text: str | None
    mcid: int | None

    __fields__: ClassVar[tuple[str, ...]] = ("layer", "actual_text", "mcid")
    __match_args__ = ("layer", "actual_text", "mcid")

    def __init__(
        self,
        layer: str | None = None,
        actual_text: str | None = None,
        mcid: int | None = None,
    ) -> None:
        self.layer = layer
        self.actual_text = actual_text
        self.mcid = mcid

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"layer={self.layer!r}, "
            f"actual_text={self.actual_text!r}, "
            f"mcid={self.mcid!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.layer == other.layer
            and self.actual_text == other.actual_text
            and self.mcid == other.mcid
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        layer = changes.pop("layer", self.layer)
        actual_text = changes.pop("actual_text", self.actual_text)
        mcid = changes.pop("mcid", self.mcid)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(layer, actual_text, mcid)


class GraphicsState:
    __slots__ = (
        "ctm",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "fill_space",
        "stroke_space",
        "blend_mode",
        "flatness",
        "render_intent",
        "black_point_compensation",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "decoder_resources",
        "alpha_is_shape",
        "text_knockout",
        "soft_mask",
    )

    ctm: Matrix
    fill_color: tuple[float, ...] | None
    fill_pattern: PatternPaint | None
    fill_opacity: float
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float
    fill_space: ColorSpace
    stroke_space: ColorSpace
    blend_mode: str | None
    flatness: float
    render_intent: str | None
    black_point_compensation: BlackPointCompensation
    line_width: float
    line_cap: int
    line_join: int
    miter_limit: float
    dash_pattern: tuple[tuple[float, ...], float]
    font_size: float
    horizontal_scale: float
    char_space: float
    word_space: float
    rise: float
    leading: float
    render_mode: int
    current_font: str | None
    current_decoder: FontService | None
    decoder_resources: PdfDict | None
    alpha_is_shape: bool
    text_knockout: bool
    soft_mask: SoftMask | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "ctm",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "fill_space",
        "stroke_space",
        "blend_mode",
        "flatness",
        "render_intent",
        "black_point_compensation",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "decoder_resources",
        "alpha_is_shape",
        "text_knockout",
        "soft_mask",
    )
    __match_args__ = (
        "ctm",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "fill_space",
        "stroke_space",
        "blend_mode",
        "flatness",
        "render_intent",
        "black_point_compensation",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "decoder_resources",
    )

    def __init__(
        self,
        ctm: Matrix = IDENTITY_MATRIX,
        fill_color: tuple[float, ...] | None = (0.0,),
        fill_pattern: PatternPaint | None = None,
        fill_opacity: float = 1.0,
        stroke_color: tuple[float, ...] | None = (0.0,),
        stroke_pattern: PatternPaint | None = None,
        stroke_opacity: float = 1.0,
        fill_space: ColorSpace = DEVICE_GRAY,
        stroke_space: ColorSpace = DEVICE_GRAY,
        blend_mode: str | None = None,
        flatness: float = 1.0,
        render_intent: str | None = None,
        black_point_compensation: BlackPointCompensation = "Default",
        line_width: float = 1.0,
        line_cap: int = 0,
        line_join: int = 0,
        miter_limit: float = 10.0,
        dash_pattern: tuple[tuple[float, ...], float] = ((), 0.0),
        font_size: float = 0.0,
        horizontal_scale: float = 100.0,
        char_space: float = 0.0,
        word_space: float = 0.0,
        rise: float = 0.0,
        leading: float = 0.0,
        render_mode: int = 0,
        current_font: str | None = None,
        current_decoder: FontService | None = None,
        decoder_resources: PdfDict | None = None,
        *,
        alpha_is_shape: bool = False,
        text_knockout: bool = True,
        soft_mask: SoftMask | None = None,
    ) -> None:
        self.ctm = ctm
        self.fill_color = fill_color
        self.fill_pattern = fill_pattern
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_pattern = stroke_pattern
        self.stroke_opacity = stroke_opacity
        self.fill_space = fill_space
        self.stroke_space = stroke_space
        self.blend_mode = blend_mode
        self.flatness = flatness
        self.render_intent = render_intent
        self.black_point_compensation = black_point_compensation
        self.line_width = line_width
        self.line_cap = line_cap
        self.line_join = line_join
        self.miter_limit = miter_limit
        self.dash_pattern = dash_pattern
        self.font_size = font_size
        self.horizontal_scale = horizontal_scale
        self.char_space = char_space
        self.word_space = word_space
        self.rise = rise
        self.leading = leading
        self.render_mode = render_mode
        self.current_font = current_font
        self.current_decoder = current_decoder
        self.decoder_resources = decoder_resources
        self.alpha_is_shape = alpha_is_shape
        self.text_knockout = text_knockout
        self.soft_mask = soft_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"ctm={self.ctm!r}, "
            f"fill_color={self.fill_color!r}, "
            f"fill_pattern={self.fill_pattern!r}, "
            f"fill_opacity={self.fill_opacity!r}, "
            f"stroke_color={self.stroke_color!r}, "
            f"stroke_pattern={self.stroke_pattern!r}, "
            f"stroke_opacity={self.stroke_opacity!r}, "
            f"fill_space={self.fill_space!r}, "
            f"stroke_space={self.stroke_space!r}, "
            f"blend_mode={self.blend_mode!r}, "
            f"flatness={self.flatness!r}, "
            f"render_intent={self.render_intent!r}, "
            f"black_point_compensation={self.black_point_compensation!r}, "
            f"line_width={self.line_width!r}, "
            f"line_cap={self.line_cap!r}, "
            f"line_join={self.line_join!r}, "
            f"miter_limit={self.miter_limit!r}, "
            f"dash_pattern={self.dash_pattern!r}, "
            f"font_size={self.font_size!r}, "
            f"horizontal_scale={self.horizontal_scale!r}, "
            f"char_space={self.char_space!r}, "
            f"word_space={self.word_space!r}, "
            f"rise={self.rise!r}, "
            f"leading={self.leading!r}, "
            f"render_mode={self.render_mode!r}, "
            f"current_font={self.current_font!r}, "
            f"current_decoder={self.current_decoder!r}, "
            f"decoder_resources={self.decoder_resources!r}, "
            f"alpha_is_shape={self.alpha_is_shape!r}, "
            f"text_knockout={self.text_knockout!r}, "
            f"soft_mask={self.soft_mask!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.ctm == other.ctm
            and self.fill_color == other.fill_color
            and self.fill_pattern == other.fill_pattern
            and self.fill_opacity == other.fill_opacity
            and self.stroke_color == other.stroke_color
            and self.stroke_pattern == other.stroke_pattern
            and self.stroke_opacity == other.stroke_opacity
            and self.fill_space == other.fill_space
            and self.stroke_space == other.stroke_space
            and self.blend_mode == other.blend_mode
            and self.flatness == other.flatness
            and self.render_intent == other.render_intent
            and self.black_point_compensation == other.black_point_compensation
            and self.line_width == other.line_width
            and self.line_cap == other.line_cap
            and self.line_join == other.line_join
            and self.miter_limit == other.miter_limit
            and self.dash_pattern == other.dash_pattern
            and self.font_size == other.font_size
            and self.horizontal_scale == other.horizontal_scale
            and self.char_space == other.char_space
            and self.word_space == other.word_space
            and self.rise == other.rise
            and self.leading == other.leading
            and self.render_mode == other.render_mode
            and self.current_font == other.current_font
            and self.current_decoder == other.current_decoder
            and self.decoder_resources == other.decoder_resources
            and self.alpha_is_shape == other.alpha_is_shape
            and self.text_knockout == other.text_knockout
            and self.soft_mask == other.soft_mask
        )

    __hash__ = None  # type: ignore[assignment]

    def __replace__(self, /, **changes: Any) -> Self:
        ctm = changes.pop("ctm", self.ctm)
        fill_color = changes.pop("fill_color", self.fill_color)
        fill_pattern = changes.pop("fill_pattern", self.fill_pattern)
        fill_opacity = changes.pop("fill_opacity", self.fill_opacity)
        stroke_color = changes.pop("stroke_color", self.stroke_color)
        stroke_pattern = changes.pop("stroke_pattern", self.stroke_pattern)
        stroke_opacity = changes.pop("stroke_opacity", self.stroke_opacity)
        fill_space = changes.pop("fill_space", self.fill_space)
        stroke_space = changes.pop("stroke_space", self.stroke_space)
        blend_mode = changes.pop("blend_mode", self.blend_mode)
        flatness = changes.pop("flatness", self.flatness)
        render_intent = changes.pop("render_intent", self.render_intent)
        black_point_compensation = changes.pop(
            "black_point_compensation", self.black_point_compensation
        )
        line_width = changes.pop("line_width", self.line_width)
        line_cap = changes.pop("line_cap", self.line_cap)
        line_join = changes.pop("line_join", self.line_join)
        miter_limit = changes.pop("miter_limit", self.miter_limit)
        dash_pattern = changes.pop("dash_pattern", self.dash_pattern)
        font_size = changes.pop("font_size", self.font_size)
        horizontal_scale = changes.pop("horizontal_scale", self.horizontal_scale)
        char_space = changes.pop("char_space", self.char_space)
        word_space = changes.pop("word_space", self.word_space)
        rise = changes.pop("rise", self.rise)
        leading = changes.pop("leading", self.leading)
        render_mode = changes.pop("render_mode", self.render_mode)
        current_font = changes.pop("current_font", self.current_font)
        current_decoder = changes.pop("current_decoder", self.current_decoder)
        decoder_resources = changes.pop("decoder_resources", self.decoder_resources)
        alpha_is_shape = changes.pop("alpha_is_shape", self.alpha_is_shape)
        text_knockout = changes.pop("text_knockout", self.text_knockout)
        soft_mask = changes.pop("soft_mask", self.soft_mask)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            ctm,
            fill_color,
            fill_pattern,
            fill_opacity,
            stroke_color,
            stroke_pattern,
            stroke_opacity,
            fill_space,
            stroke_space,
            blend_mode,
            flatness,
            render_intent,
            black_point_compensation,
            line_width,
            line_cap,
            line_join,
            miter_limit,
            dash_pattern,
            font_size,
            horizontal_scale,
            char_space,
            word_space,
            rise,
            leading,
            render_mode,
            current_font,
            current_decoder,
            decoder_resources,
            alpha_is_shape=alpha_is_shape,
            text_knockout=text_knockout,
            soft_mask=soft_mask,
        )

    @property
    def color_rendering(self) -> ColorRendering:
        return color_rendering(self.render_intent, self.black_point_compensation)

    def __copy__(self) -> GraphicsState:
        # Every q copies the state, so the fields are assigned one by one
        # rather than looped over by name: six times faster.
        new = object.__new__(GraphicsState)
        new.ctm = self.ctm
        new.fill_color = self.fill_color
        new.fill_pattern = self.fill_pattern
        new.fill_opacity = self.fill_opacity
        new.stroke_color = self.stroke_color
        new.stroke_pattern = self.stroke_pattern
        new.stroke_opacity = self.stroke_opacity
        new.fill_space = self.fill_space
        new.stroke_space = self.stroke_space
        new.blend_mode = self.blend_mode
        new.flatness = self.flatness
        new.render_intent = self.render_intent
        new.black_point_compensation = self.black_point_compensation
        new.line_width = self.line_width
        new.line_cap = self.line_cap
        new.line_join = self.line_join
        new.miter_limit = self.miter_limit
        new.dash_pattern = self.dash_pattern
        new.font_size = self.font_size
        new.horizontal_scale = self.horizontal_scale
        new.char_space = self.char_space
        new.word_space = self.word_space
        new.rise = self.rise
        new.leading = self.leading
        new.render_mode = self.render_mode
        new.current_font = self.current_font
        new.current_decoder = self.current_decoder
        new.decoder_resources = self.decoder_resources
        new.alpha_is_shape = self.alpha_is_shape
        new.text_knockout = self.text_knockout
        new.soft_mask = self.soft_mask
        return new


@lru_cache(maxsize=64)
def color_rendering(intent: str | None, black_point: BlackPointCompensation) -> ColorRendering:
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
    "PATH_CLOSE",
    "PATH_CURVE",
    "PATH_LINE",
    "PATH_MOVE",
    "PATH_OPERAND_COUNTS",
    "PATH_RECT",
    "PatternPaint",
    "PdfPath",
    "ShadingPattern",
    "TilingPattern",
)
