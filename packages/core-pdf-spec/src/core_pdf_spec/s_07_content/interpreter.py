# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import (
    NON_PAINTING_RENDER_MODES,
    ContentSink,
    GraphicsState,
    MarkedContentEntry,
    PatternPaint,
    PdfPath,
    ShadingPattern,
    TilingPattern,
)
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    OperationHandler,
    validate_content_operands,
)
from core_pdf_spec.s_07_content.streams import (
    ContentStreamExecutor,
    ContentStreamFrame,
    StreamKey,
    StreamState,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    parse_float,
)
from core_pdf_spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf_spec.s_08_graphics.color import (
    initial_color_components,
    normalize_color_components,
)
from core_pdf_spec.s_08_graphics.color_rendering import (
    parse_black_point_compensation,
    parse_rendering_intent,
)
from core_pdf_spec.s_08_graphics.color_spec import (
    DEVICE_CMYK,
    DEVICE_GRAY,
    DEVICE_RGB,
    ColorSpace,
    parse_color_space,
)
from core_pdf_spec.s_08_graphics.geometry import transform_bbox
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.metrics import text_adjustment_vector
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph, FontProvider, FontService
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask, parse_soft_mask
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString


def moved_to(matrix: Matrix, e: float, f: float) -> Matrix:
    """`matrix` with its translation replaced.

    Matrix._replace(e=e, f=f) without its keyword handling and _make: the
    text operators move the text and line matrices several times per string
    shown, and this is a third of the cost.
    """
    return tuple.__new__(Matrix, (matrix[0], matrix[1], matrix[2], matrix[3], e, f))


def parse_error_from(error: Exception) -> PdfParseError:
    """`error` restated as a PdfParseError, chained as `raise ... from error` chains it."""
    failure = PdfParseError(str(error))
    failure.__cause__ = error
    return failure


class ContentInterpreter:
    # The executor a subclass wants built for it. Overriding the attribute is
    # what keeps __init__ from constructing a base executor that the subclass
    # would only replace and discard.
    stream_executor_type: ClassVar[type[ContentStreamExecutor]] = ContentStreamExecutor

    def __init__(
        self,
        resolver: PdfValueResolver,
        sink: ContentSink,
        font_provider: FontProvider,
        lexer_factory: Callable[[bytes | memoryview], PdfLexer] = PdfLexer,
        *,
        semantic_context: SemanticContext | None = None,
    ):
        self.resolver = resolver
        self.sink = sink
        self.font_provider = font_provider
        self.lexer_factory = lexer_factory
        self.semantic_context = semantic_context
        self.graphics = GraphicsState()
        self.initial_alpha_is_shape = False
        self.initial_text_knockout = True
        self.in_text_object = False
        self.text_matrix = IDENTITY_MATRIX
        self.line_matrix = IDENTITY_MATRIX
        self.stack: list[GraphicsState] = []
        self.graphics_stack_floor = 0
        self.current_path = PdfPath()
        self.current_point: tuple[float, float] | None = None
        self.subpath_start: tuple[float, float] | None = None
        self.pending_clip_rule_value: str | None = None
        self.xobject_depth = 0
        self.compatibility_depth = 0
        self.marked_content_stack: list[MarkedContentEntry] = []
        self.type3_uncolored = False
        self.resources: PdfDict = {}
        self.operator_overrides: dict[str, OperationHandler] = {}
        self.default_handlers: dict[str, OperationHandler] = {
            name: getattr(self, handler) for name, handler in CONTENT_OPERATOR_HANDLERS.items()
        }
        self.stream_executor = self.stream_executor_type(self)

    def create_lexer(self, data: bytes | memoryview) -> PdfLexer:
        lexer = self.lexer_factory(data)
        if self.semantic_context is not None:
            try:
                lexer.semantic_context = self.semantic_context
            except BaseException:
                lexer.close()
                raise
        return lexer

    def reject[T](self, error: Exception, context: str, fallback: T) -> T:
        """Refuse content ISO 32000 does not permit, or recover from it.

        This interpreter raises `error`. A tolerant subclass may return
        instead: the caller then proceeds with `fallback`, the value a reader
        recovering from the failure uses, while `context` names the kind of
        failure. Callers that can continue past a failure ignore the return.
        """
        raise error

    def append_cubic_curve(
        self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> None:
        if self.current_point is None:
            raise PdfParseError("curve has no current point")
        x0, y0 = self.current_point
        self.current_path.cubic_to(
            (x0, y0, x1, y1, x2, y2, x3, y3), self.graphics.ctm, float(self.graphics.flatness)
        )
        self.current_point = (x3, y3)

    def capture_stream_state(self) -> StreamState:
        return StreamState(
            graphics_state=copy(self.graphics),
            resources=self.resources,
            text_matrix=self.text_matrix,
            line_matrix=self.line_matrix,
            graphics_stack_floor=self.graphics_stack_floor,
            graphics_stack_len=len(self.stack),
            marked_content_stack_len=len(self.marked_content_stack),
            xobject_depth=self.xobject_depth,
            compatibility_depth=self.compatibility_depth,
            pending_clip_rule=self.pending_clip_rule_value,
            initial_alpha_is_shape=self.initial_alpha_is_shape,
            initial_text_knockout=self.initial_text_knockout,
            in_text_object=self.in_text_object,
        )

    def restore_stream_state(self, state: StreamState) -> None:
        self.resources = state.resources
        self.text_matrix = state.text_matrix
        self.line_matrix = state.line_matrix
        self.graphics_stack_floor = state.graphics_stack_floor
        self.xobject_depth = state.xobject_depth
        self.compatibility_depth = state.compatibility_depth
        self.pending_clip_rule_value = state.pending_clip_rule
        self.initial_alpha_is_shape = state.initial_alpha_is_shape
        self.initial_text_knockout = state.initial_text_knockout
        self.in_text_object = state.in_text_object
        while len(self.stack) > state.graphics_stack_len:
            self.pop_graphics_save()
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.graphics = copy(state.graphics_state)

    def execute_operation(
        self, name: str, operands: ContentOperands, depth: int
    ) -> ContentStreamFrame | None:
        if name not in CONTENT_OPERATOR_HANDLERS:
            if self.compatibility_depth:
                return None
            raise PdfParseError(f"unknown content operator: {name}")
        override = self.operator_overrides.get(name)
        handler = override or self.default_handlers.get(name)
        if handler is None:
            raise PdfParseError(f"unsupported content operator: {name}")
        validate_content_operands(name, operands)
        if name == "BX":
            self.compatibility_depth += 1
        elif name == "EX":
            if not self.compatibility_depth:
                raise PdfParseError("unmatched EX operator")
            self.compatibility_depth -= 1
        if name in {"l", "c", "v", "y"} and self.current_point is None:
            raise PdfParseError("path operator has no current point")
        if name == "EMC" and not self.marked_content_stack:
            raise PdfParseError("unmatched EMC operator")
        if override is not None:
            self.validate_color_operation(name, operands)
        return handler(operands, depth)

    def validate_color_operation(self, name: str, operands: ContentOperands) -> None:
        if name not in {"SC", "SCN", "sc", "scn"} or self.type3_uncolored:
            return
        space = self.graphics.stroke_space if name in {"SC", "SCN"} else self.graphics.fill_space
        self.prepare_color_components(space, operands, allow_special=name in {"SCN", "scn"})

    def prepare_color_components(
        self, space: ColorSpace, operands: ContentOperands, *, allow_special: bool
    ) -> tuple[float, ...] | None:
        if not allow_special and space.kind not in {
            "DeviceGray",
            "DeviceRGB",
            "DeviceCMYK",
            "CalGray",
            "CalRGB",
            "Lab",
            "Indexed",
        }:
            self.reject(PdfParseError("color space requires SCN or scn"), "color-operator", None)
        if allow_special and space.kind == "Pattern":
            if not operands:
                return self.reject(
                    PdfParseError("Pattern color requires a pattern name"), "color-components", None
                )
            if not isinstance(operands[-1], PdfName):
                self.reject(
                    PdfParseError("Pattern color requires a pattern name"), "color-components", None
                )
            if space.base is None:
                if len(operands) != 1:
                    self.reject(
                        PdfParseError("invalid color component operands"), "color-components", None
                    )
                return ()
            return self.normalize_color_components(space.base, operands[:-1])
        return self.normalize_color_components(space, operands)

    def lookup_page_resource(self, category: str, name: str) -> object:
        entries = self.resolve_resources(self.resources.get(category))
        return entries.get(name) if entries is not None else None

    def resolve_resources(self, value: object) -> PdfDict | None:
        return resolve_resource_dict(value, self.resolver)

    def matrix_operand(self, value: object, context: str) -> Matrix:
        value = self.resolver.deep_resolve(value)
        if value is None:
            return IDENTITY_MATRIX
        return Matrix.from_operand(value)

    def get_decoder(self) -> FontService:
        if self.graphics.current_decoder is not None:
            return self.graphics.current_decoder
        if self.graphics.current_font is None:
            raise PdfParseError("text operation has no selected font")
        font_reference = self.lookup_page_resource("Font", self.graphics.current_font)
        font = self.resolver.resolve(font_reference)
        if not isinstance(font, dict):
            raise PdfParseError("font resource must be a dictionary")
        resolved_font = self.resolver.resolve_font_dict(font)
        decoder = self.font_provider(resolved_font, self.resources)
        self.graphics.current_decoder = decoder
        self.graphics.decoder_resources = self.resources
        return decoder

    def op_Do(self, operands: ContentOperands, depth: int) -> ContentStreamFrame | None:
        if not operands:
            return None
        return self.append_xobject(operands[0], depth)

    def append_xobject(self, name_obj: Any, depth: int) -> ContentStreamFrame | None:
        name = self.resolver.resolve_name(name_obj)
        if not name:
            return self.reject(PdfParseError("XObject operand must be a name"), "xobject", None)
        raw_xobj = self.lookup_page_resource("XObject", name)
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.resolver.resolve(raw_xobj)
        if not isinstance(xobj, PdfStream):
            return self.reject(PdfParseError("XObject resource must be a stream"), "xobject", None)
        xobj_dict = xobj.dictionary
        subtype = self.resolver.resolve_name(xobj_dict.get("Subtype"))
        # 7.5.7: an object stream holds objects; it is never content to paint.
        if self.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return self.reject(
                PdfParseError("XObject resource is an object stream"), "xobject", None
            )
        if subtype == "Image":
            self.sink.paint_image(self, xobj)
            return None
        if subtype != "Form":
            return self.reject(PdfParseError("unsupported XObject subtype"), "xobject", None)
        return self.append_form_xobject(xobj, depth, stream_key=stream_key)

    def resolve_form_resources(self, value: object) -> PdfDict:
        resources = self.resolve_resources(value)
        return self.resources if resources is None else resources

    def resolve_form_bbox(self, value: object) -> tuple[float, float, float, float] | None:
        bbox = self.resolver.resolve_box(value)
        if bbox is None:
            return self.reject(PdfParseError("Form XObject requires a BBox"), "form-bbox", None)
        return bbox

    def append_form_xobject(
        self, xobj: PdfStream, depth: int, *, stream_key: StreamKey | None = None
    ) -> ContentStreamFrame | None:
        xobj_dict = xobj.dictionary
        group_alpha = None
        group_isolated = True
        group_knockout = False
        group = xobj_dict.get("Group")
        if group is not None:
            group_dict = self.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.resolver.resolve_name(group_dict.get("S")) == "Transparency"
            ):
                group_alpha = max(0.0, min(1.0, self.graphics.fill_opacity))
                group_isolated = self.resolver.resolve(group_dict.get("I")) is True
                group_knockout = self.resolver.resolve(group_dict.get("K")) is True
        resources = self.resolve_form_resources(xobj_dict.get("Resources"))
        xobj_matrix = xobj_dict.get("Matrix")
        nested_ctm = self.matrix_operand(xobj_matrix, "form").multiply(self.graphics.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.resolve_form_bbox(raw_form_bbox)
        if form_bbox is not None:
            x0, y0, x1, y1 = form_bbox
            form_bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        frame = self.stream_executor.queue(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            form_bbox_operand=raw_form_bbox,
            group_alpha=group_alpha,
            stream_key=stream_key,
        )
        if frame is not None:
            frame.form_bbox = form_bbox
            if group_alpha is not None:
                frame.group_isolated = group_isolated
                frame.group_knockout = group_knockout
        return frame

    def append_text(self, data: bytes | memoryview, *, decoder: FontService | None = None) -> None:
        decoder = decoder if decoder is not None else self.get_decoder()
        glyphs = decoder.decode_glyphs(data if isinstance(data, bytes) else bytes(data))
        text = "".join(glyph.unicode for glyph in glyphs)
        self.append_decoded_text(text, data, glyphs, decoder)

    def append_decoded_text(
        self,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[DecodedFontGlyph, ...],
        decoder: FontService,
    ) -> None:
        if decoder.is_type3 and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self.render_type3_glyphs(data, decoder)
            self.text_matrix = text_matrix
            self.line_matrix = line_matrix
        if not text and not data:
            return

        adv_x, adv_y = decoder.text_advance_vector(
            data,
            font_size=self.graphics.font_size,
            char_space=self.graphics.char_space,
            word_space=self.graphics.word_space,
            horizontal_scale=self.graphics.horizontal_scale,
            glyphs=glyphs,
        )
        ta, tb, tc, td, te, tf = self.text_matrix
        if text:
            self.sink.show_text(self, text, data, glyphs, decoder, adv_x, adv_y)
        self.text_matrix = moved_to(
            self.text_matrix, te + adv_x * ta + adv_y * tc, tf + adv_x * tb + adv_y * td
        )
        self.sink.text_boundary(self, "shown")

    def render_type3_glyphs(self, data: bytes | memoryview, decoder: FontService) -> None:
        if self.graphics.render_mode in NON_PAINTING_RENDER_MODES:
            return
        font = decoder.font
        char_procs = font.get("CharProcs")
        if not isinstance(char_procs, dict):
            return
        resources = self.resolve_resources(font.get("Resources"))
        if resources is None:
            resources = self.resources
        font_matrix = decoder.font_matrix

        for code in data:
            glyph_name = decoder.glyph_name(code)
            char_proc = self.resolver.resolve(char_procs.get(glyph_name) if glyph_name else None)
            if isinstance(char_proc, PdfStream):
                text_space = self.text_matrix.multiply(self.graphics.ctm)
                font_size = self.graphics.font_size
                glyph_ctm = font_matrix.multiply(
                    Matrix(
                        font_size * self.graphics.horizontal_scale / 100.0,
                        0.0,
                        0.0,
                        font_size,
                        0.0,
                        self.graphics.rise,
                    ).multiply(text_space)
                )
                previous_type3_uncolored = self.type3_uncolored
                self.sink.text_boundary(self, "type3-glyph-begin")
                self.type3_uncolored = False
                try:
                    self.stream_executor.consume(
                        char_proc, resources, glyph_ctm, self.xobject_depth + 1
                    )
                finally:
                    self.type3_uncolored = previous_type3_uncolored
                    self.sink.text_boundary(self, "type3-glyph-end")

            advance_x, advance_y = decoder.glyph_advance_vector(
                code,
                font_size=self.graphics.font_size,
                char_space=self.graphics.char_space,
                word_space=self.graphics.word_space,
                horizontal_scale=self.graphics.horizontal_scale,
                encoded_space=code == 32,
            )
            tm = self.text_matrix
            self.text_matrix = moved_to(
                tm,
                tm.e + (advance_x * tm.a + advance_y * tm.c),
                tm.f + (advance_x * tm.b + advance_y * tm.d),
            )

    def tj_array_extra_bytes(self, item: object) -> bytes:
        raise PdfParseError("TJ array entries must be strings or numbers")

    def append_tj_array(self, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            self.reject(PdfParseError("TJ requires an array"), "text-array", None)
            return
        if not array:
            return
        pending_bytes = bytearray()

        decoder = self.get_decoder()
        is_vert = decoder.is_vertical

        ta, tb, tc, td, te, tf = self.text_matrix
        for item in array:
            t = type(item)
            if t is PdfString:
                pending_bytes.extend(item.data)
            elif t is bytes:
                pending_bytes.extend(item)
            elif t is int or t is float:
                if pending_bytes:
                    self.text_matrix = moved_to(self.text_matrix, te, tf)
                    self.append_text(data=bytes(pending_bytes), decoder=decoder)
                    te, tf = self.text_matrix.e, self.text_matrix.f
                    pending_bytes.clear()
                advance_x, advance_y = text_adjustment_vector(
                    item,
                    vertical=is_vert,
                    font_size=self.graphics.font_size,
                    horizontal_scale=self.graphics.horizontal_scale,
                )
                te += advance_x * ta + advance_y * tc
                tf += advance_x * tb + advance_y * td
            else:
                pending_bytes.extend(self.tj_array_extra_bytes(item))

        if pending_bytes:
            self.text_matrix = moved_to(self.text_matrix, te, tf)
            self.append_text(data=bytes(pending_bytes), decoder=decoder)
            te, tf = self.text_matrix.e, self.text_matrix.f

        self.text_matrix = moved_to(self.text_matrix, te, tf)

    def current_actual_text_span(self) -> MarkedContentEntry | None:
        for entry in reversed(self.marked_content_stack):
            if entry.actual_text is not None:
                return entry
        return None

    def current_marked_content_mcid(self) -> int | None:
        for entry in reversed(self.marked_content_stack):
            if type(entry.mcid) is int:
                return entry.mcid
        return None

    def begin_text(self) -> None:
        self.text_matrix = self.line_matrix = IDENTITY_MATRIX

    def op_ET(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "end")
        self.in_text_object = False

    def move_text(self, tx: float, ty: float) -> None:
        self.sink.text_boundary(self, "move")
        lm = self.line_matrix
        e = tx * lm.a + ty * lm.c + lm.e
        f = tx * lm.b + ty * lm.d + lm.f
        self.text_matrix = moved_to(self.text_matrix, e, f)
        self.line_matrix = moved_to(lm, e, f)

    def op_BT(self, operands: ContentOperands, depth: int) -> None:
        self.in_text_object = True
        self.sink.text_boundary(self, "begin")
        self.begin_text()

    def op_T_star(self, operands: ContentOperands, depth: int) -> None:
        self.move_text(0.0, -self.graphics.leading)

    def op_Td(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        self.move_text(*values)

    def op_TD(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        tx, ty = values
        self.graphics.leading = -ty
        self.move_text(tx, ty)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 1 or not isinstance(operands[0], PdfString):
            raise PdfParseError("Tj requires one string")
        self.append_text(operands[0].data)

    def op_TJ(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            self.append_tj_array(operands[0])

    def op_Tm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        self.sink.text_boundary(self, "matrix")
        self.text_matrix = self.line_matrix = Matrix(*values)

    def resolve_font_name(self, value: object) -> str | None:
        name = self.resolver.resolve_name(value)
        if name is None:
            return self.reject(PdfParseError("Tf requires a font name"), "font-name", None)
        return name

    def parse_font_size(self, value: object) -> float | None:
        try:
            return self.as_float(value)
        except (TypeError, ValueError) as error:
            return self.reject(parse_error_from(error), "font-size", None)

    def op_Tf(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 2:
            self.reject(PdfParseError("Tf requires two operands"), "font-operands", None)
            if len(operands) < 2:
                return
            operands = operands[:2]
        font_name = self.resolve_font_name(operands[0])
        if font_name is None:
            return
        font_size = self.parse_font_size(operands[1])
        if font_size is None:
            return
        graphics = self.graphics
        if graphics.current_font != font_name or graphics.decoder_resources is not self.resources:
            graphics.current_decoder = None
        graphics.current_font = font_name
        graphics.font_size = font_size
        graphics.current_decoder = self.get_decoder()

    def op_TL(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.graphics.leading = values[0]

    def op_Tc(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.graphics.char_space = values[0]

    def op_Tw(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        word_space = values[0]
        if self.graphics.word_space == word_space:
            return
        self.graphics.word_space = word_space

    def op_Tr(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.graphics.render_mode = value

    def op_Tz(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.graphics.horizontal_scale = values[0]

    def op_Ts(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.graphics.rise = values[0]

    def op_quote(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        self.move_text(0.0, -self.graphics.leading)
        self.sink.text_boundary(self, "quoted")
        self.op_Tj((operands[0],), depth)

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 3 or (values := self.as_floats(operands, 2)) is None:
            return
        self.graphics.word_space, self.graphics.char_space = values
        self.move_text(0.0, -self.graphics.leading)
        self.sink.text_boundary(self, "quoted")
        self.op_Tj((operands[2],), depth)

    def op_BI(self, operands: ContentOperands, depth: int) -> None:
        if operands and hasattr(operands[0], "dictionary"):
            self.sink.paint_inline_image(self, operands[0])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "marked")
        tag = self.resolver.resolve_name(operands[0]) if operands else None
        layer: str | None = None
        actual_text: str | None = None
        mcid: int | None = None
        if len(operands) >= 2:
            properties = operands[1]
            if tag == "OC":
                layer = self.resolve_marked_content_layer(properties)
            props = self.resolve_marked_content_properties(properties)
            if props is not None:
                resolver = self.resolver
                if tag == "Span":
                    actual_text = resolver.resolve_str(props.get("ActualText"))
                mcid = resolver.resolve_int(props.get("MCID"))
        self.marked_content_stack.append(
            MarkedContentEntry(layer=layer, actual_text=actual_text, mcid=mcid)
        )

    def op_BMC(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "marked")
        self.marked_content_stack.append(MarkedContentEntry())

    def op_EMC(self, operands: ContentOperands, depth: int) -> None:
        if not self.marked_content_stack:
            self.reject(PdfParseError("unmatched EMC operator"), "marked-content", None)
            return
        self.sink.end_marked_content(self, self.marked_content_stack.pop())
        self.sink.text_boundary(self, "marked")

    def op_G(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_GRAY, stroke=True)

    def op_RG(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_RGB, stroke=True)

    def op_K(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_CMYK, stroke=True)

    def op_w(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            if values[0] < 0:
                raise PdfParseError("line width must not be negative")
            self.graphics.line_width = values[0]

    def op_J(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.graphics.line_cap = value

    def op_j(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.graphics.line_join = value

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            if values[0] < 1:
                raise PdfParseError("miter limit must be at least one")
            self.graphics.miter_limit = values[0]

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 2:
            self.reject(PdfParseError("d requires two operands"), "dash-pattern", None)
            if len(operands) < 2:
                return
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            if isinstance(array_obj, (list, tuple)):
                dash_array = tuple(self.as_float(value) for value in array_obj)
            else:
                dash_array = self.reject(
                    PdfParseError("dash pattern must be an array"), "dash-pattern", ()
                )
        except (TypeError, ValueError) as error:
            self.reject(parse_error_from(error), "dash-pattern", None)
            return
        self.graphics.dash_pattern = (dash_array, phase)

    def op_m(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        x, y = values
        self.current_path.move_to(x, y)
        self.current_point = self.subpath_start = (x, y)

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None:
            self.reject(PdfParseError("path operator has no current point"), "path", None)
            return
        if (values := self.as_floats(operands, 2)) is None:
            return
        x, y = values
        self.current_path.line_to(x, y)
        self.current_point = (x, y)

    def op_re(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 4)) is None:
            return
        x, y, w, h = values
        self.current_path.rect(x, y, w, h)
        self.current_point = (x, y)
        self.subpath_start = (x, y)

    def op_h(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            self.current_path.close()
            self.current_point = self.subpath_start

    def op_c(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        x1, y1, x2, y2, x3, y3 = values
        self.append_cubic_curve(x1, y1, x2, y2, x3, y3)

    def op_v(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None:
            self.reject(PdfParseError("path operator has no current point"), "path", None)
            return
        if (values := self.as_floats(operands, 4)) is None:
            return
        x0, y0 = self.current_point
        x2, y2, x3, y3 = values
        self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None:
            self.reject(PdfParseError("path operator has no current point"), "path", None)
            return
        if (values := self.as_floats(operands, 4)) is None:
            return
        x1, y1, x3, y3 = values
        self.append_cubic_curve(x1, y1, x3, y3, x3, y3)

    def close_current_subpath(self) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            self.current_path.close()

    def complete_path(
        self, kind: str | None, fill_rule: str = "nonzero", *, close: bool = False
    ) -> None:
        if close:
            self.close_current_subpath()
        if kind is not None:
            self.sink.paint_path(self, self.current_path, kind, fill_rule)
        if self.pending_clip_rule_value is not None:
            self.sink.clip_path(self, self.current_path, self.pending_clip_rule_value)
        self.current_path = PdfPath()
        self.current_point = None
        self.subpath_start = None
        self.pending_clip_rule_value = None

    def op_paint_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("stroke")

    def op_paint_close_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("stroke", close=True)

    def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fill", "nonzero")

    def op_paint_fill_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fill", "evenodd")

    def op_paint_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fillstroke", "nonzero")

    def op_paint_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fillstroke", "evenodd")

    def op_paint_close_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fillstroke", "nonzero", close=True)

    def op_paint_close_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path("fillstroke", "evenodd", close=True)

    def op_paint_clear(self, operands: ContentOperands, depth: int) -> None:
        self.complete_path(None)

    def op_W(self, operands: ContentOperands, depth: int) -> None:
        self.pending_clip_rule_value = "nonzero"

    def op_W_star(self, operands: ContentOperands, depth: int) -> None:
        self.pending_clip_rule_value = "evenodd"

    def set_device_color(
        self, operands: ContentOperands, space: ColorSpace, *, stroke: bool
    ) -> None:
        # The count comes from the space so the two cannot disagree.
        count = len(space.component_ranges)
        if self.type3_uncolored or len(operands) < count:
            return
        normalized = self.normalize_color_components(space, operands[:count])
        if normalized is None:
            return
        self.set_paint(space, normalized, None, stroke=stroke)

    def set_paint(
        self,
        space: ColorSpace | None,
        color: tuple[float, ...] | None,
        pattern: PatternPaint | None,
        *,
        stroke: bool,
    ) -> None:
        """Set the stroking or nonstroking color and pattern, and the space unless None."""
        graphics = self.graphics
        if stroke:
            if space is not None:
                graphics.stroke_space = space
            graphics.stroke_color = color
            graphics.stroke_pattern = pattern
        else:
            if space is not None:
                graphics.fill_space = space
            graphics.fill_color = color
            graphics.fill_pattern = pattern

    def resolve_color_space(self, name_obj: Any) -> ColorSpace:
        name = self.resolver.resolve_name(name_obj)
        if name is None:
            return self.reject(
                PdfParseError("color space operand must be a name"), "color-space", DEVICE_GRAY
            )
        value = (
            name
            if name in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}
            else self.resolver.deep_resolve(self.lookup_page_resource("ColorSpace", name))
        )
        if value is None:
            value = self.reject(PdfParseError("missing color space resource"), "color-space", name)
        return self.parse_named_color_space(value, name)

    def parse_named_color_space(self, value: object, name: str) -> ColorSpace:
        """The color space `value`, the family or resource the operand `name` selects."""
        try:
            return parse_color_space(value)
        except (ValueError, TypeError) as error:
            return self.reject(parse_error_from(error), "color-space", ColorSpace(name, ()))

    def normalize_color_components(
        self, spec: ColorSpace, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        try:
            return normalize_color_components(spec, components)
        except ValueError as error:
            raise PdfParseError(str(error)) from error

    def initial_color_components(
        self, spec: ColorSpace, *, stroke: bool
    ) -> tuple[float, ...] | None:
        try:
            return initial_color_components(spec)
        except ValueError as error:
            raise PdfParseError(str(error)) from error

    def set_color_space(self, operands: ContentOperands, *, stroke: bool) -> None:
        if self.type3_uncolored:
            return
        if operands:
            space = self.resolve_color_space(operands[0])
            color = self.initial_color_components(space, stroke=stroke)
            self.set_paint(space, color, None, stroke=stroke)

    def op_CS(self, operands: ContentOperands, depth: int) -> None:
        self.set_color_space(operands, stroke=True)

    def op_cs(self, operands: ContentOperands, depth: int) -> None:
        self.set_color_space(operands, stroke=False)

    def op_SC(self, operands: ContentOperands, depth: int) -> None:
        self.set_color(operands, stroke=True, allow_pattern=False)

    def set_color(self, operands: ContentOperands, *, stroke: bool, allow_pattern: bool) -> None:
        if self.type3_uncolored:
            return
        space = self.graphics.stroke_space if stroke else self.graphics.fill_space
        normalized = self.prepare_color_components(space, operands, allow_special=allow_pattern)
        if normalized is None:
            return
        if space.kind == "Pattern":
            pattern = self.resolve_pattern_color(
                operands[-1], space=space, base_components=normalized
            )
            color = pattern.base_color if isinstance(pattern, TilingPattern) else None
            self.set_paint(None, color, pattern, stroke=stroke)
        else:
            self.set_paint(None, normalized, None, stroke=stroke)

    def op_SCN(self, operands: ContentOperands, depth: int) -> None:
        self.set_color(operands, stroke=True, allow_pattern=True)

    def op_sc(self, operands: ContentOperands, depth: int) -> None:
        self.set_color(operands, stroke=False, allow_pattern=False)

    def op_scN(self, operands: ContentOperands, depth: int) -> None:
        self.set_color(operands, stroke=False, allow_pattern=True)

    def op_i(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.graphics.flatness = max(0.0, min(100.0, values[0]))

    def op_ri(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        try:
            self.graphics.render_intent = parse_rendering_intent(self.named_value(operands[0]))
        except ValueError as error:
            raise PdfParseError(str(error)) from error

    def op_MP(self, operands: ContentOperands, depth: int) -> None:
        return

    def op_DP(self, operands: ContentOperands, depth: int) -> None:
        return

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        name = self.resolver.resolve_name(value)
        if name is not None or not allow_text:
            return name
        return self.resolver.resolve_str(value)

    def resolve_marked_content_properties(self, value: Any) -> PdfDict | None:
        if value is None:
            return None
        resolved = self.resolver.resolve(value)
        if isinstance(resolved, dict):
            return resolved
        name = self.resolver.resolve_name(value)
        if not name:
            return None
        props = self.resolver.resolve(self.lookup_page_resource("Properties", name))
        return props if isinstance(props, dict) else None

    def resolve_marked_content_layer(self, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.resolver.resolve(value)
        if isinstance(resolved, dict):
            oc = resolved.get("OC")
            if oc is not None:
                return self.named_value(oc, allow_text=True)

        return self.named_value(value, allow_text=True)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        pass

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        pass

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def resource_operand_name(self, operands: ContentOperands) -> str | None:
        """The resource name a resource operator (sh, gs) takes as its operand."""
        name = self.resolver.resolve_name(operands[0]) if operands else None
        if not name:
            return self.reject(PdfParseError("resource operator requires a name"), "resource", None)
        return name

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        name = self.resource_operand_name(operands)
        if name is None:
            return
        shading = self.resolver.resolve_dict(self.lookup_page_resource("Shading", name))
        if not isinstance(shading, dict):
            self.reject(PdfParseError("missing shading resource"), "shading", None)
            return
        self.sink.paint_shading(self, shading)

    @staticmethod
    def as_float(value: Any) -> float:
        if type(value) in (float, int):
            parsed = parse_float(value, default=None)
            if parsed is not None:
                return parsed
        raise PdfParseError("numeric operand must be a PDF number")

    def as_floats(self, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        if len(operands) < count:
            raise PdfParseError("missing numeric operand")
        return tuple(self.as_float(operand) for operand in operands[:count])

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        raise PdfParseError("integer operand must be a PDF integer")

    def as_int_operand(self, operands: ContentOperands) -> int | None:
        if not operands:
            raise PdfParseError("missing numeric operand")
        return self.as_int(operands[0])

    def resolve_extgstate(self, name: str) -> dict[str, Any] | None:
        extgstate = self.resolver.resolve(self.lookup_page_resource("ExtGState", name))
        if not isinstance(extgstate, dict):
            return None
        source = extgstate
        values = {
            key: value
            for key, value in source.items()
            if key != "SMask" and (key != "TK" or not self.in_text_object)
        }
        resolved = self.resolver.resolve_dict(values)
        if not isinstance(resolved, dict):
            return None
        if "SMask" in source:
            resolved["SMask"] = source["SMask"]
        return resolved  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    def resolve_soft_mask(self, value: object) -> SoftMask | None:
        return parse_soft_mask(value, self.resolver, ctm=self.graphics.ctm)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.stack.append(self.graphics.__copy__())
        self.sink.save_graphics(self)

    def pop_graphics_save(self) -> GraphicsState:
        saved = self.stack.pop()
        self.sink.restore_graphics(self)
        return saved

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if len(self.stack) <= self.graphics_stack_floor:
            self.reject(PdfParseError("unmatched Q operator"), "graphics-state", None)
            return
        self.graphics = self.pop_graphics_save()

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        self.graphics.ctm = Matrix(*values).multiply(self.graphics.ctm)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_GRAY, stroke=False)

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_RGB, stroke=False)

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        self.set_device_color(operands, DEVICE_CMYK, stroke=False)

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        name = self.resource_operand_name(operands)
        if name is None:
            return
        extgstate = self.resolve_extgstate(name)
        if extgstate is None:
            self.reject(
                PdfParseError("missing ExtGState resource"), "extended-graphics-state", None
            )
            return
        try:
            self.apply_extgstate(extgstate)
        except (TypeError, ValueError) as error:
            self.reject(parse_error_from(error), "extended-graphics-state", None)

    def apply_extgstate(self, extgstate: dict[str, Any]) -> None:
        intent = self.resolver.resolve(extgstate.get("RI"))
        if intent is not None:
            self.graphics.render_intent = parse_rendering_intent(self.named_value(intent))
        black_point = self.resolver.resolve(extgstate.get("UseBlackPtComp"))
        if black_point is not None:
            self.graphics.black_point_compensation = parse_black_point_compensation(
                self.named_value(black_point)
            )
        fill_opacity = extgstate.get("ca")
        if fill_opacity is not None:
            self.graphics.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
        stroke_opacity = extgstate.get("CA")
        if stroke_opacity is not None:
            self.graphics.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
        blend_mode = extgstate.get("BM")
        if blend_mode is not None:
            if isinstance(blend_mode, (list, tuple)):
                blend_mode = blend_mode[0] if blend_mode else None
            if blend_mode is not None:
                self.graphics.blend_mode = self.named_value(blend_mode)
        alpha_is_shape = self.resolver.resolve(extgstate.get("AIS"))
        if alpha_is_shape is not None:
            if not isinstance(alpha_is_shape, bool):
                raise ValueError("invalid alpha source flag")
            self.graphics.alpha_is_shape = alpha_is_shape
        if not self.in_text_object:
            text_knockout = self.resolver.resolve(extgstate.get("TK"))
            if text_knockout is not None:
                if not isinstance(text_knockout, bool):
                    raise ValueError("invalid text knockout flag")
                self.graphics.text_knockout = text_knockout
        soft_mask = self.resolver.resolve(extgstate.get("SMask"))
        if soft_mask is not None:
            self.graphics.soft_mask = self.resolve_soft_mask(soft_mask)

    def resolve_pattern_resource(self, name_operand: object) -> tuple[object, PdfDict] | None:
        pattern_name = self.resolver.resolve_name(name_operand)
        if not pattern_name:
            return None
        pattern = self.resolver.resolve(self.lookup_page_resource("Pattern", pattern_name))
        pattern_dict: PdfDict | None
        if isinstance(pattern, PdfStream):
            pattern_dict = pattern.dictionary
        else:
            pattern_dict = self.resolver.resolve_dict(pattern) if pattern is not None else None
        return (pattern, pattern_dict) if isinstance(pattern_dict, dict) else None

    def resolve_pattern_color(
        self, pattern_name: object, *, space: ColorSpace, base_components: tuple[float, ...]
    ) -> PatternPaint | None:
        invalid = "invalid pattern resource or operands"
        resource = self.resolve_pattern_resource(pattern_name)
        if resource is None:
            return self.reject(PdfParseError(invalid), "pattern", None)
        pattern, pattern_dict = resource
        pattern_type = self.resolver.resolve_int(pattern_dict.get("PatternType"))
        if pattern_type == 2:
            if space.base is not None:
                self.reject(
                    PdfParseError("shading pattern requires a colored Pattern space"),
                    "pattern",
                    None,
                )
            shading: object = pattern_dict.get("Shading")
            shading = self.resolver.resolve(shading)
            shading_dict = self.resolver.resolve_dict(shading) if shading is not None else None
            if not isinstance(shading_dict, dict):
                return self.reject(PdfParseError(invalid), "pattern", None)
            extgstate = self.resolver.resolve_dict(pattern_dict.get("ExtGState"))
            return ShadingPattern(dict(shading_dict), extgstate=extgstate)
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return self.reject(PdfParseError(invalid), "pattern", None)
        paint_type = self.resolver.resolve_int(pattern_dict.get("PaintType"))
        if paint_type is None:
            # Recovery reads a missing or malformed PaintType as colored.
            paint_type = self.reject(PdfParseError(invalid), "pattern", 1)
        if paint_type not in {1, 2}:
            return self.reject(PdfParseError(invalid), "pattern", None)
        base_spec = space.base
        if (paint_type == 2) != (base_spec is not None):
            self.reject(
                PdfParseError("pattern PaintType does not match its color space"), "pattern", None
            )
        base_color = base_components if paint_type == 2 else None
        bbox = self.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            return self.reject(PdfParseError(invalid), "pattern", None)
        x_step = self.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return self.reject(PdfParseError(invalid), "pattern", None)
        matrix = self.matrix_operand(pattern_dict.get("Matrix"), "pattern")
        resources = self.resolve_resources(pattern_dict.get("Resources")) or {}
        return TilingPattern(
            bbox=bbox,
            x_step=float(x_step),
            y_step=float(y_step),
            stream=pattern,
            resources=resources,
            matrix=matrix,
            paint_type=paint_type,
            base_color=base_color,
            base_color_spec=base_spec,
            alpha_is_shape=self.initial_alpha_is_shape,
            text_knockout=self.initial_text_knockout,
        )


__all__ = ("ContentInterpreter",)
