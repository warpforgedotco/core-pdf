# SPDX-License-Identifier: AGPL-3.0-only
"""PDF content operators and their graphics/text state transitions."""

from __future__ import annotations

import typing
from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING, Any, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.model import (
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
from core_pdf_spec.standards import SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.inline_images import InlineImage


class ContentInterpreter:
    """Execute PDF content with strict operands and an explicit graphics state."""

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
        self.text_matrix = IDENTITY_MATRIX
        self.line_matrix = IDENTITY_MATRIX
        self.stack: list[GraphicsState] = []
        self.graphics_stack_floor = 0
        self.current_path = PdfPath()
        self.current_point: tuple[float, float] | None = None
        self.subpath_start: tuple[float, float] | None = None
        self.internal_pending_clip_rule: str | None = None
        self.xobject_depth = 0
        self.compatibility_depth = 0
        self.marked_content_stack: list[MarkedContentEntry] = []
        self.type3_uncolored = False
        self.resources: PdfDict = {}
        self.operator_overrides: dict[str, OperationHandler] = {}
        self.internal_default_handlers: dict[str, OperationHandler] = {
            name: getattr(self, handler) for name, handler in CONTENT_OPERATOR_HANDLERS.items()
        }
        self.stream_executor = ContentStreamExecutor(self)

    def create_lexer(self, data: bytes | memoryview) -> PdfLexer:
        """Create a content parser carrying the selected document semantics.

        A supplied context applies to every nested stream. Without one, retain
        any context explicitly supplied by a custom lexer factory.
        """
        lexer = self.lexer_factory(data)
        if self.semantic_context is not None:
            try:
                lexer.semantic_context = self.semantic_context
            except BaseException:
                lexer.close()
                raise
        return lexer

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
            pending_clip_rule=self.internal_pending_clip_rule,
        )

    def restore_stream_state(self, state: StreamState) -> None:
        self.resources = state.resources
        self.text_matrix = state.text_matrix
        self.line_matrix = state.line_matrix
        self.graphics_stack_floor = state.graphics_stack_floor
        self.xobject_depth = state.xobject_depth
        self.compatibility_depth = state.compatibility_depth
        self.internal_pending_clip_rule = state.pending_clip_rule
        while len(self.stack) > state.graphics_stack_len:
            self.pop_graphics_save()
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.graphics = copy(state.graphics_state)

    def execute_operation(
        self, name: str, operands: ContentOperands, depth: int
    ) -> ContentStreamFrame | None:
        """Validate before mutation, then execute in the active stream's scope."""
        if name not in CONTENT_OPERATOR_HANDLERS:
            if self.compatibility_depth:
                return None
            raise PdfParseError(f"unknown content operator: {name}")
        override = self.operator_overrides.get(name)
        handler = override or self.internal_default_handlers.get(name)
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
            self.internal_validate_color_operation(name, operands)
        return handler(operands, depth)

    def internal_validate_color_operation(self, name: str, operands: ContentOperands) -> None:
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
            raise PdfParseError("color space requires SCN or scn")
        if space.kind == "Pattern":
            if not operands or not isinstance(operands[-1], PdfName):
                raise PdfParseError("Pattern color requires a pattern name")
            if space.base is None:
                if len(operands) != 1:
                    raise PdfParseError("invalid color component operands")
                return ()
            return self.normalize_color_components(space.base, operands[:-1])
        return self.normalize_color_components(space, operands)

    def lookup_page_resource(self, category: str, name: str) -> object:
        """Return the raw selected entry; its consumer chooses how far to resolve it."""
        entries = self.resolve_resources(self.resources.get(category))
        return entries.get(name) if entries is not None else None

    def resolve_resources(self, value: object) -> PdfDict | None:
        return resolve_resource_dict(value, self.resolver)

    def matrix_operand(self, value: object, context: str) -> Matrix:
        """Resolve an optional six-number matrix; readers may recover by context."""
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
        resolved_font = self.resolver.resolve_font_dict(cast(PdfDict, font))
        decoder = self.font_provider(
            cast(dict[str, Any], resolved_font), cast(dict[str, Any], self.resources)
        )
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
            raise PdfParseError("XObject operand must be a name")
        raw_xobj = self.lookup_page_resource("XObject", name)
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.resolver.resolve(raw_xobj)
        if not isinstance(xobj, PdfStream):
            raise PdfParseError("XObject resource must be a stream")
        xobj_dict = xobj.dictionary
        subtype = self.resolver.resolve_name(xobj_dict.get("Subtype"))
        if subtype == "Image":
            self.sink.paint_image(self, xobj)
            return None
        if subtype != "Form":
            raise PdfParseError("unsupported XObject subtype")
        return self.append_form_xobject(xobj, depth, stream_key=stream_key)

    def resolve_form_resources(self, value: object) -> PdfDict:
        """Resolve Form resources, inheriting only an absent dictionary."""
        resources = self.resolve_resources(value)
        return self.resources if resources is None else resources

    def resolve_form_bbox(self, value: object) -> tuple[float, float, float, float] | None:
        """Require a Form BBox; readers may return None for a missing box."""
        bbox = self.resolver.resolve_box(value)
        if bbox is None:
            raise PdfParseError("Form XObject requires a BBox")
        return bbox

    def append_form_xobject(
        self, xobj: PdfStream, depth: int, *, stream_key: StreamKey | None = None
    ) -> ContentStreamFrame | None:
        """Queue a Form using its transparency, resources, matrix, and bounds.

        Callers select the Form stream and retain its source reference key.
        Reader extensions own resource and BBox recovery through the resolution
        methods; the state transition and frame construction remain shared.
        """
        xobj_dict = xobj.dictionary
        group_alpha = None
        group = xobj_dict.get("Group")
        if group is not None:
            group_dict = self.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.resolver.resolve_name(group_dict.get("S")) == "Transparency"
            ):
                # ISO 32000-1 Table 147 and 11.6.6: the invoking graphics state
                # supplies group alpha and blend mode. An isolated group has a
                # transparent backdrop even at full opacity.
                blend = self.graphics.blend_mode
                isolated = self.resolver.resolve(group_dict.get("I")) is True
                if (
                    isolated
                    or self.graphics.fill_opacity < 1.0
                    or (blend is not None and blend != "Normal")
                ):
                    group_alpha = max(0.0, min(1.0, self.graphics.fill_opacity))
        resources = self.resolve_form_resources(xobj_dict.get("Resources"))
        xobj_matrix = xobj_dict.get("Matrix")
        nested_ctm = self.matrix_operand(xobj_matrix, "form").multiply(self.graphics.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.resolve_form_bbox(raw_form_bbox)
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        return self.stream_executor.queue(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            form_bbox_operand=raw_form_bbox,
            group_alpha=group_alpha,
            stream_key=stream_key,
        )

    def append_text(self, data: bytes | memoryview, *, decoder: FontService | None = None) -> None:
        """Decode bytes once, then execute their glyphs and text advance."""
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
        """Execute decoded text, retaining painted glyphs even without Unicode."""
        if decoder.is_type3 and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self.internal_render_type3_glyphs(data, decoder)
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
        te, tf = self.text_matrix.e, self.text_matrix.f
        ta, tb, tc, td = (
            self.text_matrix.a,
            self.text_matrix.b,
            self.text_matrix.c,
            self.text_matrix.d,
        )
        if text:
            self.sink.show_text(self, text, data, glyphs, decoder, adv_x, adv_y)
        self.text_matrix = self.text_matrix._replace(
            e=te + adv_x * ta + adv_y * tc, f=tf + adv_x * tb + adv_y * td
        )
        self.sink.text_boundary(self, "shown")

    def internal_render_type3_glyphs(self, data: bytes | memoryview, decoder: FontService) -> None:
        # ISO 32000-1 9.3.6: "Only a value of 3 for text rendering mode shall
        # have any effect on text displayed in a Type 3 font", and Table 106
        # makes mode 3 invisible. Mode 7 deliberately still paints here -- for a
        # Type 3 font the clause says only mode 3 has an effect, unlike the
        # simple-font case where 7 also adds no marks.
        if self.graphics.render_mode == 3:
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
                # ISO 32000-1 9.6.5: when the glyph description begins, the CTM
                # is "the concatenation of the font matrix ... and the text space
                # that was in effect at the time the text-showing operator was
                # invoked". Text space is Trm from 9.4.4 NOTE 2:
                #
                #   Trm = [Tfs x Th, 0, 0; 0, Tfs, 0; 0, Trise, 1] x Tm x CTM
                #
                # `multiply` applies the receiver first, so the font matrix has
                # to lead. It was trailing, and the Tfs/Th/Trise factor was
                # missing entirely, which left every Type 3 glyph painted at
                # FontMatrix scale near the origin and independent of font size.
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
                self.type3_uncolored = False
                try:
                    self.stream_executor.consume(
                        char_proc, resources, glyph_ctm, self.xobject_depth + 1
                    )
                finally:
                    self.type3_uncolored = previous_type3_uncolored

            advance_x, advance_y = decoder.glyph_advance_vector(
                code,
                font_size=self.graphics.font_size,
                char_space=self.graphics.char_space,
                word_space=self.graphics.word_space,
                horizontal_scale=self.graphics.horizontal_scale,
                encoded_space=code == 32,
            )
            tm = self.text_matrix
            self.text_matrix = tm._replace(
                e=tm.e + (advance_x * tm.a + advance_y * tm.c),
                f=tm.f + (advance_x * tm.b + advance_y * tm.d),
            )

    def tj_array_extra_bytes(self, item: object) -> bytes:
        """Reject a TJ entry outside the exact PDF string and number types.

        Parsing extensions may override this method to supply bytes for an
        otherwise unsupported entry. It is called during array execution,
        preserving the order of text emission, adjustments, and failures.
        """
        raise PdfParseError("TJ array entries must be strings or numbers")

    def append_tj_array(self, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            raise PdfParseError("TJ requires an array")
        if not array:
            return
        pending_bytes = bytearray()

        decoder = (
            self.graphics.current_decoder
            if self.graphics.current_decoder is not None
            else self.get_decoder()
        )
        is_vert = decoder.is_vertical

        te, tf = self.text_matrix.e, self.text_matrix.f
        ta, tb, tc, td = (
            self.text_matrix.a,
            self.text_matrix.b,
            self.text_matrix.c,
            self.text_matrix.d,
        )
        for item in array:
            t = type(item)
            if t is PdfString:
                pending_bytes.extend(item.data)
            elif t is bytes:
                pending_bytes.extend(item)
            elif t is int or t is float:
                if pending_bytes:
                    self.text_matrix = self.text_matrix._replace(e=te, f=tf)
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
            self.text_matrix = self.text_matrix._replace(e=te, f=tf)
            self.append_text(data=bytes(pending_bytes), decoder=decoder)
            te, tf = self.text_matrix.e, self.text_matrix.f

        self.text_matrix = self.text_matrix._replace(e=te, f=tf)

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

    def internal_begin_text(self) -> None:
        self.text_matrix = self.line_matrix = IDENTITY_MATRIX

    def op_ET(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "end")

    def move_text(self, tx: float, ty: float) -> None:
        self.sink.text_boundary(self, "move")
        # Keep the affine operation order: exact layout grouping can hinge on
        # the final ULP at a character-margin boundary.
        lm = self.line_matrix
        e = tx * lm.a + ty * lm.c + lm.e
        f = tx * lm.b + ty * lm.d + lm.f
        self.text_matrix = self.text_matrix._replace(e=e, f=f)
        self.line_matrix = lm._replace(e=e, f=f)

    def op_BT(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "begin")
        self.internal_begin_text()

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
        """Resolve a Tf name; parsing extensions may return None to skip selection."""
        name = self.resolver.resolve_name(value)
        if name is None:
            raise PdfParseError("Tf requires a font name")
        return name

    def parse_font_size(self, value: object) -> float | None:
        """Parse a Tf size; parsing extensions may return None to skip selection."""
        try:
            return self.as_float(value)
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error

    def op_Tf(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 2:
            raise PdfParseError("Tf requires two operands")
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
            self.sink.paint_inline_image(self, cast("InlineImage", operands[0]))

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        # A run owns one marked-content context. Finish neighboring text before
        # changing that context so ActualText cannot replace unrelated glyphs.
        self.sink.text_boundary(self, "marked")
        tag = self.resolver.resolve_name(operands[0]) if operands else None
        layer: str | None = None
        actual_text: str | None = None
        mcid: int | None = None
        if len(operands) >= 2:
            properties = operands[1]
            if tag == "OC":
                layer = self.resolve_marked_content_layer(properties)
            # ActualText and MCID both live in this dictionary; resolving it
            # once can mean one fewer page-resource lookup per BDC.
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
            raise PdfParseError("unmatched EMC operator")
        self.sink.end_marked_content(self, self.marked_content_stack.pop())
        self.sink.text_boundary(self, "marked")

    def op_G(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceGray", 1, stroke=True)

    def op_RG(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceRGB", 3, stroke=True)

    def op_K(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceCMYK", 4, stroke=True)

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
            raise PdfParseError("d requires two operands")
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            if not isinstance(array_obj, (list, tuple)):
                raise PdfParseError("dash pattern must be an array")
            dash_array = tuple(self.as_float(value) for value in array_obj)
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error
        self.graphics.dash_pattern = (dash_array, phase)

    def op_m(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        x, y = values
        self.current_path.move_to(x, y)
        self.current_point = self.subpath_start = (x, y)

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None:
            raise PdfParseError("path operator has no current point")
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
            raise PdfParseError("path operator has no current point")
        if (values := self.as_floats(operands, 4)) is None:
            return
        x0, y0 = self.current_point
        x2, y2, x3, y3 = values
        self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None:
            raise PdfParseError("path operator has no current point")
        if (values := self.as_floats(operands, 4)) is None:
            return
        x1, y1, x3, y3 = values
        # `y` doubles the endpoint as the second control point, unlike `v`,
        # which uses the current point as the first one.
        self.append_cubic_curve(x1, y1, x3, y3, x3, y3)

    def internal_close_current_subpath(self) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            self.current_path.close()

    def internal_complete_path(
        self, kind: str | None, fill_rule: str = "nonzero", *, close: bool = False
    ) -> None:
        """Paint, install the pending clip, and discard the completed path.

        ISO 32000-1 8.5.4: W/W* affect the clipping path only after the path
        has been painted. The n operator completes a path without painting it.
        """
        if close:
            self.internal_close_current_subpath()
        if kind is not None:
            self.sink.paint_path(self, self.current_path, kind, fill_rule)
        if self.internal_pending_clip_rule is not None:
            self.sink.clip_path(self, self.current_path, self.internal_pending_clip_rule)
        self.current_path = PdfPath()
        self.current_point = None
        self.subpath_start = None
        self.internal_pending_clip_rule = None

    def op_paint_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("stroke")

    def op_paint_close_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("stroke", close=True)

    def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fill", "nonzero")

    def op_paint_fill_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fill", "evenodd")

    def op_paint_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fillstroke", "nonzero")

    def op_paint_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fillstroke", "evenodd")

    def op_paint_close_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fillstroke", "nonzero", close=True)

    def op_paint_close_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path("fillstroke", "evenodd", close=True)

    def op_paint_clear(self, operands: ContentOperands, depth: int) -> None:
        self.internal_complete_path(None)

    def op_W(self, operands: ContentOperands, depth: int) -> None:
        self.internal_pending_clip_rule = "nonzero"

    def op_W_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_pending_clip_rule = "evenodd"

    def internal_set_device_color(
        self, operands: ContentOperands, color_space: str, count: int, *, stroke: bool
    ) -> None:
        if self.type3_uncolored or len(operands) < count:
            return
        space = {"DeviceGray": DEVICE_GRAY, "DeviceRGB": DEVICE_RGB, "DeviceCMYK": DEVICE_CMYK}[
            color_space
        ]
        normalized = self.normalize_color_components(space, operands[:count])
        if normalized is None:
            return
        if stroke:
            self.graphics.stroke_space = space
            self.graphics.stroke_color = normalized
            self.graphics.stroke_pattern = None
        else:
            self.graphics.fill_space = space
            self.graphics.fill_color = normalized
            self.graphics.fill_pattern = None

    def resolve_color_space(self, name_obj: Any) -> ColorSpace:
        name = self.resolver.resolve_name(name_obj)
        if name is None:
            raise PdfParseError("color space operand must be a name")
        value = (
            name
            if name in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}
            else self.resolver.deep_resolve(self.lookup_page_resource("ColorSpace", name))
        )
        if value is None:
            raise PdfParseError("missing color space resource")
        try:
            return parse_color_space(value)
        except (ValueError, TypeError) as error:
            raise PdfParseError(str(error)) from error

    def normalize_color_components(
        self, spec: ColorSpace, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        """Normalize one PDF color; readers may recover malformed components here."""
        try:
            return normalize_color_components(spec, components)
        except ValueError as error:
            raise PdfParseError(str(error)) from error

    def initial_color_components(
        self, spec: ColorSpace, *, stroke: bool
    ) -> tuple[float, ...] | None:
        """Initialize a selected space; readers may retain color for invalid spaces."""
        try:
            return initial_color_components(spec)
        except ValueError as error:
            raise PdfParseError(str(error)) from error

    def internal_set_color_space(self, operands: ContentOperands, *, stroke: bool) -> None:
        if self.type3_uncolored:
            # 9.6.5.2: every colour operator is ignored inside an uncoloured
            # Type 3 glyph, `cs`/`CS` included. The colour setters already
            # refuse to move the colour, so without this the glyph would carry
            # a colour space describing a colour it was not allowed to set.
            return
        if operands:
            space = self.resolve_color_space(operands[0])
            color = self.initial_color_components(space, stroke=stroke)
            if stroke:
                self.graphics.stroke_space = space
                self.graphics.stroke_color = color
                self.graphics.stroke_pattern = None
            else:
                self.graphics.fill_space = space
                self.graphics.fill_color = color
                self.graphics.fill_pattern = None

    def op_CS(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=True)

    def op_cs(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color_space(operands, stroke=False)

    def op_SC(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=False)

    def internal_set_color(
        self, operands: ContentOperands, *, stroke: bool, allow_pattern: bool
    ) -> None:
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
            if stroke:
                self.graphics.stroke_pattern = pattern
                self.graphics.stroke_color = color
            else:
                self.graphics.fill_pattern = pattern
                self.graphics.fill_color = color
        elif stroke:
            self.graphics.stroke_color = normalized
            self.graphics.stroke_pattern = None
        else:
            self.graphics.fill_color = normalized
            self.graphics.fill_pattern = None

    def op_SCN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=True)

    def op_sc(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=False)

    def op_scN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=True)

    def op_i(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.graphics.flatness = max(0.0, min(100.0, values[0]))

    def op_ri(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        value = self.named_value(operands[0])
        if isinstance(value, str):
            self.graphics.render_intent = value

    def op_MP(self, operands: ContentOperands, depth: int) -> None:
        # A marked-content point is not a scope. Only BMC/BDC push and EMC pops.
        return

    def op_DP(self, operands: ContentOperands, depth: int) -> None:
        # A property-bearing marked-content point likewise has no lasting state.
        return

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        """Resolve a PDF name, or a text string where the grammar allows it."""
        name = self.resolver.resolve_name(value)
        if name is not None or not allow_text:
            return name
        return self.resolver.resolve_str(value)

    def resolve_marked_content_properties(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        resolved = self.resolver.resolve(value)
        if isinstance(resolved, dict):
            return cast("dict[str, Any]", resolved)
        name = self.resolver.resolve_name(value)
        if not name:
            return None
        props = self.resolver.resolve(self.lookup_page_resource("Properties", name))
        return cast("dict[str, Any]", props) if isinstance(props, dict) else None

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
        """Observe BX after the validated executor has entered its scope."""

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        """Observe EX after the validated executor has left its scope."""

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            raise PdfParseError("resource operator requires a name")
        name = self.resolver.resolve_name(operands[0])
        if not name:
            raise PdfParseError("resource operator requires a name")
        shading = self.resolver.resolve_dict(self.lookup_page_resource("Shading", name))
        if not isinstance(shading, dict):
            raise PdfParseError("missing shading resource")
        self.sink.paint_shading(self, shading)

    @staticmethod
    def as_float(value: Any) -> float:
        if type(value) in (float, int):
            parsed = parse_float(value, default=None)
            if parsed is not None:
                return parsed
        raise PdfParseError("numeric operand must be a PDF number")

    def as_floats(self, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        """Return numeric operands, raising without changing state if invalid."""
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
        resolved = self.resolver.resolve_dict(self.lookup_page_resource("ExtGState", name))
        if not isinstance(resolved, dict):
            return None
        return cast("dict[str, Any]", resolved)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.stack.append(copy(self.graphics))
        self.sink.save_graphics(self)

    def pop_graphics_save(self) -> GraphicsState:
        saved = self.stack.pop()
        self.sink.restore_graphics(self)
        return saved

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if len(self.stack) <= self.graphics_stack_floor:
            raise PdfParseError("unmatched Q operator")
        self.graphics = self.pop_graphics_save()

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        self.graphics.ctm = Matrix(*values).multiply(self.graphics.ctm)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceGray", 1, stroke=False)

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceRGB", 3, stroke=False)

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceCMYK", 4, stroke=False)

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            raise PdfParseError("resource operator requires a name")
        name = self.resolver.resolve_name(operands[0])
        if not name:
            raise PdfParseError("resource operator requires a name")
        extgstate = self.resolve_extgstate(name)
        if extgstate is None:
            raise PdfParseError("missing ExtGState resource")
        try:
            self.apply_extgstate(extgstate)
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error

    def apply_extgstate(self, extgstate: dict[str, Any]) -> None:
        """Apply supported fields in order through the state's coercion hooks.

        Earlier changes remain applied if a later field raises. Reader callers
        own resource recovery and exception handling around this shared step.
        """
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

    def resolve_pattern_resource(self, name_operand: object) -> tuple[object, PdfDict] | None:
        """Look up a selected pattern source and its dictionary without decoding it.

        An absent name or dictionary returns None; the selection caller decides
        whether to reject it. Resolver failures propagate to the caller.
        """
        pattern_name = self.resolver.resolve_name(name_operand)
        if not pattern_name:
            return None
        pattern = self.resolver.resolve(self.lookup_page_resource("Pattern", pattern_name))
        pattern_dict: PdfDict | None
        if isinstance(pattern, PdfStream):
            pattern_dict = cast(PdfDict, pattern.dictionary)
        else:
            pattern_dict = self.resolver.resolve_dict(pattern) if pattern is not None else None
        return (pattern, pattern_dict) if isinstance(pattern_dict, dict) else None

    def resolve_pattern_color(
        self, pattern_name: object, *, space: ColorSpace, base_components: tuple[float, ...]
    ) -> PatternPaint | None:
        resource = self.resolve_pattern_resource(pattern_name)
        if resource is None:
            raise PdfParseError("invalid pattern resource or operands")
        pattern, pattern_dict = resource
        pattern_type = self.resolver.resolve_int(pattern_dict.get("PatternType"))
        if pattern_type == 2:
            if space.base is not None:
                raise PdfParseError("shading pattern requires a colored Pattern space")
            shading: object = pattern_dict.get("Shading")
            shading = self.resolver.resolve(shading)
            shading_dict = self.resolver.resolve_dict(shading) if shading is not None else None
            if not isinstance(shading_dict, dict):
                raise PdfParseError("invalid pattern resource or operands")
            return ShadingPattern(dict(shading_dict))
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            raise PdfParseError("invalid pattern resource or operands")
        paint_type = self.resolver.resolve_int(pattern_dict.get("PaintType"))
        if paint_type not in {1, 2}:
            raise PdfParseError("invalid pattern resource or operands")
        base_spec = space.base
        if (paint_type == 2) != (base_spec is not None):
            raise PdfParseError("pattern PaintType does not match its color space")
        base_color = base_components if paint_type == 2 else None
        bbox = self.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            raise PdfParseError("invalid pattern resource or operands")
        x_step = self.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            raise PdfParseError("invalid pattern resource or operands")
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
        )


__all__ = ("ContentInterpreter",)
