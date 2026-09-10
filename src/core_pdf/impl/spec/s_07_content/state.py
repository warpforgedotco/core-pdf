# SPDX-License-Identifier: AGPL-3.0-only
"""PDF content operators and their graphics/text state transitions."""

from __future__ import annotations

import operator
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl._impl.model.geometry import transform_bbox
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.events import ContentSink
from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    ContentRecovery,
    OperationHandler,
)
from core_pdf.impl.spec.s_07_content.paths import PdfPath
from core_pdf.impl.spec.s_07_content.patterns import PatternPaint, ShadingPattern, TilingPattern
from core_pdf.impl.spec.s_07_content.soft_masks import SoftMaskSelection
from core_pdf.impl.spec.s_07_content.stream_execution import ContentStreamExecutor
from core_pdf.impl.spec.s_07_content.stream_state import (
    GRAPHICS_STATE_FIELDS,
    STREAM_STATE_MIRRORED,
    GraphicsSave,
    StreamState,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float_strict,
    parse_int_strict,
)
from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.spec.s_09_fonts.service import DecodedFontGlyph as DecodedGlyph
from core_pdf.impl.spec.s_09_fonts.service import FontProvider
from core_pdf.impl.spec.s_09_fonts.service import FontService as FontDecoder
from core_pdf.impl.types import PdfReference, PdfString, Rectangle

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.inline_images import InlineImage

internal_capture_graphics_state = operator.attrgetter(*GRAPHICS_STATE_FIELDS)
internal_NON_PAINTING_RENDER_MODES = frozenset({3, 7})


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> PdfValueResolver: ...


class TextState:
    document: TextDocument

    current_path: PdfPath

    current_point: tuple[float, float] | None

    subpath_start: tuple[float, float] | None

    stack: list[GraphicsSave]

    graphics_stack_floor: int

    fill_color: tuple[float, ...] | None

    fill_pattern: PatternPaint | None

    fill_opacity: float

    stroke_color: tuple[float, ...] | None

    stroke_pattern: PatternPaint | None

    stroke_opacity: float

    blend_mode: str | None

    soft_mask: SoftMaskSelection | None

    flatness: int

    render_intent: str | None

    fill_color_space: str

    fill_color_spec: ImageColorSpec | None

    fill_color_components: tuple[float, ...] | None

    stroke_color_space: str

    stroke_color_spec: ImageColorSpec | None

    stroke_color_components: tuple[float, ...] | None

    dash_pattern: tuple[list[float], float]

    font_operand: object

    font_size_operand: object

    font_widths: tuple[float, ...] | None

    current_font: str | None

    current_decoder: FontDecoder | None

    current_decoder_resources_id: int | None

    marked_content_stack: list[MarkedContentEntry]

    type3_uncolored: bool

    resources: PdfDict

    op_handlers: dict[str, OperationHandler]

    def __init__(
        self,
        document: TextDocument,
        sink: ContentSink,
        font_provider: FontProvider,
        recovery: ContentRecovery | None = None,
        lexer_factory: Callable[[bytes | memoryview], PdfLexer] = PdfLexer,
    ):
        self.document = document
        self.sink = sink
        self.font_provider = font_provider
        self.recovery = recovery
        self.lexer_factory = lexer_factory

        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = (
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        self.fill_color = (0.0,)
        self.fill_pattern = None
        self.fill_opacity = 1.0
        self.stroke_color = (0.0, 0.0, 0.0)
        self.stroke_pattern = None
        self.stroke_opacity = 1.0
        self.blend_mode = None
        self.soft_mask = None
        self.flatness = 0
        self.render_intent = None
        self.fill_color_space = "DeviceGray"
        self.fill_color_spec = None
        self.stroke_color_spec = None
        self.fill_color_components = None
        self.stroke_color_components = None
        self.stroke_color_space = "DeviceGray"
        self.line_width = 1.0
        self.line_cap = 0
        self.line_join = 0
        self.miter_limit = 10.0
        self.dash_pattern = ([], 0.0)
        self.stack = []
        self.graphics_stack_floor = 0
        self.current_path = PdfPath()
        self.current_point = None
        self.subpath_start = None
        self.font_size = 0.0
        self.font_operand = None
        self.font_size_operand = None
        self.horizontal_scale = 100.0
        self.char_space = 0.0
        self.word_space = 0.0
        self.font_scale = self.font_size / 1000.0
        self.font_ascent = 0.0
        self.font_descent = 0.0
        self.font_space_width = 0.0
        self.font_widths = None
        self.text_advance_scale = self.font_size * self.horizontal_scale / 100000.0
        self.char_space_scale = 0.0
        self.word_space_scale = 0.0
        self.rise = 0.0
        self.leading = 0.0
        self.render_mode = 0
        self.current_font = None
        self.current_decoder = None
        self.current_decoder_resources_id = None
        self.xobject_depth = 0
        self.compatibility_depth = 0
        self.marked_content_stack = []
        self.type3_uncolored = False
        self.resources = {}
        self.resources_id = 0
        self.op_handlers = {
            name: getattr(self, handler) for name, handler in CONTENT_OPERATOR_HANDLERS.items()
        }

        self.combined_A = 1.0
        self.combined_B = 0.0
        self.combined_C = 0.0
        self.combined_D = 1.0
        self.stream_executor = ContentStreamExecutor(self)

    @property
    def ctm(self) -> Matrix:
        return Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)

    @ctm.setter
    def ctm(self, val: Matrix) -> None:
        self.ca, self.cb, self.cc, self.cd, self.ce, self.cf = val
        self.update_combined()

    @property
    def text_matrix(self) -> Matrix:
        return Matrix(self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f)

    @text_matrix.setter
    def text_matrix(self, val: Matrix) -> None:
        self.tm_a, self.tm_b, self.tm_c, self.tm_d, self.tm_e, self.tm_f = val
        self.update_combined()

    @property
    def line_matrix(self) -> Matrix:
        return Matrix(self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f)

    @line_matrix.setter
    def line_matrix(self, val: Matrix) -> None:
        self.lm_a, self.lm_b, self.lm_c, self.lm_d, self.lm_e, self.lm_f = val

    def update_combined(self) -> None:
        combined = self.text_matrix.multiply(self.ctm)
        self.combined_A = combined.a
        self.combined_B = combined.b
        self.combined_C = combined.c
        self.combined_D = combined.d

    def append_cubic_curve(
        self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> None:
        if self.current_point is None:
            self.current_point = (x3, y3)
            return
        x0, y0 = self.current_point
        self.current_path.cubic_to((x0, y0, x1, y1, x2, y2, x3, y3), self.ctm, float(self.flatness))
        self.current_point = (x3, y3)

    def update_text_scales(self) -> None:
        fs = self.font_size
        self.font_scale = fs / 1000.0
        self.text_advance_scale = fs * self.horizontal_scale / 100000.0
        if fs:
            self.char_space_scale = self.char_space * 1000.0 / fs
            self.word_space_scale = self.word_space * 1000.0 / fs
        else:
            self.char_space_scale = 0.0
            self.word_space_scale = 0.0

    def update_font_metrics(self) -> None:
        decoder = self.current_decoder
        if decoder is None:
            self.font_ascent = 0.0
            self.font_descent = 0.0
            self.font_space_width = 0.0
            self.font_widths = None
            return
        self.font_ascent = decoder.ascent * self.font_scale
        self.font_descent = decoder.descent * self.font_scale
        self.font_space_width = decoder.glyph_width(32) * self.font_size * 0.001
        self.font_widths = decoder.fast_widths

    def capture_stream_state(self) -> StreamState:
        return StreamState(
            graphics_state=internal_capture_graphics_state(self),
            graphics_stack_len=len(self.stack),
            marked_content_stack_len=len(self.marked_content_stack),
            **{name: getattr(self, name) for name in STREAM_STATE_MIRRORED},
        )

    def restore_stream_state(self, state: StreamState) -> None:
        for name in STREAM_STATE_MIRRORED:
            setattr(self, name, getattr(state, name))
        while len(self.stack) > state.graphics_stack_len:
            self.internal_pop_graphics_save()
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.restore_graphics_state(state.graphics_state)

    def restore_graphics_state(self, state: tuple[Any, ...]) -> None:
        for name, value in zip(GRAPHICS_STATE_FIELDS, state, strict=True):
            setattr(self, name, value)
        self.update_combined()
        self.update_text_scales()
        self.update_font_metrics()

    def consume_stream(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        *,
        clip_bbox: Rectangle | None = None,
    ) -> None:
        self.stream_executor.consume(stream, resources, ctm, depth, clip_bbox=clip_bbox)

    def lookup_page_resource(self, category: str, name: str) -> object:
        """Return the raw selected entry; its consumer chooses how far to resolve it."""
        entries = self.resolve_resources(self.resources.get(category))
        return entries.get(name) if entries is not None else None

    def resolve_resources(self, value: object) -> PdfDict | None:
        return resolve_resource_dict(
            value,
            self.document.resolver,
            on_invalid=self.recovery.invalid_resources if self.recovery is not None else None,
        )

    def internal_handle_error(self, error: Exception, context: str) -> None:
        if self.recovery is not None:
            self.recovery.handle_error(error, context)
            return
        if isinstance(error, PdfParseError):
            raise error
        raise PdfParseError(str(error)) from error

    def internal_matrix_operand(self, value: object, context: str) -> Matrix:
        if value is None:
            return IDENTITY_MATRIX
        try:
            return Matrix.from_operand(value)
        except ValueError:
            if self.recovery is not None:
                return self.recovery.matrix_operand(value, context)
            raise

    def decode_operand(
        self, operand: object, decoder: FontDecoder
    ) -> tuple[str, bytes, tuple[DecodedGlyph, ...]]:
        text: str | None
        if type(operand) is PdfString:
            data, text = operand.data, None
        elif type(operand) is bytes:
            data, text = operand, None
        elif type(operand) is str:
            data, text = operand.encode("latin-1", "replace"), operand
        else:
            text = self.document.resolver.resolve_str(operand)
            if text is None:
                return "", b"", ()
            data = text.encode("latin-1", "replace")

        glyphs = decoder.decode_glyphs(data if isinstance(data, bytes) else bytes(data))
        if text is None:
            text = "".join([glyph.unicode for glyph in glyphs])
        return text, data, glyphs

    def get_decoder(self, *, update_metrics: bool = True) -> "FontDecoder":
        if self.current_decoder is not None:
            return self.current_decoder

        try:
            font_obj_ref = (
                self.lookup_page_resource("Font", self.current_font) if self.current_font else None
            )
        except PdfParseError as error:
            self.internal_handle_error(error, "font-resource")
            font_obj_ref = None
        if font_obj_ref is None:
            return self.font_provider({}, typing.cast(dict[str, Any], self.resources))

        try:
            font_obj = self.document.resolver.resolve(font_obj_ref)
        except PdfParseError as error:
            self.internal_handle_error(error, "font-resolution")
            font_obj = None
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        if not isinstance(font_obj, dict):
            decoder = self.font_provider({}, typing.cast(dict[str, Any], self.resources))
            self.current_decoder = decoder
            self.current_decoder_resources_id = self.resources_id
            if update_metrics:
                self.update_font_metrics()
            return decoder

        font_dict = typing.cast(PdfDict, font_obj)
        resolved_font = self.document.resolver.resolve_font_dict(font_dict)
        decoder = self.font_provider(
            typing.cast(dict[str, Any], resolved_font), typing.cast(dict[str, Any], self.resources)
        )
        self.current_decoder = decoder
        self.current_decoder_resources_id = self.resources_id
        if update_metrics:
            self.update_font_metrics()
        return decoder

    def op_Do(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        self.append_xobject(operands[0], depth)

    def append_xobject(self, name_obj: Any, depth: int) -> None:
        name = self.document.resolver.resolve_name(name_obj)
        if not name:
            return
        raw_xobj = self.lookup_page_resource("XObject", name)
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.document.resolver.resolve(raw_xobj)
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = xobj.dictionary
        subtype = self.document.resolver.resolve_name(xobj_dict.get("Subtype"))
        if self.document.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return
        if subtype == "Image":
            self.sink.paint_image(self, xobj)
            return
        if subtype != "Form":
            return
        group_alpha = None
        group = xobj_dict.get("Group")
        if group is not None:
            group_dict = self.document.resolver.resolve_dict(group)
            if (
                isinstance(group_dict, dict)
                and self.document.resolver.resolve_name(group_dict.get("S")) == "Transparency"
            ):
                # PDF 32000-1 Table 147: a transparency group dictionary holds
                # S/CS/I/K and nothing else. The constant alpha and blend mode
                # that composite the finished group into its backdrop come from
                # the graphics state in effect at the `Do` (11.6.6), so reading
                # a /ca off the group dictionary found nothing and dropped the
                # group entirely -- the contents then painted straight onto the
                # page at full opacity in Normal mode, losing the blend.
                #
                # An explicitly isolated group has its own transparent backdrop,
                # even at full opacity: a child's blend must not see the page.
                blend = self.blend_mode
                isolated = self.document.resolver.resolve(group_dict.get("I")) is True
                if (
                    isolated
                    or self.fill_opacity < 1.0
                    or self.soft_mask is not None
                    or (blend is not None and blend != "Normal")
                ):
                    group_alpha = max(0.0, min(1.0, self.fill_opacity))
        resources = self.resolve_resources(xobj_dict.get("Resources")) or self.resources
        xobj_matrix = xobj_dict.get("Matrix")
        nested_ctm = self.internal_matrix_operand(xobj_matrix, "form").multiply(self.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.document.resolver.resolve_box(raw_form_bbox)
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        self.stream_executor.queue(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            form_bbox_operand=raw_form_bbox,
            group_alpha=group_alpha,
            stream_key=stream_key,
            swallow_parse_errors=self.recovery.skip_form_errors
            if self.recovery is not None
            else False,
        )

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.ca + y * self.cc + self.ce,
            x * self.cb + y * self.cd + self.cf,
        )

    def append_text(
        self,
        operand: Any = None,
        *,
        data: bytes | memoryview | None = None,
        decoder: FontDecoder | None = None,
    ) -> None:
        decoder = decoder if decoder is not None else self.get_decoder()

        glyphs: tuple[DecodedGlyph, ...]
        if data is not None:
            glyphs = decoder.decode_glyphs(data if isinstance(data, bytes) else bytes(data))
            # Keep undecodable painted glyphs in the page program. Native
            # consumers can retain the replacement marker, while legacy
            # facades project the source code as their exact ``(cid:N)``
            # spelling. Dropping them here also lost their cursor advance.
            text = "".join([glyph.unicode for glyph in glyphs])
        else:
            text, data, glyphs = self.decode_operand(operand, decoder)
        rendered_type3_glyphs = False
        if decoder.is_type3 and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self.internal_render_type3_glyphs(data, decoder)
            rendered_type3_glyphs = True
            self.text_matrix = text_matrix
            self.line_matrix = line_matrix
        if not text:
            if data and rendered_type3_glyphs:
                adv_x, adv_y = decoder.text_advance_vector(
                    data,
                    font_size=self.font_size,
                    char_space=self.char_space,
                    word_space=self.word_space,
                    horizontal_scale=self.horizontal_scale,
                    glyphs=glyphs,
                )
                te, tf = self.tm_e, self.tm_f
                ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
                self.tm_e = te + adv_x * ta + adv_y * tc
                self.tm_f = tf + adv_x * tb + adv_y * td
                self.sink.text_boundary(self, "shown")
            return

        adv_x, adv_y = decoder.text_advance_vector(
            data,
            font_size=self.font_size,
            char_space=self.char_space,
            word_space=self.word_space,
            horizontal_scale=self.horizontal_scale,
            glyphs=glyphs,
        )
        te, tf = self.tm_e, self.tm_f
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        self.sink.show_text(self, text, data, glyphs, decoder, adv_x, adv_y)
        self.tm_e = te + adv_x * ta + adv_y * tc
        self.tm_f = tf + adv_x * tb + adv_y * td
        self.sink.text_boundary(self, "shown")

    def internal_render_type3_glyphs(self, data: bytes | memoryview, decoder: FontDecoder) -> None:
        # ISO 32000-1 9.3.6: "Only a value of 3 for text rendering mode shall
        # have any effect on text displayed in a Type 3 font", and Table 106
        # makes mode 3 invisible. Mode 7 deliberately still paints here -- for a
        # Type 3 font the clause says only mode 3 has an effect, unlike the
        # simple-font case where 7 also adds no marks.
        if self.render_mode == 3:
            return
        font = decoder.font
        char_procs = font.get("CharProcs")
        if not isinstance(char_procs, dict):
            return
        resources = self.resolve_resources(font.get("Resources")) or self.resources
        font_matrix = decoder.font_matrix
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale
        scale = self.text_advance_scale

        for code in data:
            glyph_name = decoder.glyph_name(code)
            char_proc = self.document.resolver.resolve(
                char_procs.get(glyph_name) if glyph_name else None
            )
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
                text_space = Matrix(
                    self.combined_A,
                    self.combined_B,
                    self.combined_C,
                    self.combined_D,
                    self.tm_e * self.ca + self.tm_f * self.cc + self.ce,
                    self.tm_e * self.cb + self.tm_f * self.cd + self.cf,
                )
                font_size = self.font_size
                glyph_ctm = font_matrix.multiply(
                    Matrix(
                        font_size * self.horizontal_scale / 100.0,
                        0.0,
                        0.0,
                        font_size,
                        0.0,
                        self.rise,
                    ).multiply(text_space)
                )
                previous_type3_uncolored = self.type3_uncolored
                self.type3_uncolored = False
                try:
                    self.consume_stream(char_proc, resources, glyph_ctm, self.xobject_depth + 1)
                finally:
                    self.type3_uncolored = previous_type3_uncolored

            total = widths[code] + cs
            if code == 32:
                total += ws
            advance = total * scale
            if decoder.is_vertical:
                self.tm_e += -advance * self.tm_c
                self.tm_f += -advance * self.tm_d
            else:
                self.tm_e += advance * self.tm_a
                self.tm_f += advance * self.tm_b

    def append_tj_array(self, array: Any) -> None:
        if not isinstance(array, (list, tuple)):
            return
        if not array:
            return
        pending_bytes = bytearray()
        scale = self.text_advance_scale

        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        is_vert = decoder.is_vertical

        te, tf = self.tm_e, self.tm_f
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        for item in array:
            t = type(item)
            if t is PdfString:
                pending_bytes.extend(item.data)
            elif t is bytes:
                pending_bytes.extend(item)
            elif t is int or t is float:
                if pending_bytes:
                    self.tm_e, self.tm_f = te, tf
                    self.append_text(data=bytes(pending_bytes), decoder=decoder)
                    te, tf = self.tm_e, self.tm_f
                    pending_bytes.clear()
                adjustment = item * scale
                if is_vert:
                    te -= adjustment * tc
                    tf -= adjustment * td
                else:
                    te -= adjustment * ta
                    tf -= adjustment * tb
            elif t is str:
                pending_bytes.extend(item.encode("latin-1"))

        if pending_bytes:
            self.tm_e, self.tm_f = te, tf
            self.append_text(data=bytes(pending_bytes), decoder=decoder)
            te, tf = self.tm_e, self.tm_f

        self.tm_e, self.tm_f = te, tf

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
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        self.update_combined()

    def op_ET(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "end")

    def internal_move_text(self, tx: float, ty: float) -> None:
        self.sink.text_boundary(self, "move")
        # Preserve the specification's affine operation order. Exact layout
        # grouping can hinge on the final ULP at a character-margin boundary.
        self.tm_e = tx * self.lm_a + ty * self.lm_c + self.lm_e
        self.tm_f = tx * self.lm_b + ty * self.lm_d + self.lm_f
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def internal_show_text(self, operand: ContentOperand) -> None:
        decoder = self.current_decoder if self.current_decoder is not None else self.get_decoder()
        if type(operand) is PdfString:
            self.append_text(
                data=operand.data,
                decoder=decoder,
            )
        else:
            self.append_text(operand, decoder=decoder)

    def op_BT(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "begin")
        self.internal_begin_text()

    def op_T_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_move_text(0.0, -self.leading)

    def op_Td(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        self.internal_move_text(*values)

    def op_TD(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        tx, ty = values
        self.leading = -ty
        self.internal_move_text(tx, ty)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        # Operators consume their operands from the top of the operand stack.
        # A well-formed Tj has exactly one string, but damaged streams sometimes
        # leave older operands before it.  Those older values are not part of
        # the text-showing operation.
        self.internal_show_text(operands[-1])

    def op_TJ(self, operands: ContentOperands, depth: int) -> None:
        if operands:
            self.append_tj_array(operands[0])

    def op_Tm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        a, b, c, d_, e, f = values
        self.sink.text_boundary(self, "matrix")
        self.tm_a = self.lm_a = a
        self.tm_b = self.lm_b = b
        self.tm_c = self.lm_c = c
        self.tm_d = self.lm_d = d_
        self.tm_e = self.lm_e = e
        self.tm_f = self.lm_f = f
        self.update_combined()

    def op_Tf(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        font_operand = operands[0]
        font_size_operand = operands[1]
        decoder_matches_resources = self.current_decoder_resources_id == self.resources_id
        if (
            self.current_decoder is not None
            and decoder_matches_resources
            and font_operand is self.font_operand
        ):
            if font_size_operand is not self.font_size_operand:
                try:
                    font_size = self.as_float(font_size_operand)
                except (TypeError, ValueError) as error:
                    self.internal_handle_error(error, "font-size")
                    return
                if self.font_size != font_size:
                    self.font_size = font_size
                    self.update_text_scales()
                    self.update_font_metrics()
                self.font_size_operand = font_size_operand
            return
        font_name = self.document.resolver.resolve_name(font_operand)
        if font_name is None:
            return
        try:
            font_size = self.as_float(font_size_operand)
        except (TypeError, ValueError) as error:
            self.internal_handle_error(error, "font-size")
            return
        if (
            self.current_font == font_name
            and self.current_decoder is not None
            and decoder_matches_resources
        ):
            if self.font_size != font_size:
                self.font_size = font_size
                self.update_text_scales()
                self.update_font_metrics()
            self.font_operand = font_operand
            self.font_size_operand = font_size_operand
            return
        self.current_font = font_name
        self.font_size = font_size
        self.update_text_scales()
        self.font_operand = font_operand
        self.font_size_operand = font_size_operand
        self.current_decoder = None
        self.current_decoder = self.get_decoder(update_metrics=False)
        self.update_font_metrics()

    def op_TL(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.leading = values[0]

    def op_Tc(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.char_space = values[0]
        self.update_text_scales()

    def op_Tw(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        word_space = values[0]
        if self.word_space == word_space:
            return
        self.word_space = word_space
        self.update_text_scales()

    def op_Tr(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.render_mode = value

    def op_Tz(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.horizontal_scale = values[0]
        self.update_text_scales()

    def op_Ts(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is None:
            return
        self.rise = values[0]

    def op_quote(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        self.internal_move_text(0.0, -self.leading)
        self.sink.text_boundary(self, "quoted")
        self.internal_show_text(operands[0])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 3 or (values := self.as_floats(operands, 2)) is None:
            return
        self.word_space, self.char_space = values
        self.update_text_scales()
        self.internal_move_text(0.0, -self.leading)
        self.sink.text_boundary(self, "quoted")
        self.internal_show_text(operands[2])

    def op_BI(self, operands: ContentOperands, depth: int) -> None:
        if operands and hasattr(operands[0], "dictionary"):
            self.sink.paint_inline_image(self, cast("InlineImage", operands[0]))

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        # A run owns one marked-content context. Finish neighboring text before
        # changing that context so ActualText cannot replace unrelated glyphs.
        self.sink.text_boundary(self, "marked")
        tag = self.document.resolver.resolve_name(operands[0]) if operands else None
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
                resolver = self.document.resolver
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
        if self.marked_content_stack:
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
            self.line_width = max(0.0, values[0])

    def op_J(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.line_cap = value

    def op_j(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.line_join = value

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.miter_limit = max(1.0, values[0])

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 2:
            return
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            dash_array = (
                [self.as_float(value) for value in array_obj]
                if isinstance(array_obj, (list, tuple))
                else []
            )
        except (TypeError, ValueError) as error:
            self.internal_handle_error(error, "dash-pattern")
            return
        self.dash_pattern = (dash_array, phase)

    def op_m(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        x, y = values
        self.current_path.move_to(x, y)
        self.current_point = self.subpath_start = (x, y)

    def op_l(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None or (values := self.as_floats(operands, 2)) is None:
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
        if self.current_point is None or (values := self.as_floats(operands, 4)) is None:
            return
        x0, y0 = self.current_point
        x2, y2, x3, y3 = values
        self.append_cubic_curve(x0, y0, x2, y2, x3, y3)

    def op_y(self, operands: ContentOperands, depth: int) -> None:
        if self.current_point is None or (values := self.as_floats(operands, 4)) is None:
            return
        x1, y1, x3, y3 = values
        # `y` doubles the endpoint as the second control point, unlike `v`,
        # which uses the current point as the first one.
        self.append_cubic_curve(x1, y1, x3, y3, x3, y3)

    def internal_close_current_subpath(self) -> None:
        if self.current_point is not None and self.subpath_start is not None:
            self.current_path.close()

    def internal_end_path(self) -> None:
        """Discard the current point and subpath origin after a painting operator."""
        self.current_point = None
        self.subpath_start = None

    def op_paint_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("stroke")
        self.internal_end_path()

    def op_paint_close_stroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("stroke")
        self.internal_end_path()

    def op_paint_fill(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fill", "nonzero")
        self.internal_end_path()

    def op_paint_fill_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fill", "evenodd")
        self.internal_end_path()

    def op_paint_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fillstroke", "nonzero")
        self.internal_end_path()

    def op_paint_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.flush_drawing("fillstroke", "evenodd")
        self.internal_end_path()

    def op_paint_close_fillstroke(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "nonzero")
        self.internal_end_path()

    def op_paint_close_fillstroke_evenodd(self, operands: ContentOperands, depth: int) -> None:
        self.internal_close_current_subpath()
        self.flush_drawing("fillstroke", "evenodd")
        self.internal_end_path()

    def op_paint_clear(self, operands: ContentOperands, depth: int) -> None:
        self.current_path.clear()
        self.internal_end_path()

    def op_W(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("nonzero")

    def op_W_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("evenodd")

    def normalize_colors(self, *components: Any) -> tuple[float, ...] | None:
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError as error:
                self.internal_handle_error(error, "color-components")
                return None
        if not values:
            return None
        return tuple(values)

    def internal_set_device_color(
        self, operands: ContentOperands, color_space: str, count: int, *, stroke: bool
    ) -> None:
        if self.type3_uncolored or len(operands) < count:
            return
        normalized = self.normalize_color_operands(operands[:count])
        if normalized is None:
            return
        # Device operators select both a space and its components. Leaving a
        # previous Indexed/Separation spec behind misinterprets a later sc/SC.
        if stroke:
            self.stroke_color_space = color_space
            self.stroke_color_spec = None
            self.stroke_color = normalized
            self.stroke_color_components = normalized
            self.stroke_pattern = None
        else:
            self.fill_color_space = color_space
            self.fill_color_spec = None
            self.fill_color = normalized
            self.fill_color_components = normalized
            self.fill_pattern = None

    def normalize_color_operands(self, o: Any) -> tuple[float, ...] | None:
        # Plain numeric operands (the overwhelming majority) clamp directly;
        # anything else -- strings, names, nulls -- goes through the resolver.
        if o and all(type(c) is float or type(c) is int for c in o):
            return tuple(max(0.0, min(1.0, float(c))) for c in o)
        return self.normalize_colors(*o)

    def resolve_color_space(self, name_obj: Any) -> tuple[str, ImageColorSpec | None]:
        """Resolve a cs/CS resource once for both its name and conversion spec."""
        name = self.document.resolver.resolve_name(name_obj)
        if name is None:
            return "DeviceGray", None
        value = self.document.resolver.deep_resolve(self.lookup_page_resource("ColorSpace", name))
        if value is None:
            # An inline device space (`/DeviceRGB cs`) names no resource.
            value = name
        base = value[0] if isinstance(value, (list, tuple)) and value else value
        color_space = normalize_pdf_name(base) or name
        try:
            spec = self.sink.parse_color_space(value)
        except (ValueError, TypeError) as error:
            self.internal_handle_error(error, "color-space")
            spec = None
        return color_space, spec

    def internal_color_from_operands(
        self, operands: Any, spec: ImageColorSpec | None
    ) -> tuple[float, ...] | None:
        """Turn `sc`/`scn` operands into the colour they select.

        Device-space operands are their own components and pass through. An
        Indexed operand is a palette index and a Separation/DeviceN operand is a
        tint (ISO 32000-1 8.6.6.3, 8.6.6.4), so those resolve to sRGB here --
        clamping them to 0..1 and painting them directly rendered a spot colour
        as an inverted grey and ignored the palette entirely.
        """
        if spec is not None and spec.kind in {"Indexed", "Separation", "DeviceN"}:
            values = self.internal_numeric_operands(operands)
            if values is not None:
                converted = self.sink.convert_color(spec, values)
                if converted is not None:
                    return converted
        return self.normalize_color_operands(operands)

    def internal_numeric_operands(self, operands: Any) -> list[float] | None:
        values: list[float] = []
        for operand in operands:
            if type(operand) is float or type(operand) is int:
                values.append(float(operand))
            else:
                return None
        return values or None

    def internal_set_color_space(self, operands: ContentOperands, *, stroke: bool) -> None:
        if self.type3_uncolored:
            # 9.6.5.2: every colour operator is ignored inside an uncoloured
            # Type 3 glyph, `cs`/`CS` included. The colour setters already
            # refuse to move the colour, so without this the glyph would carry
            # a colour space describing a colour it was not allowed to set.
            return
        if operands:
            color_space, spec = self.resolve_color_space(operands[0])
            if stroke:
                self.stroke_color_space = color_space
                self.stroke_color_spec = spec
                self.stroke_color_components = None
            else:
                self.fill_color_space = color_space
                self.fill_color_spec = spec
                self.fill_color_components = None

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
        color_space = self.stroke_color_space if stroke else self.fill_color_space
        if allow_pattern and color_space == "Pattern":
            pattern = self.resolve_pattern_color(operands)
            if stroke:
                self.stroke_pattern = pattern
            else:
                self.fill_pattern = pattern
            if len(operands) > 1:
                normalized = self.internal_color_from_operands(
                    operands[:-1], self.stroke_color_spec if stroke else self.fill_color_spec
                )
                if normalized is not None:
                    numeric = self.internal_numeric_operands(operands[:-1])
                    components = tuple(numeric) if numeric is not None else None
                    if stroke:
                        self.stroke_color = normalized
                        self.stroke_color_components = components
                    else:
                        self.fill_color = normalized
                        self.fill_color_components = components
            return
        normalized = self.internal_color_from_operands(
            operands, self.stroke_color_spec if stroke else self.fill_color_spec
        )
        if normalized is not None:
            numeric = self.internal_numeric_operands(operands)
            components = tuple(numeric) if numeric is not None else None
            if stroke:
                self.stroke_color = normalized
                self.stroke_color_components = components
                self.stroke_pattern = None
            else:
                self.fill_color = normalized
                self.fill_color_components = components
                self.fill_pattern = None

    def op_SCN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=True, allow_pattern=True)

    def op_sc(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=False)

    def op_scN(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_color(operands, stroke=False, allow_pattern=True)

    def op_i(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.flatness = max(0, min(100, int(values[0])))

    def op_ri(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        value = self.sink.named_value(operands[0])
        if isinstance(value, str):
            self.render_intent = value

    def op_MP(self, operands: ContentOperands, depth: int) -> None:
        # A marked-content point is not a scope. Only BMC/BDC push and EMC pops.
        return

    def op_DP(self, operands: ContentOperands, depth: int) -> None:
        # A property-bearing marked-content point likewise has no lasting state.
        return

    def resolve_marked_content_properties(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            return cast("dict[str, Any]", resolved)
        name = self.document.resolver.resolve_name(value)
        if not name:
            return None
        props = self.document.resolver.resolve(self.lookup_page_resource("Properties", name))
        return cast("dict[str, Any]", props) if isinstance(props, dict) else None

    def resolve_marked_content_layer(self, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            oc = resolved.get("OC")
            if oc is not None:
                return self.sink.named_value(oc, allow_text=True)

        return self.sink.named_value(value, allow_text=True)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        shading = self.document.resolver.resolve_dict(self.lookup_page_resource("Shading", name))
        if isinstance(shading, dict):
            self.sink.paint_shading(self, shading)

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
        return parse_float_strict(value, "invalid numeric operand")

    def as_floats(self, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        """The leading `count` numeric operands, with errors delegated to the consumer."""
        if len(operands) < count:
            self.internal_handle_error(PdfParseError("missing numeric operand"), "numeric-operands")
            return None
        try:
            return tuple([self.as_float(operands[i]) for i in range(count)])
        except (TypeError, ValueError) as error:
            self.internal_handle_error(error, "numeric-operands")
            return None

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        return parse_int_strict(value, "invalid numeric operand")

    def as_int_operand(self, operands: ContentOperands) -> int | None:
        """The first integer operand, with errors delegated to the consumer."""
        if not operands:
            self.internal_handle_error(PdfParseError("missing numeric operand"), "integer-operand")
            return None
        try:
            return self.as_int(operands[0])
        except (TypeError, ValueError) as error:
            self.internal_handle_error(error, "integer-operand")
            return None

    def resolve_extgstate(self, name: str) -> dict[str, Any] | None:
        resolved = self.document.resolver.resolve_dict(self.lookup_page_resource("ExtGState", name))
        if not isinstance(resolved, dict):
            return None
        return cast("dict[str, Any]", resolved)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.stack.append(GraphicsSave(internal_capture_graphics_state(self)))
        self.sink.save_graphics(self)

    def internal_pop_graphics_save(self) -> tuple[Any, ...]:
        saved = self.stack.pop()
        self.sink.restore_graphics(self)
        return saved.graphics_state

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if len(self.stack) <= self.graphics_stack_floor:
            return
        self.restore_graphics_state(self.internal_pop_graphics_save())

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        matrix = Matrix(*values)
        self.ctm = matrix.multiply(self.ctm)
        self.sink.concatenate_matrix(self, matrix)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceGray", 1, stroke=False)

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceRGB", 3, stroke=False)

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceCMYK", 4, stroke=False)

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
        if "SMask" in extgstate:
            mask = self.document.resolver.resolve(extgstate["SMask"])
            if isinstance(mask, dict):
                source = self.lookup_page_resource("ExtGState", name)
                source_key = (
                    ("ref", source.object_number, source.generation_number)
                    if isinstance(source, PdfReference)
                    else ("direct", id(source), 0)
                )
                self.soft_mask = SoftMaskSelection(
                    cast(PdfDict, mask), self.ctm, self.resources, source_key
                )
            elif self.document.resolver.resolve_name(mask) == "None":
                self.soft_mask = None
        try:
            fill_opacity = extgstate.get("ca")
            if fill_opacity is not None:
                self.fill_opacity = max(0.0, min(1.0, self.as_float(fill_opacity)))
            stroke_opacity = extgstate.get("CA")
            if stroke_opacity is not None:
                self.stroke_opacity = max(0.0, min(1.0, self.as_float(stroke_opacity)))
            blend_mode = extgstate.get("BM")
            if blend_mode is not None:
                if isinstance(blend_mode, (list, tuple)):
                    blend_mode = blend_mode[0] if blend_mode else None
                if blend_mode is not None:
                    self.blend_mode = self.sink.named_value(blend_mode)
        except (TypeError, ValueError) as error:
            self.internal_handle_error(error, "extended-graphics-state")
            return

    def flush_drawing(self, kind: str, fill_rule: str = "nonzero") -> None:
        self.sink.paint_path(self, self.current_path, kind, fill_rule)
        self.current_path = PdfPath()

    def internal_record_clip(self, fill_rule: str) -> None:
        self.sink.clip_path(self, self.current_path, fill_rule)

    def resolve_pattern_color(self, operands: tuple[Any, ...]) -> PatternPaint | None:
        if not operands:
            return None
        pattern_name = self.document.resolver.resolve_name(operands[-1])
        if not pattern_name:
            return None
        pattern = self.document.resolver.resolve(self.lookup_page_resource("Pattern", pattern_name))
        pattern_dict: PdfDict | None
        if isinstance(pattern, PdfStream):
            pattern_dict = cast(PdfDict, pattern.dictionary)
        else:
            pattern_dict = (
                self.document.resolver.resolve_dict(pattern) if pattern is not None else None
            )
        if not isinstance(pattern_dict, dict):
            return None
        pattern_type = self.document.resolver.resolve_int(pattern_dict.get("PatternType"))
        if pattern_type == 2:
            shading: object = pattern_dict.get("Shading")
            shading = self.document.resolver.resolve(shading)
            shading_dict = (
                self.document.resolver.resolve_dict(shading) if shading is not None else None
            )
            if not isinstance(shading_dict, dict):
                return None
            return ShadingPattern(dict(shading_dict))
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return None
        paint_type = self.document.resolver.resolve_int(pattern_dict.get("PaintType"), 1)
        if paint_type not in {1, 2}:
            return None
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                return None
        bbox = self.document.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            return None
        x_step = self.document.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.document.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            return None
        matrix = self.internal_matrix_operand(
            self.document.resolver.deep_resolve(pattern_dict.get("Matrix")), "pattern"
        )
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
        )
