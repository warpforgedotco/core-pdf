# SPDX-License-Identifier: AGPL-3.0-only
"""PDF content operators and their graphics/text state transitions."""

from __future__ import annotations

import math
import operator
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.events import ContentSink
from core_pdf_spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf_spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    OperationHandler,
    validate_content_operands,
)
from core_pdf_spec.s_07_content.paths import PdfPath
from core_pdf_spec.s_07_content.patterns import PatternPaint, ShadingPattern, TilingPattern
from core_pdf_spec.s_07_content.stream_execution import ContentStreamExecutor
from core_pdf_spec.s_07_content.stream_state import (
    GRAPHICS_STATE_FIELDS,
    STREAM_STATE_MIRRORED,
    GraphicsSave,
    StreamState,
)
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf_spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
)
from core_pdf_spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf_spec.s_08_graphics.color_spec import ImageColorSpec, color_spec_from_value
from core_pdf_spec.s_08_graphics.geometry import transform_bbox
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import DecodedFontGlyph as DecodedGlyph
from core_pdf_spec.s_09_fonts.service import FontProvider
from core_pdf_spec.s_09_fonts.service import FontService as FontDecoder
from core_pdf_spec.types import PdfName, PdfReference, PdfString, Rectangle

if TYPE_CHECKING:
    from core_pdf_spec.s_07_content.inline_images import InlineImage

internal_capture_graphics_state = operator.attrgetter(*GRAPHICS_STATE_FIELDS)
NON_PAINTING_RENDER_MODES = frozenset({3, 7})


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

    flatness: int

    render_intent: str | None

    fill_color_space: str

    fill_color_spec: ImageColorSpec | None

    stroke_color_space: str

    stroke_color_spec: ImageColorSpec | None

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
        lexer_factory: Callable[[bytes | memoryview], PdfLexer] = PdfLexer,
    ):
        self.document = document
        self.sink = sink
        self.font_provider = font_provider
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
        self.stroke_color = (0.0,)
        self.stroke_pattern = None
        self.stroke_opacity = 1.0
        self.blend_mode = None
        self.flatness = 0
        self.render_intent = None
        self.fill_color_space = "DeviceGray"
        self.fill_color_spec = None
        self.stroke_color_spec = None
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
            raise PdfParseError("curve has no current point")
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
            self.pop_graphics_save()
        del self.marked_content_stack[state.marked_content_stack_len :]
        self.restore_graphics_state(state.graphics_state)

    def restore_graphics_state(self, state: tuple[Any, ...]) -> None:
        for name, value in zip(GRAPHICS_STATE_FIELDS, state, strict=True):
            setattr(self, name, value)
        self.update_combined()
        self.update_text_scales()
        self.update_font_metrics()

    def get_operation_handler(self, name: str) -> OperationHandler | None:
        """Validate operands before executing the selected PDF state transition."""
        handler = self.op_handlers.get(name)
        if handler is None:
            return None

        def execute(operands: ContentOperands, depth: int) -> None:
            validate_content_operands(name, operands)
            if name in {"l", "c", "v", "y"} and self.current_point is None:
                raise PdfParseError("path operator has no current point")
            if name == "EMC" and not self.marked_content_stack:
                raise PdfParseError("unmatched EMC operator")
            if name in {"gs", "sh"}:
                resource_name = self.document.resolver.resolve_name(operands[0])
                category = "ExtGState" if name == "gs" else "Shading"
                value = self.document.resolver.resolve(
                    self.lookup_page_resource(category, resource_name or "")
                )
                if not isinstance(value, dict) and not (
                    name == "sh" and isinstance(value, PdfStream)
                ):
                    raise PdfParseError(f"missing or invalid {category} resource")
            if name in {"SC", "SCN", "sc", "scn"} and not self.type3_uncolored:
                stroke = name in {"SC", "SCN"}
                color_space = self.stroke_color_space if stroke else self.fill_color_space
                spec = self.stroke_color_spec if stroke else self.fill_color_spec
                components = operands
                expected: int | None
                if color_space == "Pattern":
                    if (
                        name not in {"SCN", "scn"}
                        or not operands
                        or not isinstance(operands[-1], PdfName)
                    ):
                        raise PdfParseError("Pattern color requires a pattern name")
                    components = operands[:-1]
                    expected = len(components)
                else:
                    expected = {
                        "DeviceGray": 1,
                        "CalGray": 1,
                        "DeviceRGB": 3,
                        "CalRGB": 3,
                        "Lab": 3,
                        "DeviceCMYK": 4,
                        "Indexed": 1,
                        "Separation": 1,
                    }.get(color_space)
                    if expected is None:
                        expected = spec.channels if spec is not None else 1
                if len(components) != expected or any(
                    type(value) not in (int, float) or not math.isfinite(cast(float, value))
                    for value in components
                ):
                    raise PdfParseError("invalid color component operands")
            handler(operands, depth)

        return execute

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
        )

    def matrix_operand(self, value: object, context: str) -> Matrix:
        if value is None:
            return IDENTITY_MATRIX
        return Matrix.from_operand(value)

    def decode_operand(
        self, operand: object, decoder: FontDecoder
    ) -> tuple[str, bytes, tuple[DecodedGlyph, ...]]:
        if not isinstance(operand, PdfString):
            raise PdfParseError("text operand must be a PDF string")
        data = bytes(operand.data)
        glyphs = decoder.decode_glyphs(data)
        return "".join(glyph.unicode for glyph in glyphs), data, glyphs

    def get_decoder(self, *, update_metrics: bool = True) -> FontDecoder:
        if self.current_decoder is not None:
            return self.current_decoder
        if self.current_font is None:
            raise PdfParseError("text operation has no selected font")
        font_reference = self.lookup_page_resource("Font", self.current_font)
        font = self.document.resolver.resolve(font_reference)
        if not isinstance(font, dict):
            raise PdfParseError("font resource must be a dictionary")
        resolved_font = self.document.resolver.resolve_font_dict(cast(PdfDict, font))
        decoder = self.font_provider(
            cast(dict[str, Any], resolved_font), cast(dict[str, Any], self.resources)
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
            raise PdfParseError("XObject operand must be a name")
        raw_xobj = self.lookup_page_resource("XObject", name)
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.document.resolver.resolve(raw_xobj)
        if not isinstance(xobj, PdfStream):
            raise PdfParseError("XObject resource must be a stream")
        xobj_dict = xobj.dictionary
        subtype = self.document.resolver.resolve_name(xobj_dict.get("Subtype"))
        if subtype == "Image":
            self.sink.paint_image(self, xobj)
            return
        if subtype != "Form":
            raise PdfParseError("unsupported XObject subtype")
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
                if isolated or self.fill_opacity < 1.0 or (blend is not None and blend != "Normal"):
                    group_alpha = max(0.0, min(1.0, self.fill_opacity))
        resources = self.resolve_resources(xobj_dict.get("Resources"))
        if resources is None:
            resources = self.resources
        xobj_matrix = xobj_dict.get("Matrix")
        nested_ctm = self.matrix_operand(xobj_matrix, "form").multiply(self.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.document.resolver.resolve_box(raw_form_bbox)
        if form_bbox is None:
            raise PdfParseError("Form XObject requires a BBox")
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
        if decoder.is_type3 and data:
            text_matrix = self.text_matrix
            line_matrix = self.line_matrix
            self.internal_render_type3_glyphs(data, decoder)
            self.text_matrix = text_matrix
            self.line_matrix = line_matrix
        if not text:
            if data:
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
            else:
                pending_bytes.extend(self.tj_array_extra_bytes(item))

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

    def move_text(self, tx: float, ty: float) -> None:
        self.sink.text_boundary(self, "move")
        # Preserve the specification's affine operation order. Exact layout
        # grouping can hinge on the final ULP at a character-margin boundary.
        self.tm_e = tx * self.lm_a + ty * self.lm_c + self.lm_e
        self.tm_f = tx * self.lm_b + ty * self.lm_d + self.lm_f
        self.lm_e = self.tm_e
        self.lm_f = self.tm_f

    def show_text_operand(self, operand: ContentOperand) -> None:
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
        self.move_text(0.0, -self.leading)

    def op_Td(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        self.move_text(*values)

    def op_TD(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 2)) is None:
            return
        tx, ty = values
        self.leading = -ty
        self.move_text(tx, ty)

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 1:
            raise PdfParseError("Tj requires one string")
        self.show_text_operand(operands[0])

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
        if len(operands) != 2:
            raise PdfParseError("Tf requires two operands")
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
                    raise PdfParseError(str(error)) from error
                if self.font_size != font_size:
                    self.font_size = font_size
                    self.update_text_scales()
                    self.update_font_metrics()
                self.font_size_operand = font_size_operand
            return
        font_name = self.document.resolver.resolve_name(font_operand)
        if font_name is None:
            raise PdfParseError("Tf requires a font name")
        try:
            font_size = self.as_float(font_size_operand)
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error
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
        self.move_text(0.0, -self.leading)
        self.sink.text_boundary(self, "quoted")
        self.show_text_operand(operands[0])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 3 or (values := self.as_floats(operands, 2)) is None:
            return
        self.word_space, self.char_space = values
        self.update_text_scales()
        self.move_text(0.0, -self.leading)
        self.sink.text_boundary(self, "quoted")
        self.show_text_operand(operands[2])

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
            self.line_width = values[0]

    def op_J(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.line_cap = value

    def op_j(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.line_join = value

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            if values[0] < 1:
                raise PdfParseError("miter limit must be at least one")
            self.miter_limit = values[0]

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) != 2:
            raise PdfParseError("d requires two operands")
        try:
            phase = self.as_float(operands[1])
            array_obj = operands[0]
            if not isinstance(array_obj, (list, tuple)):
                raise PdfParseError("dash pattern must be an array")
            dash_array = [self.as_float(value) for value in array_obj]
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error
        self.dash_pattern = (dash_array, phase)

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
                raise PdfParseError(str(error)) from error
        if not values:
            raise PdfParseError("color requires components")
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
            self.stroke_pattern = None
        else:
            self.fill_color_space = color_space
            self.fill_color_spec = None
            self.fill_color = normalized
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
            raise PdfParseError("color space operand must be a name")
        value = self.document.resolver.deep_resolve(self.lookup_page_resource("ColorSpace", name))
        if value is None:
            if name not in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}:
                raise PdfParseError("missing color space resource")
            value = name
        base = value[0] if isinstance(value, (list, tuple)) and value else value
        color_space = normalize_pdf_name(base) or name
        try:
            spec = color_spec_from_value(value)
        except (ValueError, TypeError) as error:
            raise PdfParseError(str(error)) from error
        return color_space, spec

    def internal_color_from_operands(
        self, operands: Any, spec: ImageColorSpec | None
    ) -> tuple[float, ...] | None:
        """Retain PDF components; output colour conversion belongs to consumers."""
        values = tuple(self.as_float(value) for value in operands)
        if spec is not None and spec.kind == "Indexed":
            if len(values) != 1:
                raise PdfParseError("Indexed color requires one component")
            return (float(max(0, min(spec.hival, int(values[0] + 0.5)))),)
        if spec is not None and spec.kind == "Lab":
            limits = spec.params.get("Range", [-100, 100, -100, 100])
            if len(values) != 3 or not isinstance(limits, (list, tuple)) or len(limits) != 4:
                raise PdfParseError("invalid Lab color components")
            a_min, a_max, b_min, b_max = (self.as_float(limit) for limit in limits)
            return (
                max(0.0, min(100.0, values[0])),
                max(a_min, min(a_max, values[1])),
                max(b_min, min(b_max, values[2])),
            )
        return self.normalize_color_operands(values)

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
            else:
                self.fill_color_space = color_space
                self.fill_color_spec = spec

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
                    if stroke:
                        self.stroke_color = normalized
                    else:
                        self.fill_color = normalized
            return
        normalized = self.internal_color_from_operands(
            operands, self.stroke_color_spec if stroke else self.fill_color_spec
        )
        if normalized is not None:
            if stroke:
                self.stroke_color = normalized
                self.stroke_pattern = None
            else:
                self.fill_color = normalized
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
        value = self.named_value(operands[0])
        if isinstance(value, str):
            self.render_intent = value

    def op_MP(self, operands: ContentOperands, depth: int) -> None:
        # A marked-content point is not a scope. Only BMC/BDC push and EMC pops.
        return

    def op_DP(self, operands: ContentOperands, depth: int) -> None:
        # A property-bearing marked-content point likewise has no lasting state.
        return

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        """Resolve a PDF name, or a text string where the grammar allows it."""
        name = self.document.resolver.resolve_name(value)
        if name is not None or not allow_text:
            return name
        return self.document.resolver.resolve_str(value)

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
                return self.named_value(oc, allow_text=True)

        return self.named_value(value, allow_text=True)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        if not self.compatibility_depth:
            raise PdfParseError("unmatched EX operator")
        self.compatibility_depth -= 1

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            raise PdfParseError("resource operator requires a name")
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            raise PdfParseError("resource operator requires a name")
        shading = self.document.resolver.resolve_dict(self.lookup_page_resource("Shading", name))
        if not isinstance(shading, dict):
            raise PdfParseError("missing shading resource")
        self.sink.paint_shading(self, shading)

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
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
        resolved = self.document.resolver.resolve_dict(self.lookup_page_resource("ExtGState", name))
        if not isinstance(resolved, dict):
            return None
        return cast("dict[str, Any]", resolved)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.stack.append(GraphicsSave(internal_capture_graphics_state(self)))
        self.sink.save_graphics(self)

    def pop_graphics_save(self) -> tuple[Any, ...]:
        saved = self.stack.pop()
        self.sink.restore_graphics(self)
        return saved.graphics_state

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        if len(self.stack) <= self.graphics_stack_floor:
            raise PdfParseError("unmatched Q operator")
        self.restore_graphics_state(self.pop_graphics_save())

    def op_cm(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 6)) is None:
            return
        self.ctm = Matrix(*values).multiply(self.ctm)

    def op_g(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceGray", 1, stroke=False)

    def op_rg(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceRGB", 3, stroke=False)

    def op_k(self, operands: ContentOperands, depth: int) -> None:
        self.internal_set_device_color(operands, "DeviceCMYK", 4, stroke=False)

    def op_gs(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            raise PdfParseError("resource operator requires a name")
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            raise PdfParseError("resource operator requires a name")
        extgstate = self.resolve_extgstate(name)
        if extgstate is None:
            raise PdfParseError("missing ExtGState resource")
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
                    self.blend_mode = self.named_value(blend_mode)
        except (TypeError, ValueError) as error:
            raise PdfParseError(str(error)) from error

    def flush_drawing(self, kind: str, fill_rule: str = "nonzero") -> None:
        self.sink.paint_path(self, self.current_path, kind, fill_rule)
        self.current_path = PdfPath()

    def internal_record_clip(self, fill_rule: str) -> None:
        self.sink.clip_path(self, self.current_path, fill_rule)

    def resolve_pattern_color(self, operands: tuple[Any, ...]) -> PatternPaint | None:
        if not operands:
            raise PdfParseError("invalid pattern resource or operands")
        pattern_name = self.document.resolver.resolve_name(operands[-1])
        if not pattern_name:
            raise PdfParseError("invalid pattern resource or operands")
        pattern = self.document.resolver.resolve(self.lookup_page_resource("Pattern", pattern_name))
        pattern_dict: PdfDict | None
        if isinstance(pattern, PdfStream):
            pattern_dict = cast(PdfDict, pattern.dictionary)
        else:
            pattern_dict = (
                self.document.resolver.resolve_dict(pattern) if pattern is not None else None
            )
        if not isinstance(pattern_dict, dict):
            raise PdfParseError("invalid pattern resource or operands")
        pattern_type = self.document.resolver.resolve_int(pattern_dict.get("PatternType"))
        if pattern_type == 2:
            shading: object = pattern_dict.get("Shading")
            shading = self.document.resolver.resolve(shading)
            shading_dict = (
                self.document.resolver.resolve_dict(shading) if shading is not None else None
            )
            if not isinstance(shading_dict, dict):
                raise PdfParseError("invalid pattern resource or operands")
            return ShadingPattern(dict(shading_dict))
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            raise PdfParseError("invalid pattern resource or operands")
        paint_type = self.document.resolver.resolve_int(pattern_dict.get("PaintType"))
        if paint_type not in {1, 2}:
            raise PdfParseError("invalid pattern resource or operands")
        base_color = None
        if paint_type == 2:
            base_color = self.normalize_color_operands(operands[:-1])
            if base_color is None:
                raise PdfParseError("invalid pattern resource or operands")
        bbox = self.document.resolver.resolve_box(pattern_dict.get("BBox"))
        if bbox is None:
            raise PdfParseError("invalid pattern resource or operands")
        x_step = self.document.resolver.resolve_float(pattern_dict.get("XStep"), default=None)
        y_step = self.document.resolver.resolve_float(pattern_dict.get("YStep"), default=None)
        if x_step is None or y_step is None or x_step == 0.0 or y_step == 0.0:
            raise PdfParseError("invalid pattern resource or operands")
        matrix = self.matrix_operand(
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


__all__ = ("TextDocument", "TextState", "NON_PAINTING_RENDER_MODES")
