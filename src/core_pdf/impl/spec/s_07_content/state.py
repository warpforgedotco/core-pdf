# SPDX-License-Identifier: AGPL-3.0-only
"""Content-stream interpreter state.

Holds the graphics and text state, the operator handlers, and glyph emission.
"""

from __future__ import annotations

import operator
import typing
from math import ceil, hypot
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.inline_images import InlineImage

from core_pdf.impl._impl.model.geometry import RectBox, intersect_bbox, transform_bbox
from core_pdf.impl._impl.model.glyphs import (
    GlyphCluster,
    GlyphObservation,
)
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.capture import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedLine,
    CapturedPath,
    PatternPaint,
    ShadingPattern,
    TilingPattern,
    marker_drawing,
    type3_font_matrix,
    type3_glyph_names,
)
from core_pdf.impl.spec.s_07_content.glyph_capture import (
    GlyphCapture,
    GlyphPaint,
    TextBasis,
    TextGeometry,
    capture_glyphs,
)
from core_pdf.impl.spec.s_07_content.image_capture import (
    image_source_from_stream,
    unit_square_placement,
)
from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperand,
    ContentOperands,
    OperationHandler,
)
from core_pdf.impl.spec.s_07_content.stream_execution import ContentStreamExecutor
from core_pdf.impl.spec.s_07_content.stream_state import (
    GRAPHICS_STATE_FIELDS,
    STREAM_STATE_MIRRORED,
    LayoutFormId,
    StreamState,
)
from core_pdf.impl.spec.s_07_content.text_runs import (
    RunAccumulator,
    is_garbage_text,
    normalize_extracted_text,
)
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import (
    normalize_pdf_name,
    parse_float_strict,
    parse_int,
    parse_int_strict,
)
from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf.impl.spec.s_08_graphics.color import color_operands_to_srgb
from core_pdf.impl.spec.s_08_graphics.color_spec import ImageColorSpec, color_spec_from_value
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.spec.s_09_fonts.decoder import DecodedGlyph, FontDecoder
from core_pdf.impl.spec.s_09_fonts.ligatures import detect_ligature_overrides
from core_pdf.impl.types import (
    PdfName,
    PdfReference,
    PdfString,
    Rectangle,
)

#: Reads every saved field off a ``TextState`` in one C-level call.
internal_capture_graphics_state = operator.attrgetter(*GRAPHICS_STATE_FIELDS)


MATRIX_TOLERANCE = 0.1


def detect_rotation_from_linear(
    A: float, B: float, C: float, D: float, tolerance: float = MATRIX_TOLERANCE
) -> int:
    scale_x = hypot(A, B)
    scale_y = hypot(C, D)
    if scale_x <= 0 or scale_y <= 0:
        return 0
    na, nb, nc, nd = A / scale_x, B / scale_x, C / scale_y, D / scale_y
    if (
        abs(na - 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd - 1.0) < tolerance
    ):
        return 0
    if (
        abs(na) < tolerance
        and abs(nb - 1.0) < tolerance
        and abs(nc + 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 90
    if (
        abs(na + 1.0) < tolerance
        and abs(nb) < tolerance
        and abs(nc) < tolerance
        and abs(nd + 1.0) < tolerance
    ):
        return 180
    if (
        abs(na) < tolerance
        and abs(nb + 1.0) < tolerance
        and abs(nc - 1.0) < tolerance
        and abs(nd) < tolerance
    ):
        return 270
    return 0


class TextDocument(typing.Protocol):
    @property
    def resolver(self) -> PdfValueResolver: ...

    raster_font_provider: Any

    # TextState never calls this itself, but `detect_ligature_overrides` takes a
    # FontResourceDocument, and a document only satisfies that protocol with it.
    def resolve(self, value: object, /) -> object: ...


# ISO 32000-1 Table 106: mode 3 is "neither fill nor stroke text (invisible)"
# and mode 7 is "add text to path for clipping" -- neither adds marks to the
# page. render/page.py already used this pair; extraction checked only mode 3.
internal_NON_PAINTING_RENDER_MODES = frozenset({3, 7})


class TextState:
    document: TextDocument
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_clusters: list[GlyphCluster]
    lines: list[CapturedLine]
    drawings: list[CapturedDrawing]
    current_path: CapturedPath
    current_point: tuple[float, float] | None
    subpath_start: tuple[float, float] | None
    stack: list[tuple[Any, ...]]
    clip_scope_stack: list[bool]
    fill_color: tuple[float, ...] | None
    fill_pattern: PatternPaint | None
    # Always a real number: __init__ seeds 1.0, `gs` clamps into [0, 1], and a
    # restore only ever copies a value that came from here.
    fill_opacity: float
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float
    blend_mode: str | None
    group_alpha: float | None
    flatness: int
    render_intent: str | None
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
    fill_color_space: str
    # The resolved space behind that name. The name alone cannot distinguish two
    # Separation resources with different tint transforms, and 8.6.6.3/8.6.6.4
    # need the palette or the tint transform to turn `sc`/`scn` operands into a
    # colour, so it travels with the name through q/Q.
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
    run_accumulator: RunAccumulator
    op_handlers: dict[str, OperationHandler]

    __slots__ = (
        "document",
        "runs",
        "glyphs",
        "glyph_clusters",
        "lines",
        "drawings",
        "current_path",
        "current_point",
        "subpath_start",
        "stack",
        "clip_scope_stack",
        "tm_a",
        "tm_b",
        "tm_c",
        "tm_d",
        "tm_e",
        "tm_f",
        "ca",
        "cb",
        "cc",
        "cd",
        "ce",
        "cf",
        "lm_a",
        "lm_b",
        "lm_c",
        "lm_d",
        "lm_e",
        "lm_f",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "stroke_opacity",
        "blend_mode",
        "group_alpha",
        "flatness",
        "render_intent",
        "clip_bbox",
        "layout_form_bbox",
        "layout_form_id",
        "page_clip",
        "fill_color_space",
        "fill_color_spec",
        "stroke_color_spec",
        "stroke_color_space",
        "compatibility_depth",
        "line_width",
        "line_cap",
        "line_join",
        "miter_limit",
        "dash_pattern",
        "font_size",
        "font_operand",
        "font_size_operand",
        "font_scale",
        "font_ascent",
        "font_descent",
        "font_space_width",
        "font_widths",
        "text_advance_scale",
        "char_space_scale",
        "word_space_scale",
        "horizontal_scale",
        "char_space",
        "word_space",
        "rise",
        "leading",
        "render_mode",
        "current_font",
        "current_decoder",
        "current_decoder_resources_id",
        "sequence",
        "text_object_id",
        "stream_order",
        "xobject_depth",
        "capture_source",
        "marked_content_stack",
        "type3_uncolored",
        "resources",
        "resources_id",
        "hidden_layers",
        "pending_line_break",
        "run_accumulator",
        "op_handlers",
        "combined_A",
        "combined_B",
        "combined_C",
        "combined_D",
        "inline_images",
        "stream_executor",
    )

    def __init__(
        self,
        document: TextDocument,
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
    ):
        self.document = document

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

        self.fill_color = (0.0, 0.0, 0.0)
        self.fill_pattern = None
        self.fill_opacity = 1.0
        self.stroke_color = (0.0, 0.0, 0.0)
        self.stroke_pattern = None
        self.stroke_opacity = 1.0
        self.blend_mode = None
        self.group_alpha = None
        self.flatness = 0
        self.render_intent = None
        self.clip_bbox = None
        self.layout_form_bbox = None
        self.layout_form_id = None
        # The page box bounds what can be displayed, but it is not a clip the
        # content stream established, so it is kept out of the graphics state:
        # recording it as one would give every unclipped mark the same clip
        # identity as a mark genuinely clipped to the full page, and the
        # provenance those identities feed is what layout groups runs by.
        self.page_clip = page_clip
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
        self.clip_scope_stack = []
        self.runs = []
        self.glyphs = []
        self.glyph_clusters = []
        self.lines = []
        self.drawings = []
        self.current_path = CapturedPath()
        self.current_point = None
        self.subpath_start = None
        self.font_size = 12.0
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
        self.sequence = 0
        self.text_object_id = 0
        self.stream_order = -1
        self.xobject_depth = 0
        self.capture_source = "native_text"
        self.compatibility_depth = 0
        self.marked_content_stack = []
        self.type3_uncolored = False
        self.resources = {}
        self.resources_id = 0
        self.hidden_layers = hidden_layers
        self.pending_line_break = False
        self.run_accumulator = RunAccumulator(self.runs)
        self.op_handlers = {
            name: getattr(self, handler) for name, handler in CONTENT_OPERATOR_HANDLERS.items()
        }

        self.combined_A = 1.0
        self.combined_B = 0.0
        self.combined_C = 0.0
        self.combined_D = 1.0
        self.inline_images: list[CapturedInlineImage] = []
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
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        segments: int | None = None,
    ) -> None:
        if self.current_point is None:
            self.current_point = (x3, y3)
            return
        x0, y0 = self.current_point
        if segments is None:
            x_scale = hypot(self.ca, self.cb)
            y_scale = hypot(self.cc, self.cd)
            scale = max(x_scale, y_scale, 1.0)
            control_len = (
                hypot(x1 - x0, y1 - y0) + hypot(x2 - x1, y2 - y1) + hypot(x3 - x2, y3 - y2)
            )
            flatness = max(0.1, float(self.flatness) if self.flatness else 0.25)
            segments = max(4, min(128, ceil(control_len * scale / (flatness * 8.0))))
        prev_x, prev_y = x0, y0
        path = self.current_path
        segment_step = 1.0 / segments
        for i in range(1, segments + 1):
            t = i * segment_step
            mt = 1.0 - t
            mt2 = mt * mt
            t2 = t * t
            b0 = mt2 * mt
            b1 = 3.0 * mt2 * t
            b2 = 3.0 * mt * t2
            b3 = t2 * t
            px = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
            py = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
            if not path.subpaths:
                path.move_to(prev_x, prev_y)
            path.line_to(px, py)
            prev_x, prev_y = px, py
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
        del self.stack[state.graphics_stack_len :]
        del self.clip_scope_stack[state.graphics_stack_len :]
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
        raw_category = self.resources.get(category)
        category_res = (
            self.document.resolver.resolve_dict(raw_category) if raw_category is not None else None
        )

        if isinstance(category_res, dict):
            res = category_res.get(name)
            if res is not None:
                return self.document.resolver.resolve(res)

        return None

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

        glyphs = decoder.decode_glyphs(data)
        if text is None:
            text = "".join([glyph.unicode for glyph in glyphs])
        return text, data, glyphs

    def is_text_visible(self, text: str) -> bool:
        if not text:
            return False
        first_code = ord(text[0])
        if (first_code < 32 or 0xE000 <= first_code <= 0xF8FF) and is_garbage_text(text):
            return False
        if (
            not self.marked_content_stack
            and self.render_mode not in internal_NON_PAINTING_RENDER_MODES
            and self.font_size >= 0.1
        ):
            return True

        # Render mode 3 and sub-0.1pt text paint nothing, so they are not visible
        # here. Whether such a layer is nonetheless the page's real text -- a scan
        # carrying an OCR layer -- is a property of the whole page, not of the runs
        # captured before this operator, so that call belongs to
        # `internal_hidden_text_is_trusted` once parsing has seen every run.
        if self.render_mode in internal_NON_PAINTING_RENDER_MODES or self.font_size < 0.1:
            return False

        return self.is_graphics_visible()

    def is_graphics_visible(self) -> bool:
        for entry in self.marked_content_stack:
            if entry.layer and entry.layer in self.hidden_layers:
                return False
        return True

    def get_decoder(self, *, update_metrics: bool = True) -> "FontDecoder":
        if self.current_decoder is not None:
            return self.current_decoder

        try:
            font_obj_ref = (
                self.lookup_page_resource("Font", self.current_font) if self.current_font else None
            )
        except PdfParseError:
            font_obj_ref = None
        if font_obj_ref is None:
            return FontDecoder({})

        try:
            font_obj = self.document.resolver.resolve(font_obj_ref)
        except PdfParseError:
            font_obj = None
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        if not isinstance(font_obj, dict):
            decoder = FontDecoder({})
            self.current_decoder = decoder
            self.current_decoder_resources_id = self.resources_id
            if update_metrics:
                self.update_font_metrics()
            return decoder

        font_dict = typing.cast(PdfDict, font_obj)
        resolved_font = self.document.resolver.resolve_font_dict(font_dict)
        decoder = FontDecoder(
            typing.cast(dict[str, object], resolved_font),
            ligature_overrides=detect_ligature_overrides(
                self.document, self.resources, resolved_font
            ),
            raster_font_provider=self.document.raster_font_provider,
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
        xobjects = self.resources.get("XObject")
        raw_xobj = xobjects.get(name) if isinstance(xobjects, dict) else None
        stream_key = (
            ("ref", raw_xobj.object_number, raw_xobj.generation_number)
            if isinstance(raw_xobj, PdfReference)
            else None
        )
        xobj = self.document.resolver.resolve(raw_xobj) if raw_xobj is not None else None
        if xobj is None:
            xobj = self.lookup_page_resource("XObject", name)
        if not isinstance(xobj, PdfStream):
            return
        xobj_dict = xobj.dictionary
        subtype = self.document.resolver.resolve_name(xobj_dict.get("Subtype"))
        if self.document.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return
        if subtype == "Image":
            if self.is_graphics_visible():
                width = self.document.resolver.resolve_int(xobj_dict.get("Width")) or 0
                height = self.document.resolver.resolve_int(xobj_dict.get("Height")) or 0
                bbox = None
                quad = None
                if width > 0 and height > 0:
                    bounds, quad = unit_square_placement(self.ctm)
                    bbox = RectBox(*bounds)
                source, smask_alpha = image_source_from_stream(xobj, self.document.resolver)
                # A stencil mask carries no colour samples: PDF 8.9.6.2 paints its
                # set bits in the current fill colour. Every other image ignores
                # the fill, so recording it is only meaningful for the mask case,
                # but it costs nothing to carry and the renderer decides.
                image_is_stencil = xobj_dict.get("ImageMask") is True
                self.drawings.append(
                    CapturedDrawing(
                        seqno=self.sequence,
                        fill=self.fill_color if image_is_stencil else None,
                        fill_opacity=self.fill_opacity if image_is_stencil else None,
                        blend_mode=self.blend_mode,
                        dash_pattern=self.transformed_dash_pattern(),
                        soft_mask_alpha=smask_alpha,
                        kind="image",
                        image_source=source,
                        raw_data=xobj.raw_data,
                        dictionary=dict(xobj_dict),
                        image_clip=self.clip_bbox,
                        items=[("quad", quad)] if quad is not None else [],
                        bbox=bbox,
                        stream_order=self.stream_order,
                        xobject_depth=self.xobject_depth,
                    )
                )
                self.sequence += 1
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
                # Only isolate the group when compositing would actually differ;
                # at ca == 1 in Normal mode a group buffer is a no-op, and
                # painting directly stays the cheaper path.
                blend = self.blend_mode
                if self.fill_opacity < 1.0 or (blend is not None and blend != "Normal"):
                    group_alpha = max(0.0, min(1.0, self.fill_opacity))
        raw_resources = xobj_dict.get("Resources")
        resources = cast(
            PdfDict,
            (
                raw_resources
                if isinstance(raw_resources, dict)
                else self.document.resolver.resolve_dict(raw_resources)
            )
            or self.resources,
        )
        xobj_matrix = xobj_dict.get("Matrix")
        if isinstance(xobj_matrix, (list, tuple)) and len(xobj_matrix) > 6:
            xobj_matrix = xobj_matrix[:6]
        nested_ctm = (
            Matrix.from_operand(xobj_matrix) if xobj_matrix is not None else IDENTITY_MATRIX
        ).multiply(self.ctm)
        raw_form_bbox = xobj_dict.get("BBox")
        form_bbox = self.document.resolver.resolve_box(raw_form_bbox)
        transformed_form_bbox = (
            transform_bbox(form_bbox, nested_ctm) if form_bbox is not None else None
        )
        layout_form_bbox = None
        if isinstance(raw_form_bbox, (list, tuple)) and len(raw_form_bbox) >= 4:
            raw_values = tuple(
                self.document.resolver.resolve_float(value, default=None)
                for value in raw_form_bbox[:4]
            )
            if all(value is not None for value in raw_values):
                raw_x, raw_y, raw_width, raw_height = typing.cast(Rectangle, raw_values)
                # PDFMiner's LTFigure constructor historically interprets the
                # four /BBox values as x, y, width, height. Preserve that raw
                # layout geometry separately from the spec-correct clipping
                # rectangle so compatibility projections can reproduce it.
                layout_form_bbox = transform_bbox(
                    (
                        raw_x,
                        raw_y,
                        raw_x + raw_width,
                        raw_y + raw_height,
                    ),
                    nested_ctm,
                )
        self.stream_executor.queue(
            xobj,
            resources,
            nested_ctm,
            depth + 1,
            clip_bbox=transformed_form_bbox,
            layout_form_bbox=layout_form_bbox,
            # Retain the invocation tree, not merely the outermost form. PDFMiner
            # emits one nested LTFigure per Form XObject even when a child's
            # historical figure bbox extends beyond its parent.
            layout_form_id=(
                *(self.layout_form_id if isinstance(self.layout_form_id, tuple) else ()),
                (stream_key, layout_form_bbox),
            ),
            group_alpha=group_alpha,
            stream_key=stream_key,
            swallow_parse_errors=True,
        )

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.ca + y * self.cc + self.ce,
            x * self.cb + y * self.cd + self.cf,
        )

    def graphics_scale(self) -> float:
        x_scale = hypot(self.ca, self.cb)
        y_scale = hypot(self.cc, self.cd)
        if x_scale == 0 and y_scale == 0:
            return 1.0
        if x_scale == 0:
            return y_scale
        if y_scale == 0:
            return x_scale
        return (x_scale + y_scale) * 0.5

    def transformed_line_width(self) -> float:
        line_width = max(0.0, self.line_width)
        if line_width == 0:
            return 0.0
        return line_width * self.graphics_scale()

    def transformed_dash_pattern(self) -> tuple[list[float], float] | None:
        dash_pattern = self.dash_pattern
        if not dash_pattern:
            return None
        dash_array, phase = dash_pattern
        scale = self.graphics_scale()
        return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale

    def flush_drawing(self, kind: str, fill_rule: str = "nonzero") -> None:
        if not self.is_graphics_visible():
            self.current_path.clear()
            return

        if (
            self.ca == 1.0
            and self.cb == 0.0
            and self.cc == 0.0
            and self.cd == 1.0
            and self.ce == 0.0
            and self.cf == 0.0
        ):
            path = self.current_path
            self.current_path = CapturedPath()
        else:
            path = self.current_path.transformed(
                Matrix(self.ca, self.cb, self.cc, self.cd, self.ce, self.cf)
            )
            self.current_path.clear()
        if path.has_segments():
            line_width = self.transformed_line_width()
            if len(path.subpaths) == 1 and len(path.subpaths[0].points) == 2:
                (x0, y0), (x1, y1) = path.subpaths[0].points
                if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01:
                    self.lines.append(CapturedLine(x0, y0, x1, y1, line_width))
            else:
                self.lines.extend(path.derived_lines(line_width))
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=self.fill_color,
                    fill_pattern=self.fill_pattern,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_pattern=self.stroke_pattern,
                    stroke_opacity=self.stroke_opacity,
                    line_width=line_width,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    kind=kind,
                    path=path,
                    stream_order=self.stream_order,
                    xobject_depth=self.xobject_depth,
                )
            )
            # A painted path must consume a sequence number like text does.
            # Sharing one with the text that follows lets a seqno-ordered
            # replay paint a cell background over the run's first glyphs.
            self.sequence += 1

    def internal_is_clipped_away(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Report whether a box falls entirely outside the active clip.

        Text only survives if it overlaps the clip, so a form XObject's /BBox,
        a `W n` clip path and the page box all suppress the marks they exclude.
        Partially clipped text is kept whole: the glyph is on the page, and
        reporting half of one would be worse than reporting it.
        """
        for clip in (self.clip_bbox, self.page_clip):
            if clip is None:
                continue
            if x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]:
                return True
        return False

    def update_pending_run(self, new_run: TextRun) -> None:
        if not self.internal_is_clipped_away(new_run.x0, new_run.y0, new_run.x1, new_run.y1):
            self.run_accumulator.append(new_run)

    def record_glyph_observations(
        self,
        text: str,
        decoder: FontDecoder,
        rotation_angle: int,
        visible: bool,
        *,
        glyphs: tuple[DecodedGlyph, ...],
        text_basis: TextBasis,
        effective_font_size: float,
        effective_font_height: float,
    ) -> GlyphCapture:
        """Snapshot this text show's inputs for the independent glyph recorder."""
        geometry = TextGeometry(
            basis=text_basis,
            font_size=self.font_size,
            font_scale=self.font_scale,
            font_ascent=self.font_ascent,
            font_descent=self.font_descent,
            advance_scale=self.text_advance_scale,
            char_space_scale=self.char_space_scale,
            word_space_scale=self.word_space_scale,
            char_space=self.char_space,
            word_space=self.word_space,
            horizontal_scale=self.horizontal_scale,
            rise=self.rise,
            rotation_angle=rotation_angle,
            effective_font_size=effective_font_size,
            effective_font_height=effective_font_height,
        )
        paint = GlyphPaint(
            visible=visible,
            clip_bbox=self.clip_bbox,
            page_clip=self.page_clip,
            fill=self.fill_color,
            render_mode=self.render_mode,
            fill_opacity=self.fill_opacity,
            stroke_color=self.stroke_color,
            stroke_opacity=self.stroke_opacity,
            line_width=self.transformed_line_width(),
            line_cap=self.line_cap,
            line_join=self.line_join,
            dash_pattern=self.transformed_dash_pattern(),
            blend_mode=self.blend_mode,
            group_alpha=self.group_alpha,
        )
        provenance = (
            ("source", self.capture_source),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            ("text_matrix", text_basis[2:]),
            ("line_matrix_origin", (self.lm_e, self.lm_f)),
            ("horizontal_scale", self.horizontal_scale),
            ("char_space", self.char_space),
            ("text_rise", self.rise),
        )
        return capture_glyphs(
            text,
            glyphs,
            decoder,
            geometry=geometry,
            paint=paint,
            font_name=self.current_font,
            provenance=provenance,
            seqno=self.sequence,
            text_object_id=self.text_object_id,
            cluster_start=len(self.glyph_clusters),
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
            glyphs = decoder.decode_glyphs(data)
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
                self.pending_line_break = False
            return

        visible = self.is_text_visible(text)

        fs = self.font_size
        rise = self.rise

        adv_x, adv_y = decoder.text_advance_vector(
            data,
            font_size=fs,
            char_space=self.char_space,
            word_space=self.word_space,
            horizontal_scale=self.horizontal_scale,
            glyphs=glyphs,
        )

        ascent = self.font_ascent
        descent = self.font_descent

        A = self.combined_A
        B = self.combined_B
        C = self.combined_C
        D = self.combined_D

        ca = self.ca
        cb = self.cb
        cc = self.cc
        cd = self.cd
        ce = self.ce
        cf = self.cf
        te, tf = self.tm_e, self.tm_f
        E = te * ca + tf * cc + ce
        F = te * cb + tf * cd + cf

        if decoder.is_vertical:
            c0_x = descent * A + rise * C + E
            c0_y = descent * B + rise * D + F
            c1_x = ascent * A + rise * C + E
            c1_y = ascent * B + rise * D + F
            adv_C = adv_y * C
            adv_D = adv_y * D
            c2_x = adv_C + c0_x
            c2_y = adv_D + c0_y
            c3_x = adv_C + c1_x
            c3_y = adv_D + c1_y
        else:
            ar = ascent + rise
            dr = descent + rise
            c0_x = dr * C + E
            c0_y = dr * D + F
            c1_x = ar * C + E
            c1_y = ar * D + F
            adv_A = adv_x * A
            adv_B = adv_x * B
            c2_x = adv_A + c0_x
            c2_y = adv_B + c0_y
            c3_x = adv_A + c1_x
            c3_y = adv_B + c1_y

        x0 = min(c0_x, c1_x, c2_x, c3_x)
        y0 = min(c0_y, c1_y, c2_y, c3_y)
        x1 = max(c0_x, c1_x, c2_x, c3_x)
        y1 = max(c0_y, c1_y, c2_y, c3_y)

        rot = detect_rotation_from_linear(A, B, C, D)
        seqno = self.sequence
        ta, tb, tc, td = self.tm_a, self.tm_b, self.tm_c, self.tm_d
        scale_factor = hypot(C, D) if decoder.is_vertical else hypot(A, B)
        effective_font_size = fs * scale_factor
        effective_font_height = fs * (hypot(A, B) if decoder.is_vertical else hypot(C, D))
        effective_space_width = self.font_space_width * scale_factor
        baseline = (
            E,
            F,
            E + adv_x * A + adv_y * C,
            F + adv_x * B + adv_y * D,
        )
        provenance = (
            ("source", self.capture_source),
            ("seqno", seqno),
            ("font_name", self.current_font),
            ("stream_order", self.stream_order),
            ("xobject_depth", self.xobject_depth),
            ("text_render_mode", self.render_mode),
            ("font_size", fs),
            ("clip_bbox", self.clip_bbox),
            ("layout_form_bbox", self.layout_form_bbox),
            ("layout_form_id", self.layout_form_id),
            *(
                (("mcid", mcid),)
                if (mcid := self.current_marked_content_mcid()) is not None
                else ()
            ),
        )
        advance_bbox = (x0, y0, x1, y1)

        new_run = TextRun(
            text=normalize_extracted_text(text),
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            tx=te,
            ty=tf,
            font_size=effective_font_size,
            font_name=self.current_font,
            space_width=effective_space_width,
            order=seqno,
            stream_order=self.stream_order,
            xobject_depth=self.xobject_depth,
            is_vertical=decoder.is_vertical,
            rotation_angle=rot,
            visible=visible,
            line_break_before=self.pending_line_break,
            seqno=seqno,
            fill_color=self.fill_color,
            advance_bbox=advance_bbox,
            ink_bbox=advance_bbox,
            baseline=baseline,
            provenance=provenance,
            confidence=None,
        )
        actual_text_span = self.current_actual_text_span()
        if actual_text_span is not None:
            new_run.confidence = 1.0
            actual_text_span.add_run(
                new_run,
                font_decoder=decoder,
                effective_font_height=effective_font_height,
            )
        else:
            captured = self.record_glyph_observations(
                text,
                decoder,
                rot,
                visible,
                glyphs=glyphs,
                text_basis=(E, F, A, B, C, D),
                effective_font_size=effective_font_size,
                effective_font_height=effective_font_height,
            )
            self.glyphs.extend(captured.glyphs)
            self.glyph_clusters.extend(captured.clusters)
            new_run.glyph_clusters = tuple(captured.clusters)
            geometry = captured.geometry
            if geometry.started:
                new_run.advance_bbox = geometry.advance
                new_run.ink_bbox = geometry.ink
                new_run.confidence = geometry.confidence
            self.update_pending_run(new_run)

        self.sequence = seqno + 1

        self.tm_e = te + adv_x * ta + adv_y * tc
        self.tm_f = tf + adv_x * tb + adv_y * td
        self.pending_line_break = False

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
        glyph_names = decoder.type3_glyph_names
        if glyph_names is None:
            glyph_names = type3_glyph_names(decoder)
            decoder.type3_glyph_names = glyph_names

        resources = font.get("Resources")
        if not isinstance(resources, dict):
            resources = self.resources
        font_matrix = type3_font_matrix(font)
        widths = self.font_widths or decoder.fast_widths
        cs = self.char_space_scale
        ws = self.word_space_scale
        scale = self.text_advance_scale

        for code in data:
            glyph_name = glyph_names.get(code)
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
        zero_copy_flush = (
            not decoder.is_cid_font and decoder.to_unicode is None and decoder.cmap is None
        )

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
                    if zero_copy_flush:
                        self.append_text(data=memoryview(pending_bytes), decoder=decoder)
                    else:
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
            if zero_copy_flush:
                self.append_text(data=memoryview(pending_bytes), decoder=decoder)
            else:
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

    def emit_actual_text_span(self, entry: MarkedContentEntry) -> None:
        actual_text = entry.actual_text
        captured = entry.run
        if actual_text is None or captured is None:
            return
        self.update_pending_run(
            captured.replace(
                text=normalize_extracted_text(actual_text),
                ink_bbox=captured.advance_bbox,
                provenance=(*captured.provenance, ("unicode_source", "actual_text")),
            )
        )
        self.glyphs.append(
            GlyphObservation(
                text=actual_text,
                ink_bbox=captured.advance_bbox,
                advance_bbox=captured.advance_bbox,
                seqno=captured.seqno,
                font_name=captured.font_name,
                font_size=captured.font_size,
                baseline=captured.baseline,
                rotation_angle=captured.rotation_angle,
                fill=captured.fill_color,
                visible=captured.visible,
                confidence=captured.confidence,
                unicode_source="actual_text",
                font_decoder=entry.font_decoder,
                effective_font_size=captured.font_size,
                effective_font_height=entry.effective_font_height,
                provenance=captured.provenance,
            )
        )

    def internal_begin_text(self) -> None:
        self.run_accumulator.flush()
        self.tm_a = self.lm_a = 1.0
        self.tm_b = self.lm_b = 0.0
        self.tm_c = self.lm_c = 0.0
        self.tm_d = self.lm_d = 1.0
        self.tm_e = self.lm_e = 0.0
        self.tm_f = self.lm_f = 0.0
        self.update_combined()

    def op_ET(self, operands: ContentOperands, depth: int) -> None:
        self.run_accumulator.flush()

    def internal_move_text(self, tx: float, ty: float) -> None:
        self.run_accumulator.flush()
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
        self.text_object_id += 1
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
        self.run_accumulator.flush()
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
                except (TypeError, ValueError):
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
        except (TypeError, ValueError):
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
        self.pending_line_break = True
        self.internal_show_text(operands[0])

    def op_double_quote(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) < 3 or (values := self.as_floats(operands, 2)) is None:
            return
        self.word_space, self.char_space = values
        self.update_text_scales()
        self.internal_move_text(0.0, -self.leading)
        self.pending_line_break = True
        self.internal_show_text(operands[2])

    def op_BI(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        operand = operands[0]
        # Duck-typed on purpose: the inline-image parser yields an InlineImage, but any
        # operand exposing ``dictionary`` is accepted here.
        if not hasattr(operand, "dictionary"):
            return
        image = cast("InlineImage", operand)
        if self.is_graphics_visible():
            dictionary = dict(image.dictionary)
            data = getattr(image, "data", b"")
            color_name = normalize_pdf_name(dictionary.get("ColorSpace"))
            if color_name is not None:
                color_resource = self.lookup_page_resource("ColorSpace", color_name)
                if color_resource is not None:
                    dictionary[PdfName.of("ColorSpace")] = cast(PdfObject, color_resource)
            source, _ = image_source_from_stream(
                PdfStream(raw_data=data, dictionary=dictionary), self.document.resolver
            )
            self.inline_images.append(
                CapturedInlineImage(
                    seqno=self.sequence,
                    dictionary=dictionary,
                    data=data,
                    image_source=source,
                    image_clip=self.clip_bbox,
                    ctm=self.ctm,
                    xobject_depth=self.xobject_depth,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    stream_order=self.stream_order,
                    fill=self.fill_color if dictionary.get("ImageMask") is True else None,
                    fill_opacity=self.fill_opacity if dictionary.get("ImageMask") is True else None,
                )
            )
            self.sequence += 1

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
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
        self.marked_content_stack.append(MarkedContentEntry())

    def op_EMC(self, operands: ContentOperands, depth: int) -> None:
        if self.marked_content_stack:
            self.emit_actual_text_span(self.marked_content_stack.pop())

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
        except (TypeError, ValueError):
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
        if (
            self.is_graphics_visible()
            and self.current_point is not None
            and self.subpath_start is not None
        ):
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

    def internal_emit_clip_scope_push(self) -> None:
        if not self.clip_scope_stack or self.clip_scope_stack[-1]:
            return
        self.clip_scope_stack[-1] = True
        self.drawings.append(marker_drawing("state-push", self.sequence))
        self.sequence += 1

    def op_W(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("nonzero")

    def internal_record_clip(self, fill_rule: str) -> None:
        path = self.current_path.transformed(self.ctm)
        if not path.has_segments():
            return
        clip_bbox = path.bbox()
        if clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, clip_bbox)
        if self.is_graphics_visible():
            self.internal_emit_clip_scope_push()
            self.drawings.append(
                CapturedDrawing(
                    seqno=self.sequence,
                    fill=None,
                    fill_opacity=None,
                    blend_mode=self.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    line_width=0.0,
                    line_cap=self.line_cap,
                    line_join=self.line_join,
                    dash_pattern=self.transformed_dash_pattern(),
                    fill_rule=fill_rule,
                    kind="clip",
                    path=path,
                )
            )
            self.sequence += 1

    def op_W_star(self, operands: ContentOperands, depth: int) -> None:
        self.internal_record_clip("evenodd")

    def normalize_colors(self, *components: Any) -> tuple[float, ...] | None:
        values: list[float] = []
        for component in components:
            try:
                values.append(max(0.0, min(1.0, self.as_float(component))))
            except ValueError:
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
            self.stroke_pattern = None
        else:
            self.fill_color_space = color_space
            self.fill_color_spec = None
            self.fill_color = normalized
            self.fill_pattern = None

    # `o` is a tuple or list of raw PDF operands, so `Any` is the honest annotation.
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
        resolve = self.document.resolver.resolve
        value = self.lookup_page_resource("ColorSpace", name)
        if value is None:
            # An inline device space (`/DeviceRGB cs`) names no resource.
            value = name
        if isinstance(value, (list, tuple)):
            # Tint functions and Indexed palettes may be indirect references.
            value = [resolve(entry) for entry in value]
        base = value[0] if isinstance(value, (list, tuple)) and value else value
        color_space = normalize_pdf_name(base) or name
        try:
            spec = color_spec_from_value(value)
        except (ValueError, TypeError):
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
                converted = color_operands_to_srgb(spec, values)
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

    def resolve_pattern_color(self, operands: tuple[Any, ...]) -> PatternPaint | None:
        if not operands:
            return None
        pattern_name = self.document.resolver.resolve_name(operands[-1])
        if not pattern_name:
            return None
        pattern = self.lookup_page_resource("Pattern", pattern_name)
        if isinstance(pattern, PdfStream):
            pattern_dict = self.document.resolver.resolve_dict(pattern.dictionary)
        else:
            pattern_dict = (
                self.document.resolver.resolve_dict(pattern) if pattern is not None else None
            )
        if not isinstance(pattern_dict, dict):
            return None
        pattern_type = parse_int(pattern_dict.get("PatternType"), None)
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
        paint_type = parse_int(pattern_dict.get("PaintType"), 1)
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
        try:
            matrix = Matrix.from_operand(pattern_dict.get("Matrix"))
        except ValueError:
            matrix = IDENTITY_MATRIX
        resources = self.document.resolver.resolve_dict(pattern_dict.get("Resources")) or {}
        nested_state = type(self)(self.document, hidden_layers=self.hidden_layers)
        try:
            nested_state.consume_stream(pattern, resources, matrix, 0)
        except Exception:
            return None
        # The cell's drawings are owned by this pattern and painted nowhere
        # else, so an uncoloured (PaintType 2) pattern recolours them in place
        # rather than copying every field into a parallel record.
        if paint_type == 2:
            for drawing in nested_state.drawings:
                if drawing.kind in {"fill", "fillstroke"}:
                    drawing.fill = base_color
                if drawing.kind in {"stroke", "fillstroke"}:
                    drawing.stroke_color = base_color
            for glyph in nested_state.glyphs:
                glyph.fill = base_color
                glyph.stroke_color = base_color
        return TilingPattern(
            bbox=bbox,
            x_step=float(x_step),
            y_step=float(y_step),
            drawings=nested_state.drawings,
            glyphs=[glyph for glyph in nested_state.glyphs if glyph.has_paint],
            inline_images=nested_state.inline_images,
        )

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
        value = self.document.resolver.resolve_name_like_value(operands[0])
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
        props = self.lookup_page_resource("Properties", name)
        return cast("dict[str, Any]", props) if isinstance(props, dict) else None

    def resolve_marked_content_layer(self, value: Any) -> str | None:
        if value is None:
            return None

        resolved = self.document.resolver.resolve(value)
        if isinstance(resolved, dict):
            oc = resolved.get("OC")
            if oc is not None:
                return self.document.resolver.resolve_name_or_text(oc)

        return self.document.resolver.resolve_name_or_text(value)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = False

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if not operands or not self.is_graphics_visible():
            return
        shading_ref = self.document.resolver.resolve_name(operands[0])
        if not shading_ref:
            return
        shading = self.lookup_page_resource("Shading", shading_ref)
        if not isinstance(shading, dict):
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.fill_color,
                fill_opacity=self.fill_opacity,
                stroke_color=self.stroke_color,
                stroke_opacity=self.stroke_opacity,
                line_width=self.line_width,
                line_cap=self.line_cap,
                line_join=self.line_join,
                dash_pattern=self.transformed_dash_pattern(),
                blend_mode=self.blend_mode,
                soft_mask_alpha=self.group_alpha,
                kind="shading",
                items=[],
                dictionary=dict(shading),
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )
        self.sequence += 1

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
        return parse_float_strict(value, "invalid numeric operand")

    @classmethod
    def as_floats(cls, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        """The leading `count` operands as floats.

        Returns None when the operator is short of operands or any of them is
        not numeric -- damaged streams produce both, and every numeric operator
        answers them the same way, by ignoring the operation.
        """
        if len(operands) < count:
            return None
        try:
            return tuple([cls.as_float(operands[i]) for i in range(count)])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        return parse_int_strict(value, "invalid numeric operand")

    @classmethod
    def as_int_operand(cls, operands: ContentOperands) -> int | None:
        """The first operand as an int, or None when absent or not numeric."""
        if not operands:
            return None
        try:
            return cls.as_int(operands[0])
        except (TypeError, ValueError):
            return None

    def resolve_extgstate(self, name: str) -> dict[str, Any] | None:
        resolved = self.lookup_page_resource("ExtGState", name)
        if not isinstance(resolved, dict):
            return None
        return cast("dict[str, Any]", resolved)

    def op_q(self, operands: ContentOperands, depth: int) -> None:
        self.clip_scope_stack.append(False)
        self.stack.append(internal_capture_graphics_state(self))

    def op_Q(self, operands: ContentOperands, depth: int) -> None:
        clip_scope_emitted = self.clip_scope_stack.pop() if self.clip_scope_stack else False
        if clip_scope_emitted:
            self.drawings.append(marker_drawing("state-pop", self.sequence))
            self.sequence += 1
        if not self.stack:
            return
        self.restore_graphics_state(self.stack.pop())

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
            return
        name = self.document.resolver.resolve_name(operands[0])
        if not name:
            return
        extgstate = self.resolve_extgstate(name)
        if not extgstate:
            return
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
                    self.blend_mode = self.document.resolver.resolve_name_like_value(blend_mode)
        except (TypeError, ValueError):
            return
