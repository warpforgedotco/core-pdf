# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.8: interpreting a content stream's graphics, text and XObject operators."""

from __future__ import annotations

import typing
from collections.abc import Callable
from copy import copy
from math import isfinite
from typing import Any, ClassVar

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.model import (
    NON_PAINTING_RENDER_MODES,
    ContentSink,
    Expansion,
    GraphicsState,
    MarkedContentEntry,
    PatternPaint,
    PdfPath,
    ShadingPattern,
    TilingPattern,
)
from core_pdf.impl.spec.s_07_content.operations import (
    ContentOperands,
    OperationHandler,
    validate_content_operands,
)
from core_pdf.impl.spec.s_07_content.streams import (
    ContentStreamExecutor,
    ContentStreamFrame,
    StreamKey,
    StreamState,
)
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.resources import resolve_resource_dict
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfValueResolver
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float
from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf.impl.spec.s_08_graphics.color import (
    initial_color_components,
    normalize_color_components,
)
from core_pdf.impl.spec.s_08_graphics.color_rendering import (
    parse_black_point_compensation,
    parse_rendering_intent,
)
from core_pdf.impl.spec.s_08_graphics.color_spec import (
    DEVICE_CMYK,
    DEVICE_GRAY,
    DEVICE_RGB,
    ColorSpace,
    parse_color_space,
)
from core_pdf.impl.spec.s_08_graphics.default_spaces import (
    DefaultSpaces,
    default_space_values,
    remap_color_space,
    valid_default_space,
)
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.spec.s_08_graphics.shading import SHADING_STREAM_KEY
from core_pdf.impl.spec.s_09_fonts.service import DecodedFontGlyph, FontProvider, FontService
from core_pdf.impl.spec.standards import SemanticContext
from core_pdf.impl.spec.types import PdfName, PdfReference, PdfString
lazy from core_pdf.impl.spec.s_08_graphics.geometry import transform_bbox
lazy from core_pdf.impl.spec.s_11_transparency.blend import first_recognized_blend_mode
lazy from core_pdf.impl.spec.s_11_transparency.blending_space import blending_space_of
lazy from core_pdf.impl.spec.s_11_transparency.soft_masks import SoftMask, parse_soft_mask
lazy from core_pdf.impl.spec.s_14_structure.expansions import expansion_text
lazy from core_pdf_cythonized._text_array import show_decoded, show_string, show_text_array

NUMERIC_TYPES = frozenset((int, float))


def moved_to(matrix: Matrix, e: float, f: float) -> Matrix:
    return tuple.__new__(Matrix, (matrix[0], matrix[1], matrix[2], matrix[3], e, f))


# What core-pdf-cythonized's text showing needs from spec: the string type, and moved_to.
TEXT_SPEC_CONTEXT = (PdfString, moved_to)


def parse_error_from(error: Exception) -> PdfParseError:
    failure = PdfParseError(str(error))
    failure.__cause__ = error
    return failure


class ContentInterpreter:
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
        # The CTM the running content stream started with: its default coordinate space,
        # which a pattern's matrix maps pattern space to (8.7.3.1).
        self.stream_ctm = IDENTITY_MATRIX
        self.stack: list[GraphicsState] = []
        self.graphics_stack_floor = 0
        self.current_path = PdfPath()
        self.current_point: tuple[float, float] | None = None
        self.subpath_start: tuple[float, float] | None = None
        self.pending_clip_rule_value: str | None = None
        self.xobject_depth = 0
        self.compatibility_depth = 0
        self.marked_content_stack: list[MarkedContentEntry] = []
        # Whether colour is set elsewhere, so the colour operators are ignored (8.6.8): in a
        # Type 3 glyph that begins with d1, in the cell of an uncoloured tiling pattern, and
        # in every stream either invokes. inherited_uncolored is the value a glyph
        # description starts with, which d0 leaves in place.
        self.type3_uncolored = False
        self.inherited_uncolored = False
        self.resources: PdfDict = {}
        self.color_space_cache: dict[int, tuple[object, ColorSpace]] = {}
        self.remapped_space_cache: dict[int, tuple[object, DefaultSpaces, ColorSpace]] = {}
        self.default_spaces_cache: dict[int, tuple[PdfDict, DefaultSpaces | None]] = {}
        self.default_spaces_memo: tuple[object, DefaultSpaces | None] = (None, None)
        self.shading_streams: dict[int, tuple[PdfStream, PdfDict]] = {}
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
        raise error

    def append_cubic_curve(
        self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> None:
        if self.current_point is None:
            self.reject(PdfParseError("curve has no current point"), "path", None)
            self.current_point = (x3, y3)
            return
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
            stream_ctm=self.stream_ctm,
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
        self.stream_ctm = state.stream_ctm
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
        if override is not None and name in {"BX", "EX"}:
            self.default_handlers[name](operands, depth)
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
        try:
            return resolve_resource_dict(value, self.resolver)
        except PdfParseError as error:
            return self.reject(error, "resources", None)

    def matrix_operand(self, value: object, context: str) -> Matrix:
        value = self.resolver.deep_resolve(value)
        if value is None:
            return IDENTITY_MATRIX
        try:
            return Matrix.from_operand(value)
        except ValueError as error:
            fallback = self.matrix_fallback(value, context)
            if fallback is None:
                raise
            return self.reject(error, context, fallback)

    def matrix_fallback(self, value: object, context: str) -> Matrix | None:
        return None

    def get_decoder(self) -> FontService:
        if self.graphics.current_decoder is not None:
            return self.graphics.current_decoder
        font_name = self.graphics.current_font
        if not font_name:
            self.reject(PdfParseError("text operation has no selected font"), "font", None)
            return self.font_provider({}, self.resources)
        try:
            font_reference = self.lookup_page_resource("Font", font_name)
        except PdfParseError as error:
            self.reject(error, "font-resource", None)
            return self.font_provider({}, self.resources)
        if font_reference is None:
            self.reject(PdfParseError("font resource must be a dictionary"), "font-resource", None)
            return self.font_provider({}, self.resources)
        try:
            font = self.resolver.resolve(font_reference)
        except PdfParseError as error:
            font = self.reject(error, "font-resolution", None)
        else:
            if isinstance(font, PdfStream) or not isinstance(font, dict):
                fallback = font.dictionary if isinstance(font, PdfStream) else font
                font = self.reject(
                    PdfParseError("font resource must be a dictionary"), "font-resource", fallback
                )
        resources = self.resources
        decoder = self.decoder_for(font_reference, font, resources)
        self.graphics.current_decoder = decoder
        self.graphics.decoder_resources = resources
        return decoder

    def decoder_for(self, font_reference: object, font: object, resources: PdfDict) -> FontService:
        resolved_font = self.resolver.resolve_font_dict(font) if isinstance(font, dict) else {}
        return self.font_provider(resolved_font, resources)

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
        if self.resolver.resolve_name(xobj_dict.get("Type")) == "ObjStm":
            return self.reject(
                PdfParseError("XObject resource is an object stream"), "xobject", None
            )
        if subtype == "Image":
            if self.type3_uncolored and not self.paints_image_mask(xobj_dict):
                return None
            self.sink.paint_image(self, xobj)
            return None
        if subtype != "Form":
            return self.reject(PdfParseError("unsupported XObject subtype"), "xobject", None)
        return self.append_form_xobject(xobj, depth, stream_key=stream_key)

    @cites(
        "ISO 32000-2",
        "8.6.8",
        introduced="1.0",
        rule="where colour is set elsewhere, only image masks of the images are painted",
    )
    def paints_image_mask(self, dictionary: object) -> bool:
        return (
            isinstance(dictionary, dict)
            and self.resolver.resolve(dictionary.get("ImageMask")) is True
        )

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
        group_color_space = None
        group_cs: object = None
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
                group_cs = group_dict.get("CS") if group_isolated else None
        resources = self.resolve_form_resources(xobj_dict.get("Resources"))
        if group_cs is not None:
            group_color_space = self.group_color_space(group_cs, self.default_spaces_of(resources))
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
                frame.group_color_space = group_color_space
        return frame

    @cites(
        "ISO 32000-2",
        "11.6.6",
        introduced="1.4",
        rule="a group's CS naming a device space is remapped through the default spaces",
    )
    def group_color_space(
        self, value: object, defaults: DefaultSpaces | None = None
    ) -> ColorSpace | None:
        """The colour space a group attributes dictionary's CS entry names, if it can be a
        blending space; None leaves the group in its parent's (11.6.6). A device space is
        remapped through defaults, those of the group XObject's resources (8.6.5.6)."""
        if value is None:
            return None
        try:
            resolved = self.resolver.deep_resolve(value)
            remapped = (
                defaults.spaces.get(self.resolver.resolve_name(resolved) or "")
                if defaults is not None
                else None
            )
            space = remapped if remapped is not None else self.parse_color_space_value(resolved)
        except ValueError, TypeError:
            return None
        return space if blending_space_of(space) is not None else None

    def current_default_spaces(self) -> DefaultSpaces | None:
        """The default colour spaces of the current resource dictionary (8.6.5.6)."""
        resources = self.resources
        memo = self.default_spaces_memo
        if memo[0] is resources:
            return memo[1]
        found = self.default_spaces_of(resources)
        self.default_spaces_memo = (resources, found)
        return found

    @cites(
        "ISO 32000-2",
        "8.6.5.6",
        introduced="1.1",
        rule="the default spaces are those of the current resource dictionary",
    )
    def default_spaces_of(self, resources: PdfDict) -> DefaultSpaces | None:
        """The valid default colour spaces a resource dictionary's ColorSpace subdictionary
        names, or None."""
        raw = resources.get("ColorSpace")
        if raw is None:
            return None
        cached = self.default_spaces_cache.get(id(resources))
        if cached is not None and cached[0] is resources:
            return cached[1]
        found: DefaultSpaces | None = None
        try:
            color_spaces = resolve_resource_dict(raw, self.resolver)
        except PdfParseError:
            color_spaces = None
        values = default_space_values(color_spaces, self.resolver.deep_resolve)
        spaces: dict[str, ColorSpace] = {}
        for device, value in values.items():
            try:
                space = self.parse_color_space_value(value)
            except ValueError, TypeError, PdfParseError:
                continue
            if valid_default_space(device, space):
                spaces[device] = space
        if spaces:
            found = DefaultSpaces({key: values[key] for key in spaces}, spaces)
        self.default_spaces_cache[id(resources)] = (resources, found)
        return found

    def append_text(self, data: bytes | memoryview, *, decoder: FontService | None = None) -> None:
        """Show a string's glyphs at the text matrix, and move it past them."""
        decoder = decoder if decoder is not None else self.get_decoder()
        show_string(
            self, data if isinstance(data, bytes) else bytes(data), decoder, TEXT_SPEC_CONTEXT
        )

    def append_decoded_text(
        self,
        text: str,
        data: bytes | memoryview,
        glyphs: tuple[DecodedFontGlyph, ...],
        decoder: FontService,
    ) -> None:
        """append_text for a string whose glyphs are already decoded: a Type 3 font's glyph
        procedures run, the sink shows the text, the text matrix moves past it by the
        font's advance, and the sink gets the "shown" boundary."""
        show_decoded(self, text, data, glyphs, decoder, TEXT_SPEC_CONTEXT)

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
                previous_inherited = self.inherited_uncolored
                self.sink.text_boundary(self, "type3-glyph-begin")
                # A glyph shown where colour is already set elsewhere keeps the restriction,
                # whether it begins with d0 or d1 (8.6.8).
                self.inherited_uncolored = previous_type3_uncolored
                try:
                    self.stream_executor.consume(
                        char_proc, resources, glyph_ctm, self.xobject_depth + 1
                    )
                finally:
                    self.type3_uncolored = previous_type3_uncolored
                    self.inherited_uncolored = previous_inherited
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
        """TJ: each run of strings shown as append_text shows it, each number moving the text
        matrix back by its thousandths of the font size."""
        if not isinstance(array, (list, tuple)):
            self.reject(PdfParseError("TJ requires an array"), "text-array", None)
            return
        if not array:
            return
        show_text_array(self, array, self.get_decoder(), TEXT_SPEC_CONTEXT)

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
            if self.type3_uncolored and not self.paints_image_mask(
                getattr(operands[0], "dictionary", None)  # noqa: B009
            ):
                return
            self.sink.paint_inline_image(self, operands[0])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def op_BDC(self, operands: ContentOperands, depth: int) -> None:
        self.sink.text_boundary(self, "marked")
        tag = self.resolver.resolve_name(operands[0]) if operands else None
        layer: str | None = None
        actual_text: str | None = None
        expansion: Expansion | None = None
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
                    expansion_value = expansion_text(props, resolver.resolve_str)
                    if expansion_value is not None:
                        expansion = Expansion(expansion_value)
                mcid = resolver.resolve_int(props.get("MCID"))
        self.marked_content_stack.append(
            MarkedContentEntry(layer=layer, actual_text=actual_text, mcid=mcid, expansion=expansion)
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
                self.reject(PdfParseError("line width must not be negative"), "line-width", None)
            self.graphics.line_width = max(0.0, values[0])

    def op_J(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.graphics.line_cap = value

    def op_j(self, operands: ContentOperands, depth: int) -> None:
        if (value := self.as_int_operand(operands)) is not None:
            self.graphics.line_join = value

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            if values[0] < 1:
                self.reject(PdfParseError("miter limit must be at least one"), "miter-limit", None)
            self.graphics.miter_limit = max(1.0, values[0])

    def op_d(self, operands: ContentOperands, depth: int) -> None:
        if len(operands) == 2 and type(operands[0]) in (list, tuple):
            numbers: typing.Sequence[float] = operands[0]  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            phase_number: float = operands[1]  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            if not numbers and phase_number == 0 and type(phase_number) in NUMERIC_TYPES:
                # "[] 0 d", the solid line reset, is by far the most common dash.
                self.graphics.dash_pattern = ((), float(phase_number))
                return
            if type(phase_number) in NUMERIC_TYPES and NUMERIC_TYPES.issuperset(map(type, numbers)):
                try:
                    dash_array = tuple(map(float, numbers))
                    phase_value = float(phase_number)
                except OverflowError:
                    pass
                else:
                    if isfinite(phase_value) and all(map(isfinite, dash_array)):
                        self.graphics.dash_pattern = (dash_array, phase_value)
                        return
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
        count = len(space.component_ranges)
        operand_count = len(operands)
        if self.type3_uncolored or operand_count < count:
            return
        normalized = self.normalize_color_components(
            space, operands if operand_count == count else operands[:count]
        )
        if normalized is None:
            return
        defaults = self.current_default_spaces()
        if defaults is not None:
            space, normalized = defaults.remap(space, normalized)
        self.set_paint(space, normalized, None, stroke=stroke)

    def set_paint(
        self,
        space: ColorSpace | None,
        color: tuple[float, ...] | None,
        pattern: PatternPaint | None,
        *,
        stroke: bool,
    ) -> None:
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
        defaults = self.current_default_spaces()
        if name in {"DeviceGray", "DeviceRGB", "DeviceCMYK", "Pattern"}:
            if defaults is not None and name in defaults.spaces:
                return defaults.spaces[name]
            return self.parse_named_color_space(name, name)
        raw = self.lookup_page_resource("ColorSpace", name)
        if defaults is not None:
            remapped = self.resolve_remapped_color_space(raw, defaults)
            if remapped is not None:
                return remapped
        cache = self.color_space_cache if type(raw) is PdfReference else None
        if cache is not None:
            cached = cache.get(id(raw))
            if cached is not None and cached[0] is raw:
                return cached[1]
        value: object = self.resolver.deep_resolve(raw)
        if value is None:
            value = self.reject(PdfParseError("missing color space resource"), "color-space", name)
            return self.parse_named_color_space(value, name)
        try:
            space = self.parse_color_space_value(value)
        except ValueError, TypeError:
            return self.parse_named_color_space(value, name)
        if cache is not None:
            cache[id(raw)] = (raw, space)
        return space

    def resolve_remapped_color_space(
        self, raw: object, defaults: DefaultSpaces
    ) -> ColorSpace | None:
        """A ColorSpace resource whose device spaces the default spaces remap, or None for
        one they leave as it is (8.6.5.6)."""
        cached = self.remapped_space_cache.get(id(raw))
        if cached is not None and cached[0] is raw and cached[1] is defaults:
            return cached[2]
        value = self.resolver.deep_resolve(raw)
        name = self.resolver.resolve_name(value)
        if name is not None:
            return defaults.spaces.get(name)
        remapped = remap_color_space(value, defaults.values)
        if remapped is value:
            return None
        try:
            space = self.parse_color_space_value(remapped)
        except ValueError, TypeError:
            return None
        if type(raw) is PdfReference:
            self.remapped_space_cache[id(raw)] = (raw, defaults, space)
        return space

    def parse_color_space_value(self, value: object) -> ColorSpace:
        return parse_color_space(value)

    def parse_named_color_space(self, value: object, name: str) -> ColorSpace:
        try:
            return self.parse_color_space_value(value)
        except (ValueError, TypeError) as error:
            return self.reject(parse_error_from(error), "color-space", ColorSpace(name, ()))

    def color_component_values(
        self, components: typing.Sequence[object]
    ) -> typing.Sequence[object]:
        return components

    def recover_color_components(
        self, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        return None

    def normalize_color_components(
        self, spec: ColorSpace, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        try:
            return normalize_color_components(spec, self.color_component_values(components))
        except (TypeError, ValueError, PdfParseError) as error:
            failure = parse_error_from(error) if isinstance(error, ValueError) else error
            fallback = (
                None
                if spec.kind in {"Indexed", "Lab"}
                else self.recover_color_components(components)
            )
            return self.reject(failure, "color-components", fallback)

    def initial_color_components(
        self, spec: ColorSpace, *, stroke: bool
    ) -> tuple[float, ...] | None:
        try:
            return initial_color_components(spec)
        except ValueError as error:
            current = self.graphics.stroke_color if stroke else self.graphics.fill_color
            return self.reject(parse_error_from(error), "color-space", current)

    def set_color_space(self, operands: ContentOperands, *, stroke: bool) -> None:
        if self.type3_uncolored:
            return
        if operands:
            space = self.resolve_color_space(operands[0])
            defaults = self.current_default_spaces()
            color = defaults.initial_color(space) if defaults is not None else None
            if color is None:
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
        if not operands or self.type3_uncolored:
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
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        if not self.compatibility_depth:
            self.reject(PdfParseError("unmatched EX operator"), "compatibility", None)
            return
        self.compatibility_depth -= 1

    def op_d0(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = self.inherited_uncolored

    def op_d1(self, operands: ContentOperands, depth: int) -> None:
        self.type3_uncolored = True

    def resource_operand_name(self, operands: ContentOperands) -> str | None:
        name = self.resolver.resolve_name(operands[0]) if operands else None
        if not name:
            return self.reject(PdfParseError("resource operator requires a name"), "resource", None)
        return name

    @cites(
        "ISO 32000-2",
        "8.7.4.3",
        introduced="1.3",
        rule="shadings of types 4 to 7 are streams, whose data holds their geometry",
    )
    def resolve_shading(self, value: object) -> PdfDict | None:
        """A shading dictionary. One read from a stream keeps the stream, whose data a mesh
        shading needs, under SHADING_STREAM_KEY; each stream gives the same dictionary."""
        resolved = self.resolver.resolve(value)
        for _ in range(8):
            if type(resolved) is not PdfReference:
                break
            resolved = self.resolver.resolve(resolved)
        if type(resolved) is not PdfStream:
            return self.resolver.resolve_dict(resolved)
        cached = self.shading_streams.get(id(resolved))
        if cached is not None and cached[0] is resolved:
            return cached[1]
        dictionary = self.resolver.resolve_dict(resolved.dictionary)
        if dictionary is None:
            return None
        shading = dict(dictionary)
        shading[SHADING_STREAM_KEY] = resolved
        self.shading_streams[id(resolved)] = (resolved, shading)
        return shading

    def op_sh(self, operands: ContentOperands, depth: int) -> None:
        if self.type3_uncolored:
            return
        name = self.resource_operand_name(operands)
        if name is None:
            return
        shading = self.resolve_shading(self.lookup_page_resource("Shading", name))
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
            return self.reject(PdfParseError("missing numeric operand"), "numeric-operands", None)
        if len(operands) == count and type(operands) is tuple:
            if count == 1 and type(operands[0]) is int:
                # A lone integer, as in "1 w"; a converted integer is always finite.
                try:
                    return (float(operands[0]),)
                except OverflowError:
                    pass
            for value in operands:
                if type(value) is not float or not isfinite(value):
                    break
            else:
                return operands  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
            if all(map(NUMERIC_TYPES.__contains__, map(type, operands))):
                try:
                    converted = tuple(map(float, operands))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                except OverflowError:
                    pass
                else:
                    if all(map(isfinite, converted)):
                        return converted
        try:
            values: list[float] = []
            append = values.append
            for index in range(count):
                value = operands[index]
                kind = type(value)
                if kind is float and isfinite(value):  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    append(value)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                elif kind is int:
                    try:
                        append(float(value))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    except OverflowError:
                        append(self.as_float(value))
                else:
                    append(self.as_float(value))
            return tuple(values)
        except (TypeError, ValueError, PdfParseError) as error:
            return self.reject(error, "numeric-operands", None)

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        raise PdfParseError("integer operand must be a PDF integer")

    def as_int_operand(self, operands: ContentOperands) -> int | None:
        if not operands:
            return self.reject(PdfParseError("missing numeric operand"), "integer-operand", None)
        value = operands[0]
        if type(value) is int:
            return value
        try:
            return self.as_int(value)
        except (TypeError, ValueError, PdfParseError) as error:
            return self.reject(error, "integer-operand", None)

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
        except PdfParseError as error:
            # Declined as the others are, rather than ending the content stream; a strict
            # interpreter's reject raises it.
            self.reject(error, "extended-graphics-state", None)

    def apply_extgstate(self, extgstate: dict[str, Any]) -> None:
        # Three groups, each applied whatever happens to the others: the older entries,
        # where the first to fail ends the rest, as it always has; Font; and the overprint
        # entries. A TypeError or ValueError from Font or overprint goes through reject with
        # the group's context. Every error a group ends with (a PdfParseError, raised by a
        # strict reject or by reading the entries, or the older entries' TypeError or
        # ValueError) is held, unchanged, until all three have run; the first is raised
        # then, and op_gs declines it through reject, which a strict interpreter raises.
        failures: list[Exception] = []
        try:
            self.apply_extgstate_entries(extgstate)
        except (TypeError, ValueError, PdfParseError) as error:
            failures.append(error)
        if "Font" in extgstate:
            self.apply_extgstate_group(
                lambda: self.apply_extgstate_font(extgstate["Font"]), "font-operands", failures
            )
        if "OP" in extgstate or "op" in extgstate or "OPM" in extgstate:
            self.apply_extgstate_group(
                lambda: self.apply_extgstate_overprint(extgstate), "overprint", failures
            )
        if failures:
            raise failures[0]

    def apply_extgstate_group(
        self, apply: Callable[[], None], context: str, failures: list[Exception]
    ) -> None:
        """Apply one group of a parameter dictionary's entries. A TypeError or ValueError
        is declined through reject as a PdfParseError of context; a PdfParseError, which a
        strict reject inside may already have raised, is added to failures as it is, as is
        whatever the decline raises."""
        try:
            apply()
        except PdfParseError as error:
            failures.append(error)
        except (TypeError, ValueError) as error:
            try:
                self.reject(parse_error_from(error), context, None)
            except PdfParseError as raised:
                failures.append(raised)

    def apply_extgstate_entries(self, extgstate: dict[str, Any]) -> None:
        self.apply_extgstate_line_style(extgstate)
        intent = self.resolver.resolve(extgstate.get("RI"))
        if intent is not None:
            self.graphics.render_intent = parse_rendering_intent(self.named_value(intent))
        black_point = self.resolver.resolve(extgstate.get("UseBlackPtComp"))
        if black_point is not None and not self.type3_uncolored:
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
            resolved_mode = self.resolver.resolve(blend_mode)
            if isinstance(resolved_mode, (list, tuple)):
                self.graphics.blend_mode = first_recognized_blend_mode(
                    [self.named_value(name) for name in resolved_mode]
                )
            else:
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

    @cites(
        "ISO 32000-2",
        "8.4.5",
        introduced="1.3",
        rule="LW, LC, LJ, ML and D set the line width, cap, join, miter limit and dash",
    )
    def apply_extgstate_line_style(self, extgstate: dict[str, Any]) -> None:
        """A parameter dictionary's line style entries, each applied as its operator, w,
        J, j, M or d, applies its operands (Table 57). These are this class's operators, not
        a subclass's: a facade that mirrors another reader's operators, one that ignores gs,
        sees no w or d here."""
        resolve = self.resolver.resolve
        line_width = extgstate.get("LW")
        if line_width is not None:
            ContentInterpreter.op_w(self, (resolve(line_width),), 0)
        line_cap = extgstate.get("LC")
        if line_cap is not None:
            ContentInterpreter.op_J(self, (resolve(line_cap),), 0)
        line_join = extgstate.get("LJ")
        if line_join is not None:
            ContentInterpreter.op_j(self, (resolve(line_join),), 0)
        miter_limit = extgstate.get("ML")
        if miter_limit is not None:
            ContentInterpreter.op_M(self, (resolve(miter_limit),), 0)
        dash = resolve(extgstate.get("D"))
        if dash is not None:
            if not isinstance(dash, (list, tuple)) or len(dash) != 2:
                self.reject(
                    PdfParseError("D must hold a dash array and a phase"), "dash-pattern", None
                )
                return
            array = resolve(dash[0])
            if isinstance(array, (list, tuple)):
                array = [resolve(value) for value in array]
            ContentInterpreter.op_d(self, (array, resolve(dash[1])), 0)

    @cites(
        "ISO 32000-2",
        "8.4.5",
        introduced="1.3",
        rule="Font, an array of a font dictionary and a size, sets the text font as Tf does",
    )
    def apply_extgstate_font(self, value: object) -> None:
        """A parameter dictionary's Font: an indirect reference to a font dictionary and a
        size, which select the text font and size as Tf does with a Font resource's name
        (Table 57). The font is named by the reference, so the state names it by its
        BaseFont. A facade mirroring a reader that ignores it overrides this. Loading the
        font may fail as it may for Tf: apply_extgstate declines a TypeError, ValueError or
        PdfParseError, and anything else Tf would let through passes on here too."""
        resolve = self.resolver.resolve
        entry = resolve(value)
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            self.reject(PdfParseError("Font must hold a font and a size"), "font-operands", None)
            return
        font_reference = entry[0]
        font = resolve(font_reference)
        if isinstance(font, PdfStream) or not isinstance(font, dict):
            self.reject(PdfParseError("Font must name a font dictionary"), "font-resource", None)
            return
        font_size = self.parse_font_size(resolve(entry[1]))
        if font_size is None:
            return
        name = self.resolver.resolve_name(font.get("BaseFont")) or "Font"
        # The decoder first, so that a font that cannot be loaded leaves the state whole.
        decoder = self.decoder_for(font_reference, font, self.resources)
        graphics = self.graphics
        graphics.current_font = name
        graphics.font_size = font_size
        graphics.current_decoder = decoder
        # No resources, so that the next Tf loads its font even when its resource name is
        # this font's BaseFont, rather than keep this decoder.
        graphics.decoder_resources = None

    @cites(
        "ISO 32000-2",
        "8.6.7",
        introduced="1.3",
        rule="OP and op set overprint for stroking and other painting, OPM the overprint mode",
    )
    def apply_extgstate_overprint(self, extgstate: dict[str, Any]) -> None:
        """A parameter dictionary's overprint entries (Table 57): OP, whether stroking
        overprints, and op, whether other painting does, which OP also sets when op is
        absent; OPM, the overprint mode, 0 or 1 (8.6.7). The state carries them to capture,
        which records the colorants each overprinting paint marks; the renderer overprints
        them where paint composites in a DeviceCMYK blending space (ISO 32000-2 11.7.4)."""
        resolve = self.resolver.resolve
        graphics = self.graphics
        flags: dict[str, bool] = {}
        for key in ("OP", "op"):
            flag = resolve(extgstate.get(key))
            if flag is None:
                continue
            if not isinstance(flag, bool):
                # Declined, and taken as absent: an invalid op leaves OP to set both.
                self.reject(PdfParseError(f"{key} must be a boolean"), "overprint", None)
                continue
            flags[key] = flag
        if "OP" in flags:
            graphics.stroke_overprint = flags["OP"]
        if "op" in flags or "OP" in flags:
            graphics.fill_overprint = flags.get("op", flags.get("OP", False))
        mode = resolve(extgstate.get("OPM"))
        if mode is not None:
            if type(mode) is int and mode in (0, 1):
                graphics.overprint_mode = mode
            else:
                self.reject(PdfParseError("OPM must be 0 or 1"), "overprint", None)

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

    @cites(
        "ISO 32000-2",
        "8.7.2",
        introduced="1.2",
        rule="a pattern's space: its Matrix on the parent content stream's default space",
    )
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
            shading_dict = self.resolve_shading(shading) if shading is not None else None
            if not isinstance(shading_dict, dict):
                return self.reject(PdfParseError(invalid), "pattern", None)
            extgstate = self.resolver.resolve_dict(pattern_dict.get("ExtGState"))
            matrix = self.matrix_operand(pattern_dict.get("Matrix"), "pattern")
            return ShadingPattern(
                dict(shading_dict), extgstate=extgstate, matrix=matrix.multiply(self.stream_ctm)
            )
        if pattern_type != 1 or not isinstance(pattern, PdfStream):
            return self.reject(PdfParseError(invalid), "pattern", None)
        paint_type = self.resolver.resolve_int(pattern_dict.get("PaintType"))
        if paint_type is None:
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
            matrix=matrix.multiply(self.stream_ctm),
            paint_type=paint_type,
            base_color=base_color,
            base_color_spec=base_spec,
            alpha_is_shape=self.initial_alpha_is_shape,
            text_knockout=self.initial_text_knockout,
        )


__all__ = ("ContentInterpreter",)
