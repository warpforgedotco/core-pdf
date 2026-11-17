# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.base.caches import IdentityCache
from core_pdf.impl.base.geometry import clamp01, intersect_bbox, transform_bbox
from core_pdf.impl.base.glyphs import GlyphObservation
from core_pdf.impl.base.runs import TextRun
from core_pdf.impl.base.text import word_gap_threshold
from core_pdf.impl.base.types import MISSING, MissingObject, Rectangle
from core_pdf.impl.capture.glyphs import GlyphPaint
from core_pdf.impl.capture.program import DEFAULT_CAPTURE, CapturedProgram, CaptureOptions
from core_pdf.impl.capture.records import (
    CapturedDrawing,
    CapturedInlineImage,
    CapturedPath,
    CapturedSoftMask,
    CapturedSubpath,
    CapturedTextBoundary,
    PatternPaint,
    TilingPattern,
    marker_drawing,
)
from core_pdf.impl.capture.recovery import CaptureRecovery, ContentFallback, content_scanner
from core_pdf.impl.capture.tolerant_state import CaptureCaches, RecoveringTextState
from core_pdf.impl.fonts.decoder import FontDecoder
from core_pdf.impl.fonts.ligatures import detect_ligature_overrides
from core_pdf.impl.graphics.color import color_operands_to_cmyk, color_operands_to_srgb
from core_pdf.impl.graphics.color_spec import GRAY_RGB_KINDS
from core_pdf.impl.recovery.lexer import PdfLexer
from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content import model
from core_pdf.impl.spec.s_07_content.interpreter import ContentInterpreter
from core_pdf.impl.spec.s_07_content.model import GraphicsState
from core_pdf.impl.spec.s_07_content.operations import OperationHandler
from core_pdf.impl.spec.s_07_content.streams import ContentStreamExecutor, ContentStreamFrame
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_07_syntax_primitives.coercion import parse_float_strict, parse_int_strict
from core_pdf.impl.spec.s_07_syntax_primitives.content_operators import CONTENT_OPERATOR_HANDLERS
from core_pdf.impl.spec.s_08_graphics.color_rendering import ColorRendering
from core_pdf.impl.spec.s_08_graphics.color_spec import DEVICE_CMYK, DEVICE_GRAY, DEVICE_RGB
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
lazy from core_pdf.impl.spec.s_11_transparency.blending_space import (
    blending_profile,
    blending_space_of,
)
lazy from core_pdf.impl.spec.s_11_transparency.soft_masks import SoftMask, colorimetric_luminosity
lazy from core_pdf_cythonized import NativeOperators, RunAccumulator

if TYPE_CHECKING:
    from core_pdf.impl.capture.records import LayoutFormId, StrokeMatrix
    from core_pdf.impl.document.pdf_document import PdfDocument
    from core_pdf.impl.recovery.resolver import ObjectResolver
    from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace
    from core_pdf.impl.spec.s_08_graphics.default_spaces import DefaultSpaces
    from core_pdf.impl.spec.s_11_transparency.blending_space import BlendingSpace


from core_pdf.impl.capture import image_painting, paint_sources, path_painting, text_painting
from core_pdf.impl.capture.recording_support import (
    GLYPH_PAINT_KEEPING_OPERATORS,
    LINE_MOVING_OPERATORS,
    MASK_KEYED_FIELDS,
    NO_MARKS,
    TEXT_LAYOUT_KEEPING_OPERATORS,
    CaptureColorMemo,
    CaptureGraphicsSave,
    CaptureMarks,
    MarkedContentEntry,
    StrokeLineRows,
    TextLayout,
    state_key,
)

NATIVE_PATH_HANDLERS: dict[str, Callable[..., None]] = {
    "m": ContentInterpreter.op_m,
    "l": RecoveringTextState.op_l,
    "c": ContentInterpreter.op_c,
    "v": RecoveringTextState.op_v,
    "y": RecoveringTextState.op_y,
    "re": ContentInterpreter.op_re,
    "h": ContentInterpreter.op_h,
}


def applies_paths_natively(handlers: Mapping[str, OperationHandler], state: object) -> bool:
    kind = type(state)
    if (
        getattr(kind, "append_cubic_curve", None) is not RecoveringTextState.append_cubic_curve
        or getattr(kind, "as_floats", None) is not RecoveringTextState.as_floats
    ):
        return False
    for name, function in NATIVE_PATH_HANDLERS.items():
        if getattr(handlers.get(name), "__func__", None) is not function:
            return False
    return True


NATIVE_STATE_HANDLERS: dict[str, Callable[..., None]] = {
    "G": ContentInterpreter.op_G,
    "g": ContentInterpreter.op_g,
    "RG": ContentInterpreter.op_RG,
    "rg": ContentInterpreter.op_rg,
    "K": ContentInterpreter.op_K,
    "k": ContentInterpreter.op_k,
    "w": ContentInterpreter.op_w,
    "d": ContentInterpreter.op_d,
    "J": ContentInterpreter.op_J,
    "j": ContentInterpreter.op_j,
    "M": ContentInterpreter.op_M,
    "Td": ContentInterpreter.op_Td,
    "TD": ContentInterpreter.op_TD,
    "Tm": ContentInterpreter.op_Tm,
    "T*": ContentInterpreter.op_T_star,
    "TL": ContentInterpreter.op_TL,
    "q": ContentInterpreter.op_q,
    "Q": ContentInterpreter.op_Q,
    "cm": ContentInterpreter.op_cm,
}


# The methods those handlers reach for the operands the scanner applies itself.
NATIVE_STATE_METHODS: dict[str, Callable[..., object]] = {
    "as_floats": ContentInterpreter.as_floats,
    "as_int_operand": ContentInterpreter.as_int_operand,
    "reject": RecoveringTextState.reject,
    "set_device_color": ContentInterpreter.set_device_color,
    "set_paint": ContentInterpreter.set_paint,
    "normalize_color_components": RecoveringTextState.normalize_color_components,
    "color_component_values": RecoveringTextState.color_component_values,
    "move_text": ContentInterpreter.move_text,
    "pop_graphics_save": ContentInterpreter.pop_graphics_save,
}


def native_state_operators(
    handlers: Mapping[str, OperationHandler], state: object
) -> frozenset[str]:
    """The operators ContentScanner may apply to state instead of calling their handlers.

    Only the spec handlers, over the helpers they are known to call, and only
    on a GraphicsState: anything overridden keeps its handler.
    """
    kind = type(state)
    if type(getattr(state, "graphics", None)) is not GraphicsState:
        return frozenset()
    for name, function in NATIVE_STATE_METHODS.items():
        if getattr(kind, name, None) is not function:
            return frozenset()
    if getattr(kind, "as_float", None) not in (ContentInterpreter.as_float, TextState.as_float):
        return frozenset()
    return frozenset(
        name
        for name, function in NATIVE_STATE_HANDLERS.items()
        if getattr(handlers.get(name), "__func__", None) is function
    )


DEFAULT_OPERATOR_NAMES: frozenset[bytes] = frozenset(
    name.encode("latin-1") for name in CONTENT_OPERATOR_HANDLERS
)


@lru_cache(maxsize=16)
def cached_native_operators(
    operators: frozenset[str],
    gray: ColorSpace = DEVICE_GRAY,
    rgb: ColorSpace = DEVICE_RGB,
    cmyk: ColorSpace = DEVICE_CMYK,
) -> NativeOperators:
    return NativeOperators(
        operators,
        GLYPH_PAINT_KEEPING_OPERATORS,
        LINE_MOVING_OPERATORS,
        TEXT_LAYOUT_KEEPING_OPERATORS,
        gray,
        rgb,
        cmyk,
        Matrix,
    )


# The operators that set a device colour, which a stream's default spaces remap.
DEVICE_COLOR_OPERATORS = frozenset({"G", "g", "RG", "rg", "K", "k"})


def remapped_native_operators(
    operators: frozenset[str], defaults: DefaultSpaces
) -> NativeOperators | None:
    """The native operators of a stream whose resources have default colour spaces: the
    device colour operators set the default spaces, when their components range over 0 to
    1 as the native operators clamp to, and are left to their handlers otherwise."""
    if defaults.unit_ranges:
        spaces = defaults.spaces
        return cached_native_operators(
            operators,
            spaces.get("DeviceGray", DEVICE_GRAY),
            spaces.get("DeviceRGB", DEVICE_RGB),
            spaces.get("DeviceCMYK", DEVICE_CMYK),
        )
    remaining = operators - DEVICE_COLOR_OPERATORS
    return cached_native_operators(remaining) if remaining else None


class CaptureStreamExecutor(ContentStreamExecutor):
    state: TextState  # ty: ignore[invalid-mutable-override]
    _operator_names: frozenset[bytes] | None = None
    _native_operators: tuple[object, frozenset[str], NativeOperators | None] | None = None

    max_depth = 10

    def reject_reentry(self, frame: ContentStreamFrame) -> bool:
        return True

    def enter(self, frame: ContentStreamFrame) -> bool:
        if not super().enter(frame):
            return False
        self.state.shared_glyph_paint = None
        self.state.text_layout = None
        return True

    def exit(self, frame: ContentStreamFrame) -> None:
        try:
            super().exit(frame)
        finally:
            self.state.shared_glyph_paint = None
            self.state.text_layout = None

    def operator_names(self, table: Mapping[str, OperationHandler]) -> frozenset[bytes]:
        state = self.state
        if table is not state.default_handlers:
            return frozenset(name.encode("latin-1") for name in table)
        if len(table) == len(DEFAULT_OPERATOR_NAMES):
            return DEFAULT_OPERATOR_NAMES
        names = self._operator_names
        if names is None:
            names = self._operator_names = frozenset(name.encode("latin-1") for name in table)
        return names

    def native_operators(self, table: Mapping[str, OperationHandler]) -> NativeOperators | None:
        state = self.state
        if type(state.graphics) is not GraphicsState:
            return None
        cached = self._native_operators
        default = table is state.default_handlers
        if default and cached is not None and cached[0] is table:
            operators, native = cached[1], cached[2]
        else:
            operators = native_state_operators(table, state)
            native = cached_native_operators(operators) if operators else None
            if default:
                self._native_operators = (table, operators, native)
        if native is None:
            return None
        defaults = state.current_default_spaces()
        return native if defaults is None else remapped_native_operators(operators, defaults)

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        handlers = state.operation_table()
        lexer = frame.lexer
        scanner = content_scanner(lexer, state if applies_paths_natively(handlers, state) else None)
        native = self.native_operators(handlers)
        if native is not None:
            scanner.set_native_state(state, native)
        fallback = ContentFallback(
            lexer,
            scanner.operands,
            state.content_recovery,
            self.operator_names(handlers).__contains__,
        )
        return scanner.run(
            lexer,
            cast("dict[str, OperationHandler]", handlers)
            if type(handlers) is dict
            else dict(handlers),
            fallback,
            state,
            frame.depth,
            GLYPH_PAINT_KEEPING_OPERATORS,
            LINE_MOVING_OPERATORS,
            TEXT_LAYOUT_KEEPING_OPERATORS,
        )

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error


class TextState(RecoveringTextState):
    document: PdfDocument[Any]
    name_resolver: ObjectResolver
    runs: list[TextRun]
    glyphs: list[GlyphObservation]
    glyph_cluster_count: int
    lines: StrokeLineRows
    drawings: list[CapturedDrawing]
    inline_images: list[CapturedInlineImage]
    hidden_layers: frozenset[str]
    page_clip: Rectangle | None
    options: CaptureOptions
    clip_bbox: Rectangle | None
    layout_form_bbox: Rectangle | None
    layout_form_id: LayoutFormId
    capture_source: str
    stream_order: int
    sequence: int
    text_object_id: int
    text_boundaries: list[CapturedTextBoundary]
    capture_text_open: bool
    capture_text_frames: dict[int, tuple[bool, bool]]
    pending_line_break: bool
    group_alpha: float | None
    run_accumulator: RunAccumulator
    capture_graphics_stack: list[CaptureGraphicsSave]
    capture_marked_entries: dict[int, MarkedContentEntry]
    capture_frames: dict[int, tuple[Rectangle | None, LayoutFormId, bool]]
    capture_patterns: IdentityCache[PatternPaint | None]
    scale_cache: tuple[Matrix, float, StrokeMatrix | None] | None
    shared_glyph_paint: GlyphPaint | None
    # Whether paint lands in a DeviceCMYK blending space (the page group's, or an isolated
    # group's), where a colour that comes down to CMYK is captured as that CMYK, and the
    # space's ICC profile (None: DeviceCMYK, the built-in profile) the CMYK is of; and the
    # values to restore when each group that changed them ends.
    capture_cmyk: bool
    capture_cmyk_profile: bytes | None
    capture_cmyk_frames: dict[int, tuple[bool, bytes | None]]
    text_layout: TextLayout | None
    stream_executor: CaptureStreamExecutor
    stream_executor_type = CaptureStreamExecutor

    def __init__(
        self,
        document: PdfDocument[Any],
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
        *,
        options: CaptureOptions = DEFAULT_CAPTURE,
        caches: CaptureCaches | None = None,
    ):
        self.document = document
        self.name_resolver = document.resolver
        self.runs = []
        self.glyphs = []
        self.glyph_cluster_count = 0
        self.lines = StrokeLineRows()
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.options = options
        self.clip_bbox = None
        self.layout_form_bbox = None
        self.layout_form_id = None
        self.capture_source = "native_text"
        self.stream_order = -1
        self.sequence = 0
        self.text_object_id = 0
        self.text_boundaries = []
        self.capture_text_open = False
        self.capture_text_frames = {}
        self.pending_line_break = False
        self.group_alpha = None
        self.run_accumulator = RunAccumulator(self.runs, word_gap_threshold)
        self.capture_graphics_stack = []
        self.capture_marked_entries = {}
        self.capture_frames = {}
        self.capture_patterns = IdentityCache()
        self.last_fill_color: CaptureColorMemo | None = None
        self.last_stroke_color: CaptureColorMemo | None = None
        self.capture_cmyk = False
        self.capture_cmyk_profile = None
        self.capture_cmyk_frames = {}

        raster_font_provider = getattr(document, "raster_font_provider", None)
        resolver_context = getattr(document.resolver, "semantic_context", None)
        font_context = getattr(document, "font_semantic_context", resolver_context)

        def font_provider(font: PdfDict, resources: PdfDict) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=raster_font_provider,
                semantic_context=font_context,
            )

        super().__init__(
            document.resolver,
            sink=self,
            font_provider=font_provider,
            lexer_factory=PdfLexer,
            semantic_context=resolver_context,
            caches=caches,
        )

        self.content_recovery = CaptureRecovery()
        self.scale_cache = None
        self.shared_glyph_paint = None
        self.text_layout = None

        self.graphics.font_size = 12.0
        self.graphics.fill_color = (0.0, 0.0, 0.0)
        self.graphics.stroke_color = (0.0, 0.0, 0.0)

    @staticmethod
    def as_float(value: Any) -> float:
        parsed = parse_float_strict(value, "invalid numeric operand", python_syntax=True)
        if not isfinite(parsed):
            raise ValueError("invalid numeric operand")
        return parsed

    @staticmethod
    def as_int(value: Any) -> int:
        return parse_int_strict(value, "invalid numeric operand", python_syntax=True)

    def capture_marks(self) -> CaptureMarks:
        return (
            len(self.runs),
            len(self.glyphs),
            len(self.drawings),
            len(self.inline_images),
            self.lines.count,
            len(self.text_boundaries),
        )

    def captured_program(self, since: CaptureMarks = NO_MARKS) -> CapturedProgram:
        runs, glyphs, drawings, inline_images, lines, text_boundaries = since
        return CapturedProgram(
            runs=tuple(self.runs[runs:]),
            glyphs=tuple(self.glyphs[glyphs:]),
            drawings=tuple(self.drawings[drawings:]),
            inline_images=tuple(self.inline_images[inline_images:]),
            lines=self.lines.since(lines),
            text_boundaries=tuple(self.text_boundaries[text_boundaries:]),
            options=self.options,
        )

    def release(self) -> None:
        del self.sink
        del self.stream_executor
        self.default_handlers.clear()

    def is_graphics_visible(self) -> bool:
        for entry in self.marked_content_stack:
            if entry.layer and entry.layer in self.hidden_layers:
                return False
        return True

    def nested_capture_state(self) -> TextState:
        return TextState(
            self.document,
            hidden_layers=self.hidden_layers,
            options=self.options,
            caches=self.caches,
        )

    def named_value(self, value: object, *, allow_text: bool = False) -> str | None:
        resolver = self.name_resolver
        if allow_text:
            return resolver.resolve_name_or_text(value)
        return resolver.resolve_name_like_value(value)

    def end_marked_content(self, state: object, entry: model.MarkedContentEntry) -> None:
        captured = self.capture_marked_entries.pop(id(entry), None)
        if captured is not None:
            self.emit_actual_text_span(captured)

    def complete_path(
        self, kind: str | None, fill_rule: str = "nonzero", *, close: bool = False
    ) -> None:
        # paint_path ignores empty paths, so an empty path with no clip pending
        # is kept rather than replaced; it was never handed to a sink.
        path = self.current_path
        if not close and self.pending_clip_rule_value is None and not path.ops:
            self.current_point = None
            self.subpath_start = None
            return
        if self.sink is not self:
            super().complete_path(kind, fill_rule, close=close)
            return
        if close:
            self.close_current_subpath()
        if kind is not None:
            self.paint_path(self, path, kind, fill_rule)
        if self.pending_clip_rule_value is not None:
            self.clip_path(self, path, self.pending_clip_rule_value)
        # paint_path and clip_path copy the path as they flatten it, so it is emptied
        # for the next one rather than replaced.
        del path.ops[:]
        del path.coords[:]
        self.current_point = None
        self.subpath_start = None
        self.pending_clip_rule_value = None

    def save_graphics(self, state: object) -> None:
        self.capture_graphics_stack.append(CaptureGraphicsSave(self.clip_bbox, self.group_alpha))

    def restore_graphics(self, state: object) -> None:
        saved = self.capture_graphics_stack.pop()
        self.clip_bbox = saved.clip_bbox
        self.group_alpha = saved.group_alpha
        if saved.clip_scope_emitted:
            self.drawings.append(marker_drawing("state-pop", self.sequence))
            self.sequence += 1

    def enter_stream(self, state: object, frame: ContentStreamFrame) -> None:
        previous_text_open = self.capture_text_open
        self.capture_text_frames[id(frame)] = (previous_text_open, False)
        self.capture_text_open = False
        self.capture_frames[id(frame)] = (
            self.layout_form_bbox,
            self.layout_form_id,
            self.pending_line_break,
        )
        if frame.form_bbox is not None:
            x0, y0, x1, y1 = frame.form_bbox
            clip_path = CapturedPath()
            if x1 > x0 and y1 > y0:
                clip_path.subpaths.append(
                    CapturedSubpath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True)
                )
                clip_path = clip_path.transformed(frame.ctm)
            self.drawings.append(
                CapturedDrawing(
                    kind="scope-begin",
                    seqno=self.sequence,
                    path=clip_path,
                    fill=None,
                    fill_opacity=None,
                    line_width=0.0,
                )
            )
            self.sequence += 1
        if frame.group_alpha is not None and frame.group_color_space is not None:
            self.capture_cmyk_frames[id(frame)] = (self.capture_cmyk, self.capture_cmyk_profile)
            self.set_capture_cmyk(
                blending_space_of(frame.group_color_space) == "DeviceCMYK",
                blending_profile(frame.group_color_space),
            )
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-begin",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                    graphics_soft_mask=self.capture_graphics_soft_mask(),
                    group_color_space=frame.group_color_space,
                )
            )
            self.sequence += 1
            self.group_alpha = None
        self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-begin"))
        self.capture_text_frames[id(frame)] = (previous_text_open, True)
        layout_bbox = None
        raw_bbox = frame.form_bbox_operand
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x, y, w, h = (
                self.resolver.resolve_float(value, default=None) for value in raw_bbox[:4]
            )
            if x is not None and y is not None and w is not None and h is not None:
                layout_bbox = transform_bbox((x, y, x + w, y + h), frame.ctm)
        self.layout_form_bbox = layout_bbox
        if frame.is_form:
            self.layout_form_id = (*(self.layout_form_id or ()), (frame.source_key, layout_bbox))
        else:
            self.layout_form_id = None
        if frame.clip_bbox is not None:
            self.clip_bbox = intersect_bbox(self.clip_bbox, frame.clip_bbox)
        self.pending_line_break = False
        self.stream_order += 1

    def exit_stream(self, state: object, frame: ContentStreamFrame) -> None:
        if id(frame) in self.capture_text_frames:
            previous_text_open, started = self.capture_text_frames.pop(id(frame))
            if started and self.capture_text_open:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "end"))
            if started:
                self.text_boundaries.append(CapturedTextBoundary(self.sequence, "stream-end"))
            self.capture_text_open = previous_text_open
        old = self.capture_frames.pop(id(frame), None)
        if old is not None:
            self.layout_form_bbox, self.layout_form_id, self.pending_line_break = old
        if self.capture_marked_entries:
            active_entries = {id(entry) for entry in self.marked_content_stack}
            self.capture_marked_entries = {
                key: value
                for key, value in self.capture_marked_entries.items()
                if key in active_entries
            }
        if frame.group_alpha is not None:
            self.drawings.append(
                marker_drawing(
                    "group-end",
                    self.sequence,
                    fill_opacity=frame.group_alpha,
                    blend_mode=self.graphics.blend_mode,
                    soft_mask_alpha=self.group_alpha,
                    group_isolated=frame.group_isolated,
                    group_knockout=frame.group_knockout,
                    alpha_is_shape=self.graphics.alpha_is_shape,
                )
            )
            self.sequence += 1
        if old is not None and frame.form_bbox is not None:
            self.drawings.append(marker_drawing("scope-end", self.sequence))
            self.sequence += 1
        if id(frame) in self.capture_cmyk_frames:
            self.set_capture_cmyk(*self.capture_cmyk_frames.pop(id(frame)))

    def set_capture_cmyk(self, cmyk: bool, profile: bytes | None = None) -> None:
        """Capture paint from now on for a DeviceCMYK blending space (cmyk) whose CMYK is
        that of profile (None: the built-in DeviceCMYK profile), or else for sRGB."""
        profile = profile if cmyk else None
        if (
            cmyk is not self.capture_cmyk or profile is not self.capture_cmyk_profile
        ) and self.options.render_details:
            self.capture_cmyk = cmyk
            self.capture_cmyk_profile = profile
            self.last_fill_color = None
            self.last_stroke_color = None
            self.shared_glyph_paint = None

    def resolve_soft_mask(self, value: object) -> SoftMask | None:
        mask = super().resolve_soft_mask(value)
        if mask is not None:
            scopes = self.caches.capture_mask_resources
            key = id(mask)
            if scopes.get_key(mask, key, MISSING) is MISSING:
                scopes.put_key(mask, key, self.resources)
        return mask

    def capture_tiling_pattern(
        self,
        nested: TextState,
        pattern: model.TilingPattern,
        rendering: ColorRendering,
        initial_alpha_is_shape: bool,
        initial_text_knockout: bool,
    ) -> TilingPattern | None:
        nested.graphics.render_intent = self.graphics.render_intent
        nested.graphics.black_point_compensation = self.graphics.black_point_compensation
        nested.graphics.alpha_is_shape = initial_alpha_is_shape
        nested.graphics.text_knockout = initial_text_knockout
        if pattern.paint_type == 2:
            self.uncolored_cell_state(nested, pattern)
        try:
            nested.stream_executor.consume(pattern.stream, pattern.resources, pattern.matrix, 0)
        except Exception:
            return None
        # The cell ran under its pattern matrix, so the program is in default user space.
        matrix = pattern.matrix
        return TilingPattern(
            pattern.bbox,
            pattern.x_step,
            pattern.y_step,
            CapturedProgram(
                glyphs=tuple(glyph for glyph in nested.glyphs if glyph.has_paint),
                drawings=tuple(nested.drawings),
                inline_images=tuple(nested.inline_images),
                text_boundaries=tuple(nested.text_boundaries),
                options=nested.options,
            ),
            matrix=None if matrix == IDENTITY_MATRIX else matrix,
        )

    @cites(
        "ISO 32000-2",
        "8.7.3.3",
        introduced="1.2",
        rule="an uncoloured cell is painted in the scn colour, its colour operators ignored",
    )
    def uncolored_cell_state(self, nested: TextState, pattern: model.TilingPattern) -> None:
        """Start an uncoloured tiling pattern's cell in the colour scn gave with the pattern,
        in its underlying space, for fills, strokes, text and image masks alike, and with the
        colour operators ignored there and in every stream the cell invokes (8.6.8), so the
        cell is a stencil that colour is poured through."""
        graphics = nested.graphics
        space = pattern.base_color_spec
        color = pattern.base_color
        if space is not None and color is not None:
            graphics.fill_space = graphics.stroke_space = space
            graphics.fill_color = graphics.stroke_color = color
        graphics.fill_pattern = graphics.stroke_pattern = None
        nested.type3_uncolored = True
        nested.inherited_uncolored = True

    def capture_graphics_soft_mask(self) -> CapturedSoftMask | None:
        mask = self.graphics.soft_mask
        if mask is None or not self.options.render_details:
            return None
        key = (
            id(mask),
            tuple(state_key(getattr(self.graphics, name)) for name in MASK_KEYED_FIELDS),
        )
        caches = self.caches
        cached = caches.capture_soft_masks.get_key(mask, key, MISSING)
        if not isinstance(cached, MissingObject):
            return cached
        group_key = id(mask.group)
        # A Luminosity mask's group composites in its CS (11.6.5.1, Table 142).
        space = blending_space_of(mask.color_space) if mask.subtype == "Luminosity" else None
        if (space is None and mask.subtype != "Alpha") or (
            group_key in caches.capture_active_mask_groups
        ):
            return None
        if len(caches.capture_active_mask_groups) >= 10:
            return None
        caches.capture_soft_masks.put_key(mask, key, None)
        graphics = copy(self.graphics)
        graphics.ctm = mask.ctm
        graphics.soft_mask = None
        graphics.fill_opacity = graphics.stroke_opacity = 1.0
        graphics.blend_mode = None
        caches.capture_active_mask_groups.add(group_key)
        nested: TextState | None = None
        try:
            nested = self.nested_capture_state()
            nested.graphics = graphics
            if space is not None:
                nested.set_capture_cmyk(space == "DeviceCMYK", blending_profile(mask.color_space))
            scope = caches.capture_mask_resources.get_key(mask, id(mask), MISSING)
            nested.resources = self.resources if isinstance(scope, MissingObject) else scope
            frame = nested.append_form_xobject(mask.group, 0)
            if frame is None:
                return None
            nested.stream_executor.consume_frame(frame)
            nested.run_accumulator.flush()
            if space is None:
                if not nested.text_boundaries:
                    return None
                captured = CapturedSoftMask(nested.captured_program(), mask.transfer)
            else:
                captured = CapturedSoftMask(
                    nested.captured_program(),
                    mask.transfer,
                    blending_space=space,
                    blending_profile=blending_profile(mask.color_space),
                    backdrop=self.soft_mask_backdrop(mask, space),
                    colorimetric=colorimetric_luminosity(mask.color_space),
                )
            return caches.capture_soft_masks.put_key(mask, key, captured)
        except PdfParseError, TypeError, ValueError, ArithmeticError:
            return None
        finally:
            caches.capture_active_mask_groups.remove(group_key)
            if nested is not None:
                nested.release()

    def soft_mask_backdrop(self, mask: SoftMask, space: BlendingSpace) -> tuple[float, ...]:
        """A Luminosity mask's BC colour as capture_color gives paint in its group: a device
        space's components, else the CMYK (in a CMYK group) or sRGB it converts to."""
        spec = mask.color_space
        color = tuple(clamp01(value) for value in mask.backdrop_color or ())
        if spec is None:
            return color
        converted: tuple[float, ...] | None = None
        if space == "DeviceCMYK" and spec.kind not in GRAY_RGB_KINDS:
            converted = color_operands_to_cmyk(
                spec,
                color,
                blending_profile(mask.color_space),
                rendering=self.graphics.color_rendering,
            )
        if converted is None:
            converted = color_operands_to_srgb(
                spec, list(color), rendering=self.graphics.color_rendering
            )
        return color if converted is None else tuple(converted)

    # Showing text.
    update_pending_run = text_painting.update_pending_run
    glyph_paint = text_painting.glyph_paint
    emit_actual_text_span = text_painting.emit_actual_text_span
    shared_paint = text_painting.shared_paint
    new_text_layout = text_painting.new_text_layout
    current_text_layout = text_painting.current_text_layout
    new_layout_style = text_painting.new_layout_style
    show_glyph_run = text_painting.show_glyph_run
    show_text = text_painting.show_text
    text_capture_context = text_painting.TEXT_CAPTURE_CONTEXT
    text_boundary = text_painting.text_boundary
    current_capture_actual_text_span = text_painting.current_capture_actual_text_span
    text_paint_mode = text_painting.text_paint_mode

    # Painting and clipping paths, and the line width and dash they are stroked with.
    paint_path = path_painting.paint_path
    clip_path = path_painting.clip_path
    emit_clip_scope_push = path_painting.emit_clip_scope_push
    is_clipped_away = path_painting.is_clipped_away
    graphics_scale = path_painting.graphics_scale
    transformed_line_width = path_painting.transformed_line_width
    transformed_dash_pattern = path_painting.transformed_dash_pattern

    # Painting images, inline images and shadings.
    captured_image_source = image_painting.captured_image_source
    paint_image = image_painting.paint_image
    paint_inline_image = image_painting.paint_inline_image
    paint_shading = image_painting.paint_shading

    # The colors and patterns paint takes.
    initial_pattern = paint_sources.initial_pattern
    capture_color = paint_sources.capture_color
    capture_shading_dictionary = paint_sources.capture_shading_dictionary
    capture_pattern = paint_sources.capture_pattern


__all__ = (
    "CaptureStreamExecutor",
    "DEFAULT_OPERATOR_NAMES",
    "NATIVE_PATH_HANDLERS",
    "NATIVE_STATE_HANDLERS",
    "NATIVE_STATE_METHODS",
    "TextState",
    "applies_paths_natively",
    "cached_native_operators",
    "native_state_operators",
)
