# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any

from core_pdf.impl.caches import IdentityCache
from core_pdf.impl.capture_host import NO_MARKS, CaptureMarks
from core_pdf.impl.capture_image_sink import ImageCaptureMixin
from core_pdf.impl.capture_paints import PaintResolutionMixin
from core_pdf.impl.capture_path_sink import PathCaptureMixin, StrokeLineRows
from core_pdf.impl.capture_program import DEFAULT_CAPTURE, CapturedProgram, CaptureOptions
from core_pdf.impl.capture_recovery import CaptureRecovery, iter_content_operations
from core_pdf.impl.capture_scopes import ScopeCaptureMixin
from core_pdf.impl.capture_text_runs import RunAccumulator
from core_pdf.impl.capture_text_sink import GlyphCaptureMixin
from core_pdf.impl.capture_tolerant_state import CaptureCaches, RecoveringTextState
from core_pdf.impl.document_contracts import CaptureDocument
from core_pdf.impl.fonts_decoder import FontDecoder
from core_pdf.impl.fonts_ligatures import detect_ligature_overrides
from core_pdf.impl.recovery_lexer import PdfLexer
from core_pdf.impl.types import Rectangle
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.operations import OperationHandler
from core_pdf_spec.s_07_content.streams import (
    ContentStreamExecutor,
    ContentStreamFrame,
)
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_07_syntax_primitives.coercion import parse_float_strict, parse_int_strict

GLYPH_PAINT_KEEPING_OPERATORS = frozenset(
    ("BT", "Td", "TD", "Tm", "T*", "Tj", "TJ", "'", '"', "Tc", "Tw", "Tz", "TL", "Ts", "Tf")
)


TEXT_LAYOUT_KEEPING_OPERATORS = frozenset(("Tj", "TJ", "TL", "Tw"))
LINE_MOVING_OPERATORS = frozenset(("Td", "TD", "T*", "'"))


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


class CaptureStreamExecutor(ContentStreamExecutor):
    state: TextState
    _operator_names: frozenset[bytes] | None = None

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
        names = self._operator_names
        if names is None:
            names = self._operator_names = frozenset(name.encode("latin-1") for name in table)
        return names

    def dispatch_frame(self, frame: ContentStreamFrame) -> ContentStreamFrame | None:
        state = self.state
        assert frame.lexer is not None
        handlers = state.operation_table()
        depth = frame.depth
        for name, operands in iter_content_operations(
            frame.lexer,
            recovery=state.content_recovery,
            is_operator=self.operator_names(handlers).__contains__,
            path_state=state if applies_paths_natively(handlers, state) else None,
        ):
            handler = handlers.get(name)
            if handler is None:
                continue
            if name not in GLYPH_PAINT_KEEPING_OPERATORS:
                state.shared_glyph_paint = None
                state.text_layout = None
            elif name in LINE_MOVING_OPERATORS:
                if (layout := state.text_layout) is not None:
                    layout.style = None
            elif name not in TEXT_LAYOUT_KEEPING_OPERATORS:
                state.text_layout = None
            child = handler(operands, depth)
            if child is not None:
                return child
        return None

    def handle_parse_error(self, frame: ContentStreamFrame, error: PdfParseError) -> None:
        if not frame.is_form:
            raise error


class TextState(
    GlyphCaptureMixin,
    PathCaptureMixin,
    ImageCaptureMixin,
    ScopeCaptureMixin,
    PaintResolutionMixin,
    RecoveringTextState,
):
    stream_executor: CaptureStreamExecutor
    stream_executor_type = CaptureStreamExecutor

    def __init__(
        self,
        document: CaptureDocument,
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
        self.run_accumulator = RunAccumulator(self.runs)
        self.capture_graphics_stack = []
        self.capture_marked_entries = {}
        self.capture_frames = {}
        self.capture_patterns = IdentityCache()

        def font_provider(font: PdfDict, resources: PdfDict) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=getattr(document, "raster_font_provider", None),
                semantic_context=getattr(
                    document,
                    "font_semantic_context",
                    getattr(document.resolver, "semantic_context", None),
                ),
            )

        super().__init__(
            document.resolver,
            sink=self,
            font_provider=font_provider,
            lexer_factory=PdfLexer,
            semantic_context=getattr(document.resolver, "semantic_context", None),
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


__all__ = ("CaptureStreamExecutor", "TextState", "applies_paths_natively")
