# SPDX-License-Identifier: AGPL-3.0-only
"""Compose PDF execution with the application's capture and font policies."""

from typing import Any

from core_pdf.impl._impl.capture.recording import RecordingMethods
from core_pdf.impl._impl.capture.recovery import CaptureRecovery
from core_pdf.impl._impl.capture.stream_execution import CaptureStreamExecutor
from core_pdf.impl._impl.capture.text_runs import RunAccumulator
from core_pdf.impl._impl.document.recovery.lexer import PdfLexer
from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.fonts.ligatures import detect_ligature_overrides
from core_pdf.impl._impl.runtime.scalars import parse_float_strict, parse_int_strict
from core_pdf.impl.types import Rectangle


class TextState(RecordingMethods):
    def __init__(
        self,
        document: Any,
        hidden_layers: frozenset[str] = frozenset(),
        page_clip: Rectangle | None = None,
        *,
        capture_ink_bounds: bool = True,
        capture_text_runs: bool = True,
    ):
        self.document = document
        self.runs = []
        self.glyphs = []
        self.glyph_cluster_count = 0
        self.lines = []
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.capture_ink_bounds = capture_ink_bounds
        self.capture_text_runs = capture_text_runs
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
        self.capture_patterns = {}
        self.capture_image_sources = {}
        self.capture_colors = {}
        self.capture_soft_masks = {}
        self.capture_mask_resources = {}
        self.capture_active_mask_groups = set()
        self.capture_font_decoders = {}
        self.capture_font_companions = {}

        def font_provider(font: dict[str, Any], resources: dict[str, Any]) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=getattr(document, "raster_font_provider", None),
                semantic_context=getattr(
                    document,
                    "internal_font_semantic_context",
                    getattr(document.resolver, "semantic_context", None),
                ),
            )

        super().__init__(
            document.resolver,
            sink=self,
            font_provider=font_provider,
            lexer_factory=PdfLexer,
            semantic_context=getattr(document.resolver, "semantic_context", None),
        )

        self.recovery = CaptureRecovery()
        self.stream_executor = CaptureStreamExecutor(self)

        # Preserve the reader's historical unset-font and diagnostic color defaults.
        self.graphics.font_size = 12.0
        self.graphics.fill_color = (0.0, 0.0, 0.0)
        self.graphics.stroke_color = (0.0, 0.0, 0.0)

    @staticmethod
    def as_float(value: Any) -> float:
        value_type = type(value)
        if value_type is float:
            return value
        if value_type is int:
            return float(value)
        return parse_float_strict(value, "invalid numeric operand")

    @staticmethod
    def as_int(value: Any) -> int:
        if type(value) is int:
            return value
        return parse_int_strict(value, "invalid numeric operand")
