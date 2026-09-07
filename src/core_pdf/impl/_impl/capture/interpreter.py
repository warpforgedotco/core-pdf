# SPDX-License-Identifier: AGPL-3.0-only
"""Compose PDF execution with the application's capture and font policies."""

from typing import Any

from core_pdf.impl._impl.capture.recording import RecordingMethods
from core_pdf.impl._impl.capture.recovery import CaptureRecovery
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
    ):
        self.runs = []
        self.glyphs = []
        self.glyph_clusters = []
        self.lines = []
        self.drawings = []
        self.inline_images = []
        self.hidden_layers = hidden_layers
        self.page_clip = page_clip
        self.clip_bbox = None
        self.layout_form_bbox = None
        self.layout_form_id = None
        self.capture_source = "native_text"
        self.stream_order = -1
        self.sequence = 0
        self.text_object_id = 0
        self.text_matrix_id = 0
        self.pending_line_break = False
        self.group_alpha = None
        self.run_accumulator = RunAccumulator(self.runs)
        self.capture_graphics_stack = []
        self.capture_marked_entries = {}
        self.capture_frames = {}
        self.capture_patterns = {}

        def font_provider(font: dict[str, Any], resources: dict[str, Any]) -> FontDecoder:
            return FontDecoder(
                font,
                ligature_overrides=detect_ligature_overrides(document, resources, font),
                raster_font_provider=getattr(document, "raster_font_provider", None),
            )

        super().__init__(
            document,
            sink=self,
            font_provider=font_provider,
            recovery=CaptureRecovery(),
            lexer_factory=PdfLexer,
        )

        # Preserve the reader's historical unset-font and diagnostic color defaults.
        self.font_size = 12.0
        self.fill_color = (0.0, 0.0, 0.0)
        self.update_text_scales()

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
