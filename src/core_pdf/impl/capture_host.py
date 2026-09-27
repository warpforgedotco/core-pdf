# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

CaptureMarks: TypeAlias = tuple[int, int, int, int, int, int]
NO_MARKS: CaptureMarks = (0, 0, 0, 0, 0, 0)

if TYPE_CHECKING:
    from core_pdf.impl.caches import IdentityCache
    from core_pdf.impl.capture_glyphs import GlyphPaint
    from core_pdf.impl.capture_path_sink import StrokeLineRows
    from core_pdf.impl.capture_program import CapturedProgram, CaptureOptions
    from core_pdf.impl.capture_recording import CaptureStreamExecutor
    from core_pdf.impl.capture_records import (
        CapturedDrawing,
        CapturedInlineImage,
        CapturedSoftMask,
        CapturedTextBoundary,
        LayoutFormId,
        PatternPaint,
    )
    from core_pdf.impl.capture_scopes import CaptureGraphicsSave
    from core_pdf.impl.capture_text_runs import RunAccumulator
    from core_pdf.impl.capture_text_sink import MarkedContentEntry, TextLayout
    from core_pdf.impl.capture_tolerant_state import RecoveringTextState
    from core_pdf.impl.document_contracts import CaptureDocument
    from core_pdf.impl.glyphs import GlyphObservation
    from core_pdf.impl.recovery_resolver import ObjectResolver
    from core_pdf.impl.runs import TextRun
    from core_pdf.impl.types import Rectangle
    from core_pdf_spec.s_08_graphics.matrix import Matrix

    class CaptureHost(RecoveringTextState):
        document: CaptureDocument
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
        scale_cache: tuple[Matrix, float] | None
        shared_glyph_paint: GlyphPaint | None
        text_layout: TextLayout | None
        stream_executor: CaptureStreamExecutor

        def captured_program(self, since: CaptureMarks = NO_MARKS) -> CapturedProgram: ...

        def release(self) -> None: ...

        def is_graphics_visible(self) -> bool: ...

        def nested_capture_state(self) -> CaptureHost: ...

        def is_clipped_away(self, x0: float, y0: float, x1: float, y1: float) -> bool: ...

        def emit_clip_scope_push(self) -> None: ...

        def initial_pattern(self, *, stroke: bool) -> bool: ...

        def text_paint_mode(self, *, check_colorants: bool = True) -> int: ...

        def capture_color(self, *, stroke: bool) -> tuple[float, ...] | None: ...

        def capture_pattern(self, pattern: object) -> PatternPaint | None: ...

        def capture_shading_dictionary(self, dictionary: dict) -> dict: ...

        def capture_graphics_soft_mask(self) -> CapturedSoftMask | None: ...

        def transformed_line_width(self) -> float: ...

        def transformed_dash_pattern(self) -> tuple[list[float], float] | None: ...

else:
    CaptureHost = object


__all__ = ("NO_MARKS", "CaptureHost", "CaptureMarks")
