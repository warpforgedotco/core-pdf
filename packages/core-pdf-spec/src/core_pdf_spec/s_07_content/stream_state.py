# SPDX-License-Identifier: AGPL-3.0-only
"""State records used while executing nested PDF content streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core_pdf_spec.s_07_content.operations import ContentOperationState
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix

if TYPE_CHECKING:
    from core_pdf_spec.s_07_syntax.lexer import PdfLexer

StreamKey = tuple[str, int, int]
# Both q/Q and nested streams restore this graphics state. Text parameters
# belong to it, but the text and line matrices are saved only across streams.
GRAPHICS_STATE_FIELDS: tuple[str, ...] = (
    "ca",
    "cb",
    "cc",
    "cd",
    "ce",
    "cf",
    "fill_color",
    "fill_pattern",
    "fill_opacity",
    "stroke_color",
    "stroke_pattern",
    "stroke_opacity",
    "fill_color_space",
    "stroke_color_space",
    "fill_color_spec",
    "stroke_color_spec",
    "blend_mode",
    "flatness",
    "render_intent",
    "line_width",
    "line_cap",
    "line_join",
    "miter_limit",
    "dash_pattern",
    "font_size",
    "font_operand",
    "font_size_operand",
    "horizontal_scale",
    "char_space",
    "word_space",
    "rise",
    "leading",
    "render_mode",
    "current_font",
    "current_decoder",
    "current_decoder_resources_id",
)


@dataclass(slots=True)
class GraphicsSave:
    """One saved PDF graphics state."""

    graphics_state: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class StreamState:
    """Graphics snapshot and the additional state isolated by a nested stream."""

    graphics_state: tuple[Any, ...]
    resources: PdfDict
    resources_id: int
    text_matrix: Matrix
    line_matrix: Matrix
    graphics_stack_floor: int
    graphics_stack_len: int
    marked_content_stack_len: int
    xobject_depth: int
    operation_state: ContentOperationState = field(
        default_factory=ContentOperationState, kw_only=True
    )
    compatibility_depth: int = field(default=0, kw_only=True)
    # Path construction is outside q/Q, but pending child clips must not leak.
    pending_clip_rule: str | None = field(default=None, kw_only=True)


@dataclass(slots=True)
class ContentStreamFrame:
    """One pending nested content stream, plus the state captured on entry."""

    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: tuple[float, float, float, float] | None
    group_alpha: float | None = None
    form_bbox_operand: object = field(default=None, kw_only=True)
    is_form: bool = field(default=False, kw_only=True)
    source_key: StreamKey | None = field(default=None, kw_only=True)
    stream_key: StreamKey | None = field(default=None, kw_only=True)
    operation_state: ContentOperationState = field(
        default_factory=ContentOperationState, init=False
    )
    lexer: PdfLexer | None = field(default=None, init=False)
    # Present only while this frame is entered, including suspension for a child.
    old_state: StreamState | None = field(default=None, init=False)


# Compatibility metadata for callers inspecting snapshots. Capture and restore
# use explicit typed assignments; graphics fields and stack lengths are separate.
STREAM_STATE_MIRRORED: tuple[str, ...] = (
    "resources",
    "resources_id",
    "text_matrix",
    "line_matrix",
    "graphics_stack_floor",
    "xobject_depth",
    "operation_state",
    "compatibility_depth",
)


__all__ = (
    "StreamKey",
    "GRAPHICS_STATE_FIELDS",
    "GraphicsSave",
    "StreamState",
    "ContentStreamFrame",
    "STREAM_STATE_MIRRORED",
)
