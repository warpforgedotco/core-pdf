# SPDX-License-Identifier: AGPL-3.0-only
"""State records used while executing nested PDF content streams."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from core_pdf.impl.spec.s_07_content.capture import PatternPaint
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
    from core_pdf.impl.spec.s_09_fonts.decoder import FontDecoder

StreamKey = tuple[str, int, int]
# An entry records one Form XObject invocation. Either half can be absent --
# a direct (non-reference) XObject has no stream key, and a form without a
# usable /BBox has no layout box -- but the entry still has to be kept so
# the tuple identifies the invocation tree. Consumers skip unusable halves.
LayoutFormId: TypeAlias = (
    tuple[tuple[StreamKey | None, tuple[float, float, float, float] | None], ...] | None
)


@dataclass(frozen=True, slots=True)
class StreamState:
    """One saved content-stream execution frame."""

    resources: PdfDict
    resources_id: int
    ctm: Matrix
    text_matrix: Matrix
    line_matrix: Matrix
    font_size: float
    font_operand: object
    font_size_operand: object
    horizontal_scale: float
    char_space: float
    word_space: float
    rise: float
    leading: float
    render_mode: int
    current_font: str | None
    current_decoder: FontDecoder | None
    current_decoder_resources_id: int | None
    graphics_stack_len: int
    marked_content_stack_len: int
    fill_color: tuple[float, ...] | None
    fill_pattern: PatternPaint | None
    fill_opacity: float
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PatternPaint | None
    stroke_opacity: float
    line_width: float
    line_cap: int
    line_join: int
    miter_limit: float
    dash_pattern: tuple[list[float], float]
    fill_color_space: str
    stroke_color_space: str
    compatibility_depth: int
    blend_mode: str | None
    group_alpha: float | None
    flatness: int
    render_intent: str | None
    clip_bbox: tuple[float, float, float, float] | None
    layout_form_bbox: tuple[float, float, float, float] | None
    layout_form_id: LayoutFormId
    pending_line_break: bool
    xobject_depth: int


@dataclass(slots=True)
class ContentStreamFrame:
    """One pending nested content stream, plus the state captured on entry."""

    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: tuple[float, float, float, float] | None
    group_alpha: float | None = None
    layout_form_bbox: tuple[float, float, float, float] | None = field(default=None, kw_only=True)
    layout_form_id: LayoutFormId = field(default=None, kw_only=True)
    stream_key: StreamKey | None = field(default=None, kw_only=True)
    swallow_parse_errors: bool = field(default=False, kw_only=True)
    lexer: PdfLexer | None = field(default=None, init=False)
    old_state: StreamState | None = field(default=None, init=False)
    outer_group_alpha: float | None = field(default=None, init=False)
    entered: bool = field(default=False, init=False)


# Every field except the two stack depths mirrors a TextState attribute of the
# same name, so capture/restore drive off this list instead of two hand-written
# copies that a new field can silently fall out of.
STREAM_STATE_MIRRORED: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(StreamState)
    if f.name not in ("graphics_stack_len", "marked_content_stack_len")
)
