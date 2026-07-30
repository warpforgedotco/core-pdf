# SPDX-License-Identifier: AGPL-3.0-only
"""State records used while executing nested PDF content streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.engine.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfDict

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder

ResourceCategoryCache = dict[str, object]
ResourceCache = dict[tuple[int, str], ResourceCategoryCache]
ResolvedResourceCache = dict[tuple[int, str], object]
StreamKey = tuple[str, int, int]


class StreamState:
    __slots__ = (
        "resources",
        "resources_id",
        "ctm",
        "text_matrix",
        "line_matrix",
        "font_size",
        "font_operand",
        "font_size_operand",
        "current_font",
        "current_decoder",
        "current_decoder_resources_id",
        "graphics_stack_len",
        "marked_content_stack_len",
        "fill_color",
        "fill_pattern",
        "fill_opacity",
        "stroke_color",
        "stroke_pattern",
        "compatibility_depth",
        "blend_mode",
        "group_alpha",
        "fill_color_space",
        "stroke_color_space",
        "flatness",
        "render_intent",
        "clip_bbox",
        "pending_line_break",
        "xobject_depth",
        "inline_images",
        "resource_cache",
        "resolved_resource_categories",
    )

    resources: PdfDict
    resources_id: int
    ctm: Matrix
    text_matrix: Matrix
    line_matrix: Matrix
    font_size: float
    font_operand: object
    font_size_operand: object
    current_font: str | None
    current_decoder: FontDecoder | None
    current_decoder_resources_id: int | None
    graphics_stack_len: int
    marked_content_stack_len: int
    fill_color: tuple[float, ...] | None
    fill_pattern: PdfDict | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: PdfDict | None
    compatibility_depth: int
    blend_mode: str | None
    group_alpha: float | None
    fill_color_space: str
    stroke_color_space: str
    flatness: int
    render_intent: str | None
    pending_line_break: bool
    xobject_depth: int

    def __init__(
        self,
        resources: PdfDict,
        resources_id: int,
        ctm: Matrix,
        text_matrix: Matrix,
        line_matrix: Matrix,
        font_size: float,
        font_operand: object,
        font_size_operand: object,
        current_font: str | None,
        current_decoder: FontDecoder | None,
        current_decoder_resources_id: int | None,
        graphics_stack_len: int,
        marked_content_stack_len: int,
        fill_color: tuple[float, ...] | None,
        fill_pattern: PdfDict | None,
        fill_opacity: float | None,
        stroke_color: tuple[float, ...] | None,
        stroke_pattern: PdfDict | None,
        fill_color_space: str,
        stroke_color_space: str,
        compatibility_depth: int,
        blend_mode: str | None,
        group_alpha: float | None,
        flatness: int,
        render_intent: str | None,
        clip_bbox: tuple[float, float, float, float] | None,
        pending_line_break: bool,
        xobject_depth: int,
        resource_cache: ResourceCache,
        resolved_resource_categories: ResolvedResourceCache,
    ) -> None:
        self.resources = resources
        self.resources_id = resources_id
        self.ctm = ctm
        self.text_matrix = text_matrix
        self.line_matrix = line_matrix
        self.font_size = font_size
        self.font_operand = font_operand
        self.font_size_operand = font_size_operand
        self.current_font = current_font
        self.current_decoder = current_decoder
        self.current_decoder_resources_id = current_decoder_resources_id
        self.graphics_stack_len = graphics_stack_len
        self.marked_content_stack_len = marked_content_stack_len
        self.fill_color = fill_color
        self.fill_pattern = fill_pattern
        self.fill_opacity = fill_opacity
        self.stroke_color = stroke_color
        self.stroke_pattern = stroke_pattern
        self.compatibility_depth = compatibility_depth
        self.fill_color_space = fill_color_space
        self.stroke_color_space = stroke_color_space
        self.blend_mode = blend_mode
        self.group_alpha = group_alpha
        self.flatness = flatness
        self.render_intent = render_intent
        self.clip_bbox = clip_bbox
        self.pending_line_break = pending_line_break
        self.xobject_depth = xobject_depth
        self.resource_cache = resource_cache
        self.resolved_resource_categories = resolved_resource_categories


class ContentStreamFrame:
    __slots__ = (
        "stream",
        "resources",
        "ctm",
        "depth",
        "clip_bbox",
        "group_alpha",
        "swallow_parse_errors",
        "lexer",
        "old_state",
        "stream_key",
        "outer_group_alpha",
        "entered",
    )

    stream: PdfStream
    resources: PdfDict
    ctm: Matrix
    depth: int
    clip_bbox: tuple[float, float, float, float] | None
    group_alpha: float | None
    swallow_parse_errors: bool
    lexer: PdfLexer | None
    old_state: StreamState | None
    stream_key: StreamKey | None
    outer_group_alpha: float | None
    entered: bool

    def __init__(
        self,
        stream: PdfStream,
        resources: PdfDict,
        ctm: Matrix,
        depth: int,
        clip_bbox: tuple[float, float, float, float] | None,
        group_alpha: float | None = None,
        *,
        stream_key: StreamKey | None = None,
        swallow_parse_errors: bool = False,
    ) -> None:
        self.stream = stream
        self.resources = resources
        self.ctm = ctm
        self.depth = depth
        self.clip_bbox = clip_bbox
        self.group_alpha = group_alpha
        self.swallow_parse_errors = swallow_parse_errors
        self.lexer = None
        self.old_state = None
        self.stream_key = stream_key
        self.outer_group_alpha = None
        self.entered = False
