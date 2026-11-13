# SPDX-License-Identifier: AGPL-3.0-only

"""TextState methods: painting and clipping paths, and the line width and dash they are
stroked with.

Each function takes the state as self; TextState lists them as its methods.
"""

from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING

from core_pdf.impl.base.geometry import intersect_bbox
from core_pdf.impl.capture.recording_support import (
    DRAWING_BBOX,
    DRAWING_COLOR_RENDERING,
    DRAWING_DICTIONARY,
    DRAWING_GROUP_ISOLATED,
    DRAWING_GROUP_KNOCKOUT,
    DRAWING_IMAGE_CLIP,
    DRAWING_IMAGE_SOURCE,
    DRAWING_LAYOUT,
    DRAWING_PAINTS,
    DRAWING_RAW_DATA,
    SPACES_THAT_MAY_NOT_PAINT,
    flatten_path,
)
from core_pdf.impl.capture.records import (
    EMPTY_DRAWING_ITEMS,
    CapturedDrawing,
    CapturedPath,
    PaintedDrawingKind,
    marker_drawing,
)
from core_pdf.impl.spec.s_07_content.model import PdfPath
from core_pdf.impl.spec.s_08_graphics.color import color_space_paints
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
lazy from core_pdf_cythonized import flatten_path_commands

if TYPE_CHECKING:
    from core_pdf.impl.capture.recording import TextState


def paint_path(self: TextState, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
    if not source.ops:
        return
    if self.marked_content_stack and not self.is_graphics_visible():
        return
    graphics = self.graphics
    fill_space = graphics.fill_space
    stroke_space = graphics.stroke_space
    # initial_pattern, color_space_paints and transformed_line_width, inlined.
    fills = kind != "stroke" and not (
        fill_space.kind == "Pattern" and graphics.fill_pattern is None
    )
    strokes = kind != "fill" and not (
        stroke_space.kind == "Pattern" and graphics.stroke_pattern is None
    )
    if not fills and not strokes:
        return
    painted: PaintedDrawingKind = (
        "fillstroke" if fills and strokes else "fill" if fills else "stroke"
    )
    fill_paints = fill_space.kind not in SPACES_THAT_MAY_NOT_PAINT or color_space_paints(fill_space)
    stroke_paints = stroke_space.kind not in SPACES_THAT_MAY_NOT_PAINT or color_space_paints(
        stroke_space
    )

    ctm = graphics.ctm
    line_width = max(0.0, graphics.line_width)
    if line_width != 0:
        # graphics_scale, with its cache check inlined.
        scale_cache = self.scale_cache
        line_width *= (
            scale_cache[1]
            if scale_cache is not None and scale_cache[0] is ctm
            else self.graphics_scale()
        )
    else:
        line_width = 0.0
    xs, ys, spans, bbox, has_segments = flatten_path_commands(
        source.ops,
        source.coords,
        None if ctm == IDENTITY_MATRIX else ctm,
        hypot,
        self.lines.rows,
        line_width,
    )
    if not has_segments:
        return
    path = CapturedPath.deferred_flattened(xs, ys, spans, bbox, has_segments)
    fill_pattern = graphics.fill_pattern
    stroke_pattern = graphics.stroke_pattern
    # capture_color's memo, checked inline for the common repeated paint.
    intent = graphics.render_intent
    black_point = graphics.black_point_compensation
    fill_color = graphics.fill_color
    last = self.last_fill_color
    if fill_color is None or fill_space is None:
        captured_fill = fill_color
    elif (
        last is not None
        and last[0] is fill_space
        and last[1] is fill_color
        and last[2] == intent
        and last[3] == black_point
    ):
        captured_fill = last[4]
    else:
        captured_fill = self.capture_color(stroke=False)
    stroke_color = graphics.stroke_color
    last = self.last_stroke_color
    if stroke_color is None or stroke_space is None:
        captured_stroke = stroke_color
    elif (
        last is not None
        and last[0] is stroke_space
        and last[1] is stroke_color
        and last[2] == intent
        and last[3] == black_point
    ):
        captured_stroke = last[4]
    else:
        captured_stroke = self.capture_color(stroke=True)
    self.drawings.append(
        # Built by its slots: CapturedDrawing's __init__ only assigns them, and its
        # __post_init__ keeps the default empty items as they are.
        DRAWING_LAYOUT.build(
            (
                self.sequence,
                captured_fill,
                graphics.fill_opacity,
                self.capture_pattern(fill_pattern)
                if fill_paints and fills and fill_pattern is not None
                else None,
                captured_stroke,
                self.capture_pattern(stroke_pattern)
                if stroke_paints and strokes and stroke_pattern is not None
                else None,
                graphics.stroke_opacity,
                line_width,
                graphics.line_cap,
                graphics.line_join,
                self.transformed_dash_pattern(),
                fill_rule,
                graphics.blend_mode,
                self.group_alpha,
                DRAWING_RAW_DATA,
                DRAWING_DICTIONARY,
                DRAWING_IMAGE_SOURCE,
                DRAWING_IMAGE_CLIP,
                painted,
                EMPTY_DRAWING_ITEMS,
                path,
                DRAWING_BBOX,
                self.stream_order,
                self.xobject_depth,
                DRAWING_COLOR_RENDERING,
                DRAWING_PAINTS,
                fill_paints,
                stroke_paints,
                DRAWING_GROUP_ISOLATED,
                DRAWING_GROUP_KNOCKOUT,
                graphics.alpha_is_shape,
                self.capture_graphics_soft_mask() if graphics.soft_mask is not None else None,
            )
        )
    )
    self.sequence += 1


def clip_path(self: TextState, state: object, source: PdfPath, fill_rule: str) -> None:
    path = flatten_path(source, self.graphics.ctm)
    if not path.has_segments():
        return
    clip_bbox = path.bbox()
    if clip_bbox is not None:
        self.clip_bbox = intersect_bbox(self.clip_bbox, clip_bbox)
    if self.is_graphics_visible():
        self.emit_clip_scope_push()
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=None,
                fill_opacity=None,
                blend_mode=self.graphics.blend_mode,
                soft_mask_alpha=self.group_alpha,
                alpha_is_shape=self.graphics.alpha_is_shape,
                line_width=0.0,
                line_cap=self.graphics.line_cap,
                line_join=self.graphics.line_join,
                dash_pattern=self.transformed_dash_pattern(),
                fill_rule=fill_rule,
                kind="clip",
                path=path,
            )
        )
        self.sequence += 1


def emit_clip_scope_push(self: TextState) -> None:
    if not self.capture_graphics_stack or self.capture_graphics_stack[-1].clip_scope_emitted:
        return
    self.capture_graphics_stack[-1].clip_scope_emitted = True
    self.drawings.append(marker_drawing("state-push", self.sequence))
    self.sequence += 1


def is_clipped_away(self: TextState, x0: float, y0: float, x1: float, y1: float) -> bool:
    for clip in (self.clip_bbox, self.page_clip):
        if clip is None:
            continue
        if x1 <= clip[0] or x0 >= clip[2] or y1 <= clip[1] or y0 >= clip[3]:
            return True
    return False


def graphics_scale(self: TextState) -> float:
    ctm = self.graphics.ctm
    cached = self.scale_cache
    if cached is not None and cached[0] is ctm:
        return cached[1]
    x_scale = hypot(ctm.a, ctm.b)
    y_scale = hypot(ctm.c, ctm.d)
    if x_scale == 0 and y_scale == 0:
        scale = 1.0
    elif x_scale == 0:
        scale = y_scale
    elif y_scale == 0:
        scale = x_scale
    else:
        scale = (x_scale + y_scale) * 0.5
    self.scale_cache = (ctm, scale)
    return scale


def transformed_line_width(self: TextState) -> float:
    line_width = max(0.0, self.graphics.line_width)
    if line_width == 0:
        return 0.0
    return line_width * self.graphics_scale()


def transformed_dash_pattern(self: TextState) -> tuple[list[float], float]:
    dash_array, phase = self.graphics.dash_pattern
    if not dash_array and phase == 0:
        # A solid line: a zero phase scales to the same zero.
        return [], float(phase)
    scale = self.graphics_scale()
    return [max(0.0, float(value) * scale) for value in dash_array], float(phase) * scale
