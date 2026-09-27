# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from array import array
from math import hypot

import numpy

from core_pdf.impl.capture_host import CaptureHost
from core_pdf.impl.capture_records import (
    EMPTY_LINES,
    CapturedDrawing,
    CapturedLines,
    CapturedPath,
    PaintedDrawingKind,
)
from core_pdf.impl.geometry import intersect_bbox
from core_pdf_cythonized import flatten_path_commands
from core_pdf_spec.s_07_content.model import PdfPath
from core_pdf_spec.s_08_graphics.color import color_space_paints
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


def flatten_path(
    source: PdfPath,
    matrix: Matrix | None,
    lines: StrokeLineRows | None = None,
    line_width: float = 0.0,
) -> CapturedPath:
    xs, ys, spans, bbox, has_segments = flatten_path_commands(
        source.ops,
        source.coords,
        matrix,
        hypot,
        None if lines is None else lines.rows,
        line_width,
    )
    return CapturedPath.deferred_flattened(xs, ys, spans, bbox, has_segments)


class StrokeLineRows:
    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: array[float] = array("d")

    @property
    def count(self) -> int:
        return len(self.rows) // 5

    def since(self, mark: int) -> CapturedLines:
        if mark >= self.count:
            return EMPTY_LINES
        table = numpy.frombuffer(self.rows, dtype=numpy.float64)
        return CapturedLines.from_array(table[mark * 5 :].reshape(-1, 5).copy())


class PathCaptureMixin(CaptureHost):
    def paint_path(self, state: object, source: PdfPath, kind: str, fill_rule: str) -> None:
        if not source.ops:
            return
        if not self.is_graphics_visible():
            return
        graphics = self.graphics
        fills = kind != "stroke" and not self.initial_pattern(stroke=False)
        strokes = kind != "fill" and not self.initial_pattern(stroke=True)
        if not fills and not strokes:
            return
        painted: PaintedDrawingKind = (
            "fillstroke" if fills and strokes else "fill" if fills else "stroke"
        )
        fill_paints = color_space_paints(graphics.fill_space)
        stroke_paints = color_space_paints(graphics.stroke_space)

        ctm = graphics.ctm
        line_width = self.transformed_line_width()
        path = flatten_path(source, None if ctm == IDENTITY_MATRIX else ctm, self.lines, line_width)
        if not path.has_segments():
            return
        self.drawings.append(
            CapturedDrawing(
                seqno=self.sequence,
                fill=self.capture_color(stroke=False),
                fill_pattern=self.capture_pattern(graphics.fill_pattern)
                if fill_paints and fills
                else None,
                fill_opacity=graphics.fill_opacity,
                stroke_color=self.capture_color(stroke=True),
                stroke_pattern=self.capture_pattern(graphics.stroke_pattern)
                if stroke_paints and strokes
                else None,
                stroke_opacity=graphics.stroke_opacity,
                line_width=line_width,
                line_cap=graphics.line_cap,
                line_join=graphics.line_join,
                dash_pattern=self.transformed_dash_pattern() if graphics.dash_pattern else None,
                fill_rule=fill_rule,
                blend_mode=graphics.blend_mode,
                soft_mask_alpha=self.group_alpha,
                alpha_is_shape=graphics.alpha_is_shape,
                kind=painted,
                graphics_soft_mask=self.capture_graphics_soft_mask()
                if graphics.soft_mask is not None
                else None,
                fill_paints=fill_paints,
                stroke_paints=stroke_paints,
                path=path,
                stream_order=self.stream_order,
                xobject_depth=self.xobject_depth,
            )
        )
        self.sequence += 1

    def clip_path(self, state: object, source: PdfPath, fill_rule: str) -> None:
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


__all__ = (
    "PathCaptureMixin",
    "StrokeLineRows",
    "flatten_path",
)
