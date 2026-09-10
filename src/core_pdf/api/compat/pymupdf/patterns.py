"""Placement of captured pattern cells for structured-text image extraction."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from core_pdf.api.compat._shared import float32
from core_pdf.api.compat.pymupdf.geometry import Matrix, Rect
from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl._impl.capture.records import TilingPattern
from core_pdf.impl._impl.model.geometry import intersect_bbox
from core_pdf.impl.types import Matrix6, Rectangle


@dataclass(frozen=True, slots=True)
class PatternCell:
    order: tuple[int, ...]
    program: CapturedProgram
    pattern_id: int
    transform: Matrix6
    anchor_transform: Matrix6
    clip: Rectangle
    initial_matrix: tuple[Matrix6, Matrix6]


def pattern_cells(
    program: CapturedProgram | PageProgram,
    page_matrix: Matrix,
    active: frozenset[int],
    transform: Matrix,
    anchor_transform: Matrix,
) -> Iterator[PatternCell]:
    """Yield direct cells, or one tile-cache cell for a repeating pattern."""
    for drawing in program.drawings:
        for paint, pattern in enumerate((drawing.fill_pattern, drawing.stroke_pattern)):
            if drawing.kind not in (
                ("fill", "fillstroke") if paint == 0 else ("stroke", "fillstroke")
            ):
                continue
            if not isinstance(pattern, TilingPattern) or id(pattern) in active:
                continue
            if drawing.rect is None:
                continue
            anchor = Matrix(drawing.stream_matrix) * anchor_transform
            base = Matrix(pattern.matrix) * anchor
            if abs(base.a * base.d - base.b * base.c) < 1e-12:
                continue
            area = Rect(drawing.rect) * transform * page_matrix
            local = area * ~(base * page_matrix)
            if local.is_empty:
                continue
            x0, x1 = sorted(
                float32(float32(value - pattern.bbox[0]) / float32(pattern.x_step))
                for value in (local.x0, local.x1)
            )
            y0, y1 = sorted(
                float32(float32(value - pattern.bbox[1]) / float32(pattern.y_step))
                for value in (local.y0, local.y1)
            )
            consumer: Rectangle | None = cast(
                Rectangle, tuple(Rect(drawing.soft_mask_clip or drawing.rect) * transform)
            )
            # A cached tile is interpreted once. Its device clip also includes
            # the pattern-space area passed to the tile container.
            cached = (x1 - x0 > 1 or y1 - y0 > 1) and not any(
                item.soft_mask is not None
                or item.blend_mode not in (None, "Normal")
                or item.fill_opacity not in (None, 1)
                or item.stroke_opacity not in (None, 1)
                for item in pattern.drawings
            )
            if cached:
                consumer = intersect_bbox(consumer, cast(Rectangle, tuple(local * ~page_matrix)))
                columns, rows = range(1), range(1)
            else:
                left, bottom = math.floor(float32(x0 + 0.001)), math.floor(float32(y0 + 0.001))
                right, top = math.ceil(float32(x1 - 0.001)), math.ceil(float32(y1 - 0.001))
                columns = range(left, max(right, left + 1))
                rows = range(bottom, max(top, bottom + 1))
            if consumer is None:
                continue
            cell_program = CapturedProgram(
                drawings=tuple(pattern.drawings),
                glyphs=tuple(pattern.glyphs),
                inline_images=tuple(pattern.inline_images),
            )
            for row in rows:
                for column in columns:
                    moved = Matrix(1, 0, 0, 1, column * pattern.x_step, row * pattern.y_step) * base
                    yield PatternCell(
                        (drawing.seqno, 1, paint, row - rows.start, column - columns.start),
                        cell_program,
                        id(pattern),
                        cast(Matrix6, tuple(~Matrix(pattern.matrix) * moved)),
                        cast(Matrix6, tuple(anchor)),
                        consumer,
                        (pattern.matrix, cast(Matrix6, tuple(moved))),
                    )
