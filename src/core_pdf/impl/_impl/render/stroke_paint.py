# SPDX-License-Identifier: AGPL-3.0-only
"""Paint one stroke's combined coverage without blending its pieces together."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from core_pdf.impl._impl.capture.records import CapturedPath
from core_pdf.impl._impl.render.blend import internal_blend_visible_pixels

if TYPE_CHECKING:
    from core_pdf.impl._impl.render.target_state import internal_RasterState


def internal_paint_stroke_once(
    target: internal_RasterState,
    path: CapturedPath,
    line_width: float,
    rgba: tuple[int, int, int, int],
    dash_pattern: tuple[list[float], float] | None,
    blend_mode: str | None,
    line_cap: int,
    line_join: int,
) -> None:
    """Keep a shape-tracked stroke elementary, ISO 32000-1/2 11.6.2.

    The stroke rasterizer expands segments, caps and joins into overlapping
    pieces. First combine their opaque coverage using the existing raster
    approximation, then apply the stroke's opacity and blend mode once. In
    particular, a round cap must not double the stroke opacity where it meets
    the body. Source shape stays independent of that opacity unless AIS is set.
    """
    pixels = target.pixels
    source_alpha = target.group_source_alpha
    source_shape = target.group_source_shape
    coverage_buffer = bytearray(len(pixels))
    # Geometry painting never enters another group. Suspend only the buffer
    # and recording planes it uses; the group stack and clip remain intact.
    target.pixels = coverage_buffer
    target.group_source_alpha = None
    target.group_source_shape = None
    try:
        target.stroke_path(
            path, line_width, (0, 0, 0, 255), dash_pattern, None, line_cap, line_join
        )
    finally:
        target.pixels = pixels
        target.group_source_alpha = source_alpha
        target.group_source_shape = source_shape
    coverage = target.pixel_view(coverage_buffer)[..., 3]
    covered_rows = numpy.flatnonzero(coverage.any(axis=1))
    if covered_rows.size == 0:
        return
    covered_columns = numpy.flatnonzero(coverage.any(axis=0))
    rows = slice(int(covered_rows[0]), int(covered_rows[-1]) + 1)
    columns = slice(int(covered_columns[0]), int(covered_columns[-1]) + 1)
    coverage = coverage[rows, columns]
    alpha = numpy.rint(coverage.astype(numpy.float64) * (rgba[3] / 255.0)).astype(numpy.uint8)
    target.record_source_coverage(rows, columns, alpha, shape=coverage)
    visible = alpha > 0
    if not numpy.any(visible):
        return
    internal_blend_visible_pixels(
        target.pixel_view(pixels)[rows, columns],
        visible,
        rgba[0] / 255.0,
        rgba[1] / 255.0,
        rgba[2] / 255.0,
        alpha[visible].astype(numpy.float64) / 255.0,
        target.internal_resolved_blend(blend_mode),
        semantic_context=target.semantic_context,
    )
