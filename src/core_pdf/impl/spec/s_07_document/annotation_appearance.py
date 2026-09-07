# SPDX-License-Identifier: AGPL-3.0-only
"""Annotation appearance placement, ISO 32000-2, 12.5.5."""

from __future__ import annotations

from core_pdf.impl._impl.model.geometry import transform_bbox
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfValueResolver
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix

ANNOTATION_FLAG_HIDDEN = 1 << 1
ANNOTATION_FLAG_NO_VIEW = 1 << 5


def internal_appearance_matrix(
    rect: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
    matrix: Matrix,
) -> Matrix:
    """Build the matrix mapping an appearance's BBox onto the annotation Rect.

    This is the algorithm in 12.5.5: transform the box by the appearance's
    /Matrix, then scale and shift that result to cover /Rect.
    """
    tx0, ty0, tx1, ty1 = transform_bbox(bbox, matrix)
    width = tx1 - tx0
    height = ty1 - ty0
    sx = (rect[2] - rect[0]) / width if width else 0.0
    sy = (rect[3] - rect[1]) / height if height else 0.0
    return Matrix(sx, 0.0, 0.0, sy, rect[0] - tx0 * sx, rect[1] - ty0 * sy)


def normal_appearance_stream(
    resolver: PdfValueResolver, appearance: object, appearance_state: object
) -> PdfStream | None:
    """Resolve the normal appearance named by AS (ISO 32000-2, 12.5.5)."""
    appearances = resolver.resolve(appearance)
    if appearances is None:
        return None
    if not isinstance(appearances, dict):
        raise ValueError("invalid annotation appearance dictionary")
    normal = resolver.resolve(appearances.get("N"))
    if isinstance(normal, PdfStream):
        return normal
    if not isinstance(normal, dict):
        raise ValueError("invalid normal appearance")
    state = resolver.resolve_name(appearance_state)
    if state is None:
        raise ValueError("appearance state is required for a state dictionary")
    selected = resolver.resolve(normal.get(state))
    if selected is None:
        return None
    if not isinstance(selected, PdfStream):
        raise ValueError("invalid appearance state stream")
    return selected
