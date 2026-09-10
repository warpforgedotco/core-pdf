"""Compose page, mask and pattern image sources in paint order."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from core_pdf.api.compat.pymupdf.geometry import Matrix, Rect
from core_pdf.api.compat.pymupdf.masks import capture_soft_masks
from core_pdf.api.compat.pymupdf.patterns import pattern_cells
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl._impl.model.geometry import intersect_bbox
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.types import Matrix6, Rectangle


def transformed_box(box: Rectangle | None, matrix: Matrix) -> Rectangle | None:
    if box is None:
        return None
    return cast(Rectangle, tuple(Rect(box) * matrix))


@dataclass(frozen=True, slots=True)
class ImageScope:
    program: CapturedProgram | PageProgram
    order: tuple[int, ...] = ()
    transform: Matrix6 = IDENTITY_MATRIX
    anchor_transform: Matrix6 = IDENTITY_MATRIX
    clip: Rectangle | None = None
    initial_matrix: tuple[Matrix6, Matrix6] | None = None


def image_scopes(
    page: PdfPage,
    scope: ImageScope,
    page_matrix: Matrix,
    active: frozenset[int] = frozenset(),
    *,
    include_masks: bool = True,
) -> Iterator[ImageScope]:
    """Share one traversal across nested paint sources without decoding images."""
    yield scope
    if include_masks:
        # capture_soft_masks already walks its mask descendants. Those scopes
        # can contain patterns, but must not walk the same mask edges twice.
        for order, program in capture_soft_masks(page, scope.program):
            yield from image_scopes(
                page,
                ImageScope(
                    program,
                    scope.order + order,
                    scope.transform,
                    scope.anchor_transform,
                    scope.clip,
                ),
                page_matrix,
                active,
                include_masks=False,
            )
    for cell in pattern_cells(
        scope.program, page_matrix, active, Matrix(scope.transform), Matrix(scope.anchor_transform)
    ):
        yield from image_scopes(
            page,
            ImageScope(
                cell.program,
                scope.order + cell.order,
                cell.transform,
                cell.anchor_transform,
                intersect_bbox(scope.clip, cell.clip),
                cell.initial_matrix,
            ),
            page_matrix,
            active | {cell.pattern_id},
        )
