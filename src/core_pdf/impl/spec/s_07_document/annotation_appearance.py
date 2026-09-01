# SPDX-License-Identifier: AGPL-3.0-only
"""Run annotation appearance streams as part of a page's content.

An annotation's marks live in its appearance stream rather than in the page's
content stream, so a page interpreted without them loses everything the reader
actually sees in a form: the filled-in values, the stamp text, the signature.
12.5.5 defines the transform that places one on the page, and this module
applies it so those streams reach the interpreter the same way a form XObject
does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_content.geometry import transform_bbox
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix

if TYPE_CHECKING:
    from core_pdf.impl.spec.s_07_content.state import TextState

# 12.5.3, Table 165.
ANNOTATION_FLAG_HIDDEN = 1 << 1
ANNOTATION_FLAG_NO_VIEW = 1 << 5

# Popups are only drawn while their parent is open, and the parent already
# carries the same text in /Contents.
SKIPPED_SUBTYPES = frozenset({"Popup", "Link"})


def internal_inheritable(document: Any, node: object, key: str) -> object:
    """Look a key up through an annotation's /Parent chain."""
    for _ in range(50):
        if not isinstance(node, dict):
            return None
        value = node.get(key)
        if value is not None:
            return value
        parent = document.resolver.resolve(node.get("Parent"))
        if parent is node:
            return None
        node = parent
    return None


def select_appearance_stream(
    resolver: Any, appearance: object, appearance_state: object
) -> PdfStream | None:
    """Pick the normal appearance to draw, per 12.5.5.

    ``/AS`` names the substate to use. When it names one ``/N`` does not
    contain, nothing is drawn: a reader must not substitute some other state,
    because doing so renders an unchecked box as checked. Without ``/AS`` the
    choice is unambiguous only when ``/N`` holds exactly one substate.
    """
    appearances = resolver.resolve(appearance)
    if not isinstance(appearances, dict):
        return None
    normal = resolver.resolve(appearances.get("N"))
    if isinstance(normal, PdfStream):
        return normal
    if not isinstance(normal, dict):
        return None
    state_name = resolver.resolve_name(appearance_state)
    if state_name is not None:
        selected = resolver.resolve(normal.get(state_name))
        return selected if isinstance(selected, PdfStream) else None
    if len(normal) == 1:
        only = resolver.resolve(next(iter(normal.values())))
        if isinstance(only, PdfStream):
            return only
    return None


def internal_should_render(document: Any, annot: dict) -> bool:
    subtype = document.resolver.resolve_name(annot.get("Subtype")) or ""
    if subtype in SKIPPED_SUBTYPES:
        return False
    flags = document.resolver.resolve_int(annot.get("F")) or 0
    if flags & (ANNOTATION_FLAG_HIDDEN | ANNOTATION_FLAG_NO_VIEW):
        return False
    if subtype == "Widget":
        # A widget without a field type or name is not a control the reader
        # would draw, and treating it as one resurrects scratch objects.
        if internal_inheritable(document, annot, "FT") is None:
            return False
        if internal_inheritable(document, annot, "T") is None:
            return False
    return True


def internal_appearance_matrix(
    rect: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float] | None,
    matrix: Matrix,
) -> Matrix:
    """Build the matrix mapping an appearance's BBox onto the annotation Rect.

    This is the algorithm in 12.5.5: transform the box by the appearance's
    /Matrix, then scale and shift that result to cover /Rect.
    """
    if bbox is None:
        return IDENTITY_MATRIX
    tx0, ty0, tx1, ty1 = transform_bbox(bbox, matrix)
    width = tx1 - tx0
    height = ty1 - ty0
    sx = (rect[2] - rect[0]) / width if width else 0.0
    sy = (rect[3] - rect[1]) / height if height else 0.0
    return Matrix(sx, 0.0, 0.0, sy, rect[0] - tx0 * sx, rect[1] - ty0 * sy)


def internal_normalized_rect(
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(rect[0], rect[2]),
        min(rect[1], rect[3]),
        max(rect[0], rect[2]),
        max(rect[1], rect[3]),
    )


def consume_annotation_appearances(page: Any, state: "TextState") -> None:
    """Interpret every visible annotation appearance on ``page``."""
    document = page.document
    try:
        annotations = page.annotation_dicts()
    except (PdfParseError, ValueError):
        return

    for annot in annotations:
        try:
            if not internal_should_render(document, annot):
                continue
            stream = select_appearance_stream(document.resolver, annot.get("AP"), annot.get("AS"))
            if stream is None:
                continue
            rect = document.resolver.resolve_box(annot.get("Rect"))
            if rect is None:
                continue
            rect = internal_normalized_rect(rect)

            raw_matrix = stream.dictionary.get("Matrix")
            if isinstance(raw_matrix, (list, tuple)) and len(raw_matrix) > 6:
                raw_matrix = raw_matrix[:6]
            matrix = Matrix.from_operand(raw_matrix) if raw_matrix is not None else IDENTITY_MATRIX
            bbox = document.resolver.resolve_box(stream.dictionary.get("BBox"))

            placement = internal_appearance_matrix(rect, bbox, matrix)
            nested_ctm = matrix.multiply(placement)
            clip = transform_bbox(bbox, nested_ctm) if bbox is not None else rect

            raw_resources = stream.dictionary.get("Resources")
            resolved_resources = (
                raw_resources
                if isinstance(raw_resources, dict)
                else document.resolver.resolve_dict(raw_resources)
            )
            resources = cast(
                PdfDict, resolved_resources if resolved_resources else page.cached_resources
            )

            previous_source = state.capture_source
            state.capture_source = "annotation_appearance"
            try:
                state.consume_stream(
                    document.resolver.resolve_stream(stream),
                    resources,
                    nested_ctm,
                    1,
                    clip_bbox=clip,
                )
            finally:
                state.capture_source = previous_source
        except (PdfParseError, ValueError):
            continue


__all__ = ("consume_annotation_appearances", "select_appearance_stream")
