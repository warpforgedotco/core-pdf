# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from core_pdf.impl.capture.interpreter import TextState
from core_pdf.impl.capture.program import AppearanceProgram, CapturedProgram, PageProgram
from core_pdf.impl.document.records import RawAnnotation, RawFormField
from core_pdf.impl.document.recovery.resources import resolve_resource_dict
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.model.geometry import normalize_rect, transform_bbox
from core_pdf_spec.s_07_document.annotation_appearance import (
    ANNOTATION_FLAG_HIDDEN,
    ANNOTATION_FLAG_NO_VIEW,
    appearance_matrix,
    normal_appearance_stream,
)
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix

SKIPPED_SUBTYPES = frozenset({"Popup", "Link"})


def internal_inheritable(document: Any, node: object, key: str) -> object:
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
    try:
        return normal_appearance_stream(resolver, appearance, appearance_state)
    except ValueError:
        if resolver.resolve_name(appearance_state) is not None:
            return None
        appearances = resolver.resolve(appearance)
        if not isinstance(appearances, dict):
            return None
        normal = resolver.resolve(appearances.get("N"))
        if isinstance(normal, dict) and len(normal) == 1:
            only = resolver.resolve(next(iter(normal.values())))
            return only if isinstance(only, PdfStream) else None
        return None


def internal_should_render(document: Any, annot: dict) -> bool:
    subtype = document.resolver.resolve_name(annot.get("Subtype")) or ""
    if subtype in SKIPPED_SUBTYPES:
        return False
    flags = document.resolver.resolve_int(annot.get("F")) or 0
    if flags & (ANNOTATION_FLAG_HIDDEN | ANNOTATION_FLAG_NO_VIEW):
        return False
    if subtype == "Widget":
        if internal_inheritable(document, annot, "FT") is None:
            return False
        if internal_inheritable(document, annot, "T") is None:
            return False
    return True


def capture_annotation_appearances(
    page: Any,
    state: TextState,
    *,
    fields: Iterable[Any] | None = None,
    annotations: Iterable[Any] | None = None,
) -> tuple[AppearanceProgram, ...]:
    document = page.document
    try:
        candidates = (
            list(page.annotation_dicts())
            if annotations is None
            else [annotation.dict for annotation in annotations]
        )
    except PdfParseError, ValueError:
        candidates = []
    candidates = list({id(annot): annot for annot in candidates}.values())
    if fields is None:
        try:
            fields = page.get_fields()
        except PdfParseError, ValueError:
            fields = ()
    seen = {id(annot) for annot in candidates}
    for field in fields:
        widget = field.widget or field.dict
        if (
            isinstance(widget, dict)
            and document.resolver.resolve_name(widget.get("Subtype")) == "Widget"
            and id(widget) not in seen
        ):
            seen.add(id(widget))
            candidates.append(widget)

    appearances: list[AppearanceProgram] = []
    for annot in candidates:
        try:
            if not internal_should_render(document, annot):
                continue
            stream = select_appearance_stream(document.resolver, annot.get("AP"), annot.get("AS"))
            if stream is None:
                continue
            rect = document.resolver.resolve_box(annot.get("Rect"))
            if rect is None:
                continue
            rect = normalize_rect(rect)

            raw_matrix = document.resolver.deep_resolve(stream.dictionary.get("Matrix"))
            if isinstance(raw_matrix, (list, tuple)) and len(raw_matrix) > 6:
                raw_matrix = raw_matrix[:6]
            matrix = Matrix.from_operand(raw_matrix) if raw_matrix is not None else IDENTITY_MATRIX
            bbox = document.resolver.resolve_box(stream.dictionary.get("BBox"))

            placement = (
                appearance_matrix(rect, bbox, matrix) if bbox is not None else IDENTITY_MATRIX
            )
            nested_ctm = matrix.multiply(placement)
            clip = transform_bbox(bbox, nested_ctm) if bbox is not None else rect

            resolved_resources = resolve_resource_dict(
                stream.dictionary.get("Resources"), document.resolver
            )
            resources = cast(PdfDict, resolved_resources or page.resources)

            previous_source = state.capture_source
            state.run_accumulator.flush()
            run_start = len(state.runs)
            glyph_start = len(state.glyphs)
            drawing_start = len(state.drawings)
            image_start = len(state.inline_images)
            line_start = len(state.lines)
            text_boundary_start = len(state.text_boundaries)
            state.capture_source = "annotation_appearance"
            try:
                state.stream_executor.consume(
                    document.resolver.resolve_stream(stream),
                    resources,
                    nested_ctm,
                    1,
                    clip_bbox=clip,
                )
            finally:
                state.run_accumulator.flush()
                state.capture_source = previous_source
                appearances.append(
                    AppearanceProgram(
                        kind=(
                            "widget"
                            if document.resolver.resolve_name(annot.get("Subtype")) == "Widget"
                            else "annotation"
                        ),
                        source=annot,
                        clip_bbox=clip,
                        program=CapturedProgram(
                            runs=tuple(state.runs[run_start:]),
                            glyphs=tuple(state.glyphs[glyph_start:]),
                            drawings=tuple(state.drawings[drawing_start:]),
                            inline_images=tuple(state.inline_images[image_start:]),
                            lines=tuple(state.lines[line_start:]),
                            text_boundaries=tuple(state.text_boundaries[text_boundary_start:]),
                        ),
                    )
                )
        except PdfParseError, ValueError:
            continue
    return tuple(appearances)


def capture_page_program(
    page: Any,
    *,
    hidden_layers: frozenset[str] | None = None,
    fields: Iterable[RawFormField] | None = None,
    annotations: Iterable[RawAnnotation] | None = None,
) -> PageProgram:
    state = TextState(
        page.document,
        hidden_layers=(
            page.document.oc_hidden_layers() if hidden_layers is None else hidden_layers
        ),
        page_clip=page.effective_page_clip(),
    )
    page.consume_contents(state)
    state.run_accumulator.flush()
    body = CapturedProgram(
        runs=tuple(state.runs),
        glyphs=tuple(state.glyphs),
        drawings=tuple(state.drawings),
        inline_images=tuple(state.inline_images),
        lines=tuple(state.lines),
        text_boundaries=tuple(state.text_boundaries),
    )
    return PageProgram(
        body=body,
        appearances=capture_annotation_appearances(
            page, state, fields=fields, annotations=annotations
        ),
    )
