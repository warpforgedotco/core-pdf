"""Capture soft-mask paints requested by structured-text image extraction."""

from __future__ import annotations

from collections.abc import Iterator

from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.capture.interpreter import TextState
from core_pdf.impl._impl.capture.program import CapturedProgram, PageProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedInlineImage
from core_pdf.impl._impl.model.geometry import intersect_bbox, rect_tuple, transform_bbox
from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.spec.s_07_content.image_capture import unit_square_placement
from core_pdf.impl.spec.s_07_content.soft_masks import SoftMaskSelection
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix


def capture_soft_masks(
    page: PdfPage,
    program: CapturedProgram | PageProgram,
    active: frozenset[tuple[str, int, int]] = frozenset(),
) -> Iterator[tuple[tuple[int, ...], CapturedProgram]]:
    """Interpret mask groups before their consumer, retaining nested paint order."""
    last_text_paint: tuple[object, ...] | None = None
    for command in program.commands:
        if isinstance(command, TextRun):
            continue
        provenance = dict(command.provenance) if isinstance(command, GlyphObservation) else {}
        selection = provenance.get("soft_mask") or getattr(command, "soft_mask", None)
        if isinstance(command, GlyphObservation):
            text_paint = (
                id(selection),
                command.text_object_id,
                command.text_render_mode,
                command.fill,
                command.fill_opacity,
                command.stroke_color,
                command.stroke_opacity,
                command.line_width,
                command.blend_mode,
                provenance.get("matrix_trace"),
                provenance.get("clip_bbox"),
                provenance.get("stream_order"),
            )
            if text_paint == last_text_paint:
                continue
            last_text_paint = text_paint
        else:
            last_text_paint = None
        if not isinstance(selection, SoftMaskSelection) or selection.source_key in active:
            continue
        resolver = page.document.resolver
        try:
            stream = resolver.resolve(selection.dictionary.get("G"))
            if not isinstance(stream, PdfStream):
                continue
            raw_matrix = resolver.deep_resolve(stream.dictionary.get("Matrix"))
            matrix = Matrix.from_operand(raw_matrix) if raw_matrix is not None else IDENTITY_MATRIX
            ctm = matrix.multiply(selection.ctm)
            bounds = resolver.resolve_box(stream.dictionary.get("BBox"))
            consumer_clip = getattr(command, "soft_mask_clip", None)
            if isinstance(command, GlyphObservation):
                consumer_clip = rect_tuple(provenance.get("clip_bbox"))
            elif isinstance(command, CapturedInlineImage):
                consumer_clip = intersect_bbox(
                    command.image_clip, unit_square_placement(command.ctm)[0]
                )
            elif isinstance(command, CapturedDrawing) and command.kind == "shading":
                consumer_clip = (
                    command.control_point_clip
                    if command.control_point_clip is not None
                    else command.shading_clip
                )
                shading_bounds = resolver.resolve_box((command.dictionary or {}).get("BBox"))
                if shading_bounds is not None and command.shading_matrix is not None:
                    consumer_clip = intersect_bbox(
                        consumer_clip, transform_bbox(shading_bounds, command.shading_matrix)
                    )
            mask_clip = transform_bbox(bounds, ctm) if bounds is not None else None
            resources = resolver.resolve_dict(stream.dictionary.get("Resources"))
            state = TextState(page.document, page_clip=page.effective_page_clip())
            state.consume_stream(
                resolver.resolve_stream(stream),
                resources or selection.resources,
                ctm,
                1,
                clip_bbox=intersect_bbox(mask_clip, consumer_clip),
            )
            captured = CapturedProgram(
                drawings=tuple(state.drawings),
                inline_images=tuple(state.inline_images),
                glyphs=tuple(state.glyphs),
            )
        except (PdfError, ValueError):
            continue
        order = (command.seqno, 0)
        yield order, captured
        for nested_order, nested in capture_soft_masks(
            page, captured, active | {selection.source_key}
        ):
            yield order + nested_order, nested


__all__ = ("capture_soft_masks",)
