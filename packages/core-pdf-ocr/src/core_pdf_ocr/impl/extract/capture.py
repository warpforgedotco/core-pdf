# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import replace
from typing import Any, cast

import numpy

from core_pdf.impl._impl.capture.program import PageProgram
from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedLine
from core_pdf.impl._impl.extract.capture import (
    capture_page as native_capture_page,
)
from core_pdf.impl._impl.extract.capture import (
    internal_capture_from_program as native_capture_from_program,
)
from core_pdf.impl._impl.extract.capture import (
    internal_capture_runs,
    internal_observations_from_runs,
    internal_promoted_hidden_runs,
    internal_run_uses_actual_text,
    internal_STRUCTURE_UNSET,
)
from core_pdf.impl._impl.extract.capture import (
    internal_glyph_evidence_fields as native_glyph_evidence_fields,
)
from core_pdf.impl._impl.extract.contracts import (
    GlyphEvidence,
    ObservationBatch,
)
from core_pdf.impl._impl.extract.contracts import (
    PageAnalysis as NativePageAnalysis,
)
from core_pdf.impl._impl.extract.quality import internal_analyze_text
from core_pdf.impl._impl.graphics.filter_registry import declared_filter_names
from core_pdf.impl._impl.model.geometry import bbox_union, rect_tuple
from core_pdf.impl._impl.model.glyphs import (
    GlyphObservation,
    GlyphUnicodeSemantics,
    glyph_unicode_semantics,
)
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl._impl.model.text import normalize_extracted_text
from core_pdf_ocr.impl.extract.contracts import (
    VECTOR_PAINT_KINDS,
    PageAnalysis,
    PageEvidence,
    StrokedVectorTextEvidence,
)
from core_pdf_ocr.impl.extract.ocr.newstroke import NewstrokeDecode, decode_newstroke_drawings

LearnedUnicodeMap = Mapping[object, Mapping[bytes, str]]
VECTOR_PAINT_OPERATION_WEIGHT = 3
STROKED_VECTOR_COMPACT_DIMENSION = 4.0
STROKED_VECTOR_RENDER_DIMENSION = 5.0
STROKED_VECTOR_MIN_DOMINANT_PATHS = 300
STROKED_VECTOR_MIN_STYLE_PATHS = 8
STROKED_VECTOR_MIN_COMPACT_RATIO = 0.60
STROKED_VECTOR_MIN_AXIS_COVERAGE = 0.35


def internal_hidden_text_needs_verification(evidence: PageEvidence) -> bool:
    quality = evidence.all_text_quality
    glyphs = evidence.glyphs
    return (
        evidence.hidden_text_layer
        and evidence.full_page_image
        and evidence.suspicious_ratio <= 0.01
        and glyphs.glyph_count >= 100
        and glyphs.mapped_ratio >= 0.99
        and glyphs.unknown_ratio <= 0.01
        and glyphs.unsupported_ratio <= 0.01
        and glyphs.low_confidence_ratio <= 0.01
        and quality.token_count >= 100
        and quality.digit_token_ratio >= 0.18
        and quality.symbol_ratio <= 0.30
        and quality.noise_score <= 0.15
    )


def internal_promoted_hidden_observations(capture: PageAnalysis) -> ObservationBatch:
    references = capture.observations.references
    runs = (
        cast("tuple[TextRun, ...]", references)
        if references and all(isinstance(reference, TextRun) for reference in references)
        else capture.program.runs
    )
    return internal_observations_from_runs(internal_promoted_hidden_runs(runs))


def internal_apply_learned_unicode_to_run(
    run: TextRun,
    learned_unicode: LearnedUnicodeMap | None = None,
    *,
    glyph_replacements: dict[int, str] | None = None,
) -> TextRun:
    if not run.glyph_clusters or not learned_unicode or internal_run_uses_actual_text(run):
        return run
    source = run.text
    cursor = 0
    output: list[str] = []
    changed = False
    applied: dict[int, str] = {}
    previous_seqno: int | None = None
    previous_x: float | None = None
    for cluster in run.glyph_clusters:
        if cluster.glyphs:
            seqno = cluster.glyphs[0].seqno
            x = cluster.advance_bbox[0]
            if previous_seqno != seqno and previous_x is not None:
                if run.rotation_angle == 0 and x < previous_x:
                    return run
                if run.rotation_angle not in {0, 90} and x > previous_x:
                    return run
            previous_seqno = seqno
            previous_x = x
        original = normalize_extracted_text(cluster.text)
        if not original:
            continue
        position = source.find(original, cursor)
        if position < 0 or source[cursor:position].strip():
            return run
        output.append(source[cursor:position])
        cursor = position + len(original)
        replacement: str | None = None
        if cluster.glyphs:
            first = cluster.glyphs[0]
            learned = learned_unicode.get(first.font_decoder)
            candidate = learned.get(first.code_bytes) if learned is not None else None
            if (
                first.code_bytes
                and original.strip()
                and isinstance(candidate, str)
                and len(candidate) == 1
                and candidate.isprintable()
                and not candidate.isspace()
                and all(
                    glyph.font_decoder is first.font_decoder
                    and glyph.code_bytes == first.code_bytes
                    for glyph in cluster.glyphs
                )
            ):
                replacement = candidate
                applied.update((id(glyph), "") for glyph in cluster.glyphs)
                for glyph in cluster.glyphs:
                    if glyph.text and not glyph.text.isspace():
                        applied[id(glyph)] = candidate
                        break
        output.append(original if replacement is None else replacement)
        changed = changed or (replacement is not None and replacement != original)
    if source[cursor:].strip():
        return run
    if glyph_replacements is not None:
        glyph_replacements.update(applied)
    if not changed:
        return run
    output.append(source[cursor:])
    return run.replace(
        text="".join(output),
        provenance=(*run.provenance, ("unicode_source", "learned_ocr")),
        glyph_clusters=run.glyph_clusters,
    )


def internal_vector_complexity(
    drawings: tuple[CapturedDrawing, ...], grid_lines: tuple[CapturedLine, ...]
) -> int:
    paint_operations = sum(drawing.kind in VECTOR_PAINT_KINDS for drawing in drawings)
    return len(grid_lines) + paint_operations * VECTOR_PAINT_OPERATION_WEIGHT


def internal_stroked_vector_style(drawing: CapturedDrawing) -> tuple[object, ...] | None:
    if (
        drawing.kind not in {"stroke", "fillstroke"}
        or drawing.path is None
        or not drawing.stroke_color
    ):
        return None
    style = drawing.stroke_style_key()
    if style is None:
        return None
    opacity, line_width, dash = style[1], style[2], style[5]
    if opacity <= 0.0 or not (0.0 < line_width <= 1.5) or (dash is not None and dash[0]):
        return None
    return (drawing.kind, *style)


def internal_stroked_vector_text_evidence(
    drawings: tuple[CapturedDrawing, ...],
    *,
    page_width: float,
    page_height: float,
    rotation: int = 0,
) -> StrokedVectorTextEvidence:
    if len(drawings) < 180 or rotation % 360 or page_width <= 0.0 or page_height <= 0.0:
        return StrokedVectorTextEvidence()

    styles: dict[tuple[object, ...], list[float]] = {}
    indexed: list[tuple[int, tuple[object, ...], tuple[float, float, float, float], float]] = []
    for index, drawing in enumerate(drawings):
        style = internal_stroked_vector_style(drawing)
        if style is None:
            continue
        box = rect_tuple(drawing.rect)
        if box is None:
            continue
        maximum_dimension = max(box[2] - box[0], box[3] - box[1])
        if maximum_dimension <= 0.0:
            continue
        stats = styles.setdefault(
            style,
            [0.0, 0.0, math.inf, -math.inf, math.inf, -math.inf],
        )
        stats[0] += 1.0
        if maximum_dimension <= STROKED_VECTOR_COMPACT_DIMENSION:
            stats[1] += 1.0
            stats[2] = min(stats[2], box[0])
            stats[3] = max(stats[3], box[2])
            stats[4] = min(stats[4], box[1])
            stats[5] = max(stats[5], box[3])
        indexed.append((index, style, box, maximum_dimension))
    if not styles:
        return StrokedVectorTextEvidence()

    dominant = max(styles.values(), key=lambda stats: stats[1])
    dominant_paths = int(dominant[1])
    dominant_ratio = dominant_paths / max(1.0, dominant[0])
    width_coverage = (dominant[3] - dominant[2]) / page_width
    height_coverage = (dominant[5] - dominant[4]) / page_height
    if (
        dominant_paths < STROKED_VECTOR_MIN_DOMINANT_PATHS
        or dominant_ratio < STROKED_VECTOR_MIN_COMPACT_RATIO
        or width_coverage < STROKED_VECTOR_MIN_AXIS_COVERAGE
        or height_coverage < STROKED_VECTOR_MIN_AXIS_COVERAGE
    ):
        return StrokedVectorTextEvidence()

    selected_styles = {
        style
        for style, stats in styles.items()
        if stats[1] >= STROKED_VECTOR_MIN_STYLE_PATHS
        and stats[1] / max(1.0, stats[0]) >= STROKED_VECTOR_MIN_COMPACT_RATIO
    }
    selected = tuple(
        (index, box)
        for index, style, box, maximum_dimension in indexed
        if style in selected_styles and maximum_dimension <= STROKED_VECTOR_RENDER_DIMENSION
    )
    bbox = bbox_union(box for _, box in selected)
    return StrokedVectorTextEvidence(
        trusted=True,
        drawing_indexes=tuple(index for index, _ in selected),
        bbox=bbox,
        candidate_paths=len(selected),
    )


def internal_uncovered_vector_area(
    drawings: tuple[CapturedDrawing, ...],
    observations: ObservationBatch,
    *,
    page_area: float | None = None,
) -> float | None:
    if not drawings or not len(observations):
        return None
    if (
        len(drawings) < 180
        or sum(drawing.kind not in {"scope-begin", "scope-end"} for drawing in drawings) < 180
    ):
        return None
    native = observations.bbox
    rectangles: list[tuple[float, float, float, float, float]] = []
    for drawing in drawings:
        if drawing.kind not in {"fill", "fillstroke"}:
            continue
        rect = rect_tuple(drawing.rect)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if page_area is not None and area >= max(1.0, page_area) * 0.80:
            continue
        if area > 0.0:
            rectangles.append((x0, y0, x1, y1, area))
    if not rectangles:
        return 0.0
    uncovered = 0.0
    for offset in range(0, len(rectangles), 64):
        batch = numpy.asarray(rectangles[offset : offset + 64], dtype=numpy.float32)
        batch_x0 = batch[:, 0, None]
        batch_y0 = batch[:, 1, None]
        batch_x1 = batch[:, 2, None]
        batch_y1 = batch[:, 3, None]
        overlap_x = numpy.maximum(
            0.0,
            numpy.minimum(native[None, :, 2], batch_x1)
            - numpy.maximum(native[None, :, 0], batch_x0),
        )
        overlap_y = numpy.maximum(
            0.0,
            numpy.minimum(native[None, :, 3], batch_y1)
            - numpy.maximum(native[None, :, 1], batch_y0),
        )
        batch_covered = numpy.sum(overlap_x * overlap_y, axis=1, dtype=numpy.float64)
        areas = batch[:, 4]
        uncovered += float(
            numpy.sum(numpy.maximum(0.0, areas - numpy.minimum(areas, batch_covered)))
        )
    return uncovered


def internal_capture_with_newstroke_text(
    capture: PageAnalysis,
    decoded: NewstrokeDecode,
) -> PageAnalysis:
    runs = decoded.runs
    observations = internal_observations_from_runs(runs)
    text = "".join(run.text for run in runs)
    boxes = observations.bbox
    widths = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    text_coverage = min(
        1.0,
        float(numpy.sum(widths * heights, dtype=numpy.float64)) / capture.evidence.page_area,
    )
    analysis = internal_analyze_text(text)
    text_quality = analysis.quality
    characters = analysis.characters
    evidence = replace(
        capture.evidence,
        native_characters=characters,
        visible_native_characters=characters,
        suspicious_characters=analysis.suspicious_characters,
        text_coverage=text_coverage,
        uncovered_vector_area=internal_uncovered_vector_area(
            capture.program.drawings,
            observations,
            page_area=capture.evidence.page_area,
        ),
        text_quality=text_quality,
        all_text_quality=text_quality,
        painted_native_characters=characters,
        vector_text_candidate_segments=decoded.candidate_segments,
        vector_text_matched_segments=decoded.matched_segments,
        vector_text_trusted=True,
    )
    return replace(
        capture,
        observations=observations,
        program=replace(capture.program, body=replace(capture.program.body, runs=runs)),
        evidence=evidence,
    )


def internal_requires_high_resolution_vector_ocr(capture: PageAnalysis) -> bool:
    evidence = capture.evidence
    if not (
        evidence.image_count == 0
        and evidence.vector_complexity >= 100_000
        and evidence.text_coverage < 0.05
    ):
        return False
    paint_count = 0
    stroke_count = 0
    compact_stroke_count = 0
    for drawing in capture.program.drawings:
        kind = drawing.kind
        if kind not in VECTOR_PAINT_KINDS:
            continue
        paint_count += 1
        if kind != "stroke":
            continue
        stroke_count += 1
        box = rect_tuple(drawing.rect)
        if box is not None and max(box[2] - box[0], box[3] - box[1]) <= 6.0:
            compact_stroke_count += 1
    return (
        stroke_count >= 10_000
        and stroke_count >= paint_count * 0.95
        and compact_stroke_count >= stroke_count * 0.95
    )


def internal_glyph_evidence_fields(
    glyphs: tuple[GlyphObservation, ...],
    runs: tuple[TextRun, ...],
    replacements: Mapping[int, str],
) -> GlyphEvidence:
    evidence = native_glyph_evidence_fields(
        (
            (
                glyph.text,
                glyph.visible,
                glyph.font_decoder,
                glyph.code_bytes,
                glyph.unicode_source,
                glyph.confidence,
            )
            for glyph in glyphs
        ),
        runs,
    )
    if not replacements:
        return evidence
    changes = {
        "glyph_count": evidence.glyph_count,
        "semantic_characters": evidence.semantic_characters,
        "authoritative_glyphs": evidence.authoritative_glyphs,
        "heuristic_glyphs": evidence.heuristic_glyphs,
        "unknown_glyphs": evidence.unknown_glyphs,
        "unsupported_glyphs": evidence.unsupported_glyphs,
        "low_confidence_glyphs": evidence.low_confidence_glyphs,
    }
    counts = {
        GlyphUnicodeSemantics.AUTHORITATIVE: "authoritative_glyphs",
        GlyphUnicodeSemantics.HEURISTIC: "heuristic_glyphs",
        GlyphUnicodeSemantics.UNSUPPORTED: "unsupported_glyphs",
        GlyphUnicodeSemantics.UNKNOWN_IDENTIFIER: "unknown_glyphs",
    }
    for glyph in glyphs:
        replacement = replacements.get(id(glyph))
        text = glyph.text
        if replacement is None or not text or text.isspace():
            continue
        semantics = glyph_unicode_semantics(text, glyph.unicode_source)
        changes[counts[semantics]] -= 1
        if semantics in {GlyphUnicodeSemantics.AUTHORITATIVE, GlyphUnicodeSemantics.HEURISTIC}:
            changes["semantic_characters"] -= sum(not character.isspace() for character in text)
        if replacement:
            changes["heuristic_glyphs"] += 1
            changes["semantic_characters"] += 1
        else:
            changes["glyph_count"] -= 1
        if glyph.confidence is None or glyph.confidence < 0.50:
            changes["low_confidence_glyphs"] -= 1
    return replace(evidence, **changes)


def internal_enrich_capture(native: NativePageAnalysis) -> PageAnalysis:
    program = native.program
    image_filters = tuple(
        filter_name
        for drawing in program.drawings
        if drawing.kind == "image" and isinstance(drawing.dictionary, dict)
        for filter_name in declared_filter_names(drawing.dictionary.get("Filter"))
    ) + tuple(
        filter_name
        for image in program.inline_images
        for filter_name in declared_filter_names(image.dictionary.get("Filter"))
    )
    evidence = PageEvidence(
        **{name: getattr(native.evidence, name) for name in native.evidence.__fields__},
        vector_complexity=internal_vector_complexity(program.drawings, program.lines),
        image_filters=image_filters,
        uncovered_vector_area=internal_uncovered_vector_area(
            program.drawings,
            native.observations,
            page_area=native.evidence.page_area,
        ),
    )
    captured = PageAnalysis(
        page=native.page,
        width=native.width,
        height=native.height,
        rotation=native.rotation,
        fields=native.fields,
        annotations=native.annotations,
        program=program,
        observations=native.observations,
        evidence=evidence,
    )
    if not program.runs and internal_requires_high_resolution_vector_ocr(captured):
        decoded = decode_newstroke_drawings(program.drawings)
        if decoded.trusted:
            captured = internal_capture_with_newstroke_text(captured, decoded)
    if not captured.evidence.vector_text_trusted:
        captured = replace(
            captured,
            evidence=replace(
                captured.evidence,
                stroked_vector_text=internal_stroked_vector_text_evidence(
                    program.drawings,
                    page_width=native.width,
                    page_height=native.height,
                    rotation=native.rotation,
                ),
            ),
        )
    return captured


def internal_capture_from_program(
    page: Any,
    program: PageProgram,
    *,
    learned_unicode: LearnedUnicodeMap | None = None,
    structure: Any = internal_STRUCTURE_UNSET,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
) -> PageAnalysis:
    runs: tuple[TextRun, ...] | None = None
    evidence: GlyphEvidence | None = None
    if learned_unicode:
        replacements: dict[int, str] = {}
        runs = tuple(
            internal_apply_learned_unicode_to_run(
                run, learned_unicode, glyph_replacements=replacements
            )
            for run in internal_capture_runs(page, program, structure)
        )
        evidence = internal_glyph_evidence_fields(
            program.glyphs,
            runs,
            replacements,
        )
    return internal_enrich_capture(
        native_capture_from_program(
            page,
            program,
            structure=structure,
            fields=fields,
            annotations=annotations,
            runs=runs,
            glyph_evidence=evidence,
        )
    )


def capture_page(
    page: Any,
    *,
    structure: Any = internal_STRUCTURE_UNSET,
    hidden_layers: frozenset[str] | None = None,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
) -> PageAnalysis:
    return internal_enrich_capture(
        native_capture_page(
            page,
            structure=structure,
            hidden_layers=hidden_layers,
            fields=fields,
            annotations=annotations,
        )
    )
