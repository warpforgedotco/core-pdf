# SPDX-License-Identifier: AGPL-3.0-only
"""Run the content stream and turn it into page evidence."""

from __future__ import annotations

import math
import re
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy

from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)
from core_pdf.impl.engine.model.geometry import rect_tuple
from core_pdf.impl.engine.model.glyphs import (
    GlyphObservation,
    GlyphUnicodeSemantics,
    glyph_unicode_semantics,
)
from core_pdf.impl.engine.model.runs import TextRun
from core_pdf.impl.engine.parse.model import (
    VECTOR_PAINT_KINDS,
    CapturedPage,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    PageEvidence,
    StrokedVectorTextEvidence,
    TextQualityStats,
    internal_analyze_text,
)
from core_pdf.impl.engine.parse.newstroke import NewstrokeDecode, decode_newstroke_drawings
from core_pdf.impl.engine.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.engine.spec.s_07_syntax_primitives.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_08_graphics.image_metadata import (
    image_filter_names,
)

WORD_TOKEN_RE = re.compile(r"\w+")
VECTOR_PAINT_OPERATION_WEIGHT = 3
LearnedUnicodeMap = Mapping[object, Mapping[bytes, str]]

# Thresholds for discarding a text layer that merely repeats another one. A layer needs
# enough tokens for an overlap ratio to mean anything, then enough overlap to be judged a
# duplicate. The two overlap floors differ because nested XObject layers repeat the page
# almost exactly, while separately clipped boxes overlap more loosely.
#
# These are unrelated to the HIDDEN_TEXT_VERIFY_* constants in parse.model, which gate the
# raster-to-text consistency check in parse.ocr. The similar values are a coincidence.
DUPLICATE_LAYER_MIN_TOKENS = 24
DUPLICATE_NESTED_LAYER_MIN_OVERLAP = 0.60
DUPLICATE_CLIPPED_LAYER_MIN_OVERLAP = 0.50


def internal_normalized_tokens(runs: tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for run in runs
        for token in WORD_TOKEN_RE.findall(str(getattr(run, "text", "")))
    )


def internal_token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    matched = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    return matched / min(len(left), len(right))


def internal_clip_bbox(run: Any) -> tuple[float, float, float, float] | None:
    for key, value in reversed(tuple(getattr(run, "provenance", ()))):
        if key != "clip_bbox" or not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(part) for part in value)
        except (TypeError, ValueError):
            return None
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    return None


def internal_discard_duplicate_nested_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    by_depth: dict[int, list[TextRun]] = {}
    for run in runs:
        by_depth.setdefault(run.xobject_depth, []).append(run)
    page_runs = by_depth.get(0, [])
    if not page_runs or len(by_depth) == 1:
        return runs
    tokens_by_depth = {
        depth: internal_normalized_tokens(nested) for depth, nested in by_depth.items()
    }
    page_tokens = tokens_by_depth[0]
    duplicate_depths = {
        depth
        for depth, nested in by_depth.items()
        if depth > 0
        and len(tokens_by_depth[depth]) >= DUPLICATE_LAYER_MIN_TOKENS
        and internal_token_overlap(tokens_by_depth[depth], page_tokens)
        >= DUPLICATE_NESTED_LAYER_MIN_OVERLAP
    }
    return tuple(run for run in runs if run.xobject_depth not in duplicate_depths)


def internal_discard_duplicate_clipped_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    groups: dict[tuple[float, float, float, float], list[TextRun]] = {}
    run_boxes: list[tuple[TextRun, tuple[float, float, float, float] | None]] = []
    for run in runs:
        box = internal_clip_bbox(run)
        run_boxes.append((run, box))
        if box is not None:
            groups.setdefault(box, []).append(run)
    if len(groups) < 2:
        return runs
    primary_box, primary_runs = max(
        groups.items(), key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
    )
    tokens_by_box = {box: internal_normalized_tokens(group) for box, group in groups.items()}
    primary_tokens = tokens_by_box[primary_box]
    if len(primary_tokens) < DUPLICATE_LAYER_MIN_TOKENS:
        return runs
    duplicate_boxes = {
        box
        for box, group in groups.items()
        if box != primary_box
        and len(tokens_by_box[box]) >= DUPLICATE_LAYER_MIN_TOKENS
        and internal_token_overlap(tokens_by_box[box], primary_tokens)
        >= DUPLICATE_CLIPPED_LAYER_MIN_OVERLAP
    }
    return tuple(run for run, box in run_boxes if box not in duplicate_boxes)


def internal_extractable_runs(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    active = tuple(run for run in runs if run.text and run.inside_active_clip)
    return internal_discard_duplicate_clipped_layers(
        internal_discard_duplicate_nested_layers(active)
    )


def internal_run_uses_actual_text(run: TextRun) -> bool:
    return any(
        key == "unicode_source" and value in {"actual_text", "structure_actual_text"}
        for key, value in run.provenance
    )


def internal_run_mcid(run: TextRun) -> int | None:
    for key, value in reversed(run.provenance):
        if key == "mcid" and type(value) is int:
            return value
    return None


def internal_apply_structure_actual_text(
    page: Any,
    runs: tuple[TextRun, ...],
) -> tuple[TextRun, ...]:
    if not any(internal_run_mcid(run) is not None for run in runs):
        return runs
    try:
        structure = page.structure
    except (IndexError, TypeError, ValueError):
        return runs
    replaced_mcids: set[int] = set()
    output: list[TextRun] = []
    for run in runs:
        mcid = internal_run_mcid(run)
        if mcid is None:
            output.append(run)
            continue
        if mcid in replaced_mcids:
            continue
        try:
            element = structure[mcid] if 0 <= mcid < len(structure) else None
            actual_text = getattr(element, "actual_text", None)
        except (IndexError, TypeError, ValueError):
            actual_text = None
        if not isinstance(actual_text, str) or not actual_text:
            output.append(run)
            continue
        replaced_mcids.add(mcid)
        output.append(
            internal_copy_run(
                run,
                text=actual_text,
                provenance=(("unicode_source", "structure_actual_text"),),
            )
        )
    return tuple(output)


def internal_glyph_evidence_fields(
    glyph_fields: Iterable[tuple[str, bool, object, bytes, str, float | None]],
    runs: tuple[TextRun, ...],
    learned_unicode: LearnedUnicodeMap | None = None,
) -> GlyphEvidence:
    authoritative = 0
    heuristic = 0
    unknown = 0
    unsupported = 0
    low_confidence = 0
    semantic_characters = 0
    glyph_count = 0
    previous_decoder: object | None = None
    learned: Mapping[bytes, str] | None = None
    for (
        glyph_text,
        ignored_visible,
        decoder,
        code_bytes,
        unicode_source,
        confidence,
    ) in glyph_fields:
        if not glyph_text or glyph_text.isspace():
            continue
        glyph_count += 1
        if decoder is not previous_decoder:
            learned = learned_unicode.get(decoder) if learned_unicode is not None else None
            previous_decoder = decoder
        candidate_text = learned.get(code_bytes) if learned is not None else None
        learned_text = (
            candidate_text if isinstance(candidate_text, str) and len(candidate_text) == 1 else None
        )
        text = learned_text or glyph_text
        semantics = (
            GlyphUnicodeSemantics.HEURISTIC
            if learned_text is not None
            else glyph_unicode_semantics(glyph_text, unicode_source)
        )
        if semantics is GlyphUnicodeSemantics.AUTHORITATIVE:
            authoritative += 1
            semantic_characters += (
                1 if len(text) == 1 else sum(not character.isspace() for character in text)
            )
        elif semantics is GlyphUnicodeSemantics.HEURISTIC:
            heuristic += 1
            semantic_characters += (
                1 if len(text) == 1 else sum(not character.isspace() for character in text)
            )
        elif semantics is GlyphUnicodeSemantics.UNSUPPORTED:
            unsupported += 1
        else:
            unknown += 1
        if learned_text is None and (confidence is None or confidence < 0.50):
            low_confidence += 1
    actual_text_characters = sum(
        sum(not character.isspace() for character in run.text)
        for run in runs
        if internal_run_uses_actual_text(run)
    )
    return GlyphEvidence(
        glyph_count=glyph_count,
        semantic_characters=semantic_characters,
        authoritative_glyphs=authoritative,
        heuristic_glyphs=heuristic,
        unknown_glyphs=unknown,
        unsupported_glyphs=unsupported,
        low_confidence_glyphs=low_confidence,
        actual_text_characters=actual_text_characters,
    )


def internal_hidden_text_is_trusted(
    *,
    native_characters: int,
    painted_characters: int,
    suspicious_characters: int,
    quality: TextQualityStats,
    glyphs: GlyphEvidence,
) -> bool:
    if native_characters < 100 or painted_characters >= native_characters * 0.20:
        return False
    if suspicious_characters / max(1, native_characters) > 0.01:
        return False
    if glyphs.actual_text_characters >= max(32, int(native_characters * 0.80)):
        return True
    if not glyphs.glyph_count:
        return False
    clean_mapping = glyphs.low_confidence_ratio <= 0.01 and glyphs.unsupported_ratio <= 0.01
    if not clean_mapping:
        return False
    if glyphs.authoritative_ratio >= 0.90:
        return True
    return (
        glyphs.mapped_ratio >= 0.99
        and glyphs.unknown_ratio <= 0.01
        and quality.wordlike_ratio >= 0.65
        and quality.noise_score <= 0.05
    )


def internal_hidden_text_needs_verification(evidence: PageEvidence) -> bool:
    """Select dense numeric scan layers for a cheap raster-to-text consistency check.

    Clean prose layers can be trusted directly. Numeric tables and indexes do not
    satisfy that language-shaped rule even when their embedded OCR text is accurate.
    Restricting the probe to clean, mapped, image-backed layers avoids spending another
    OCR pass on obviously corrupt encodings and on prose pages where native text can
    omit material visible content.
    """
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


def internal_copy_run(
    run: TextRun,
    *,
    text: str | None = None,
    visible: bool | None = None,
    provenance: tuple[tuple[str, object], ...] = (),
) -> TextRun:
    return TextRun(
        text=run.text if text is None else text,
        x0=run.x0,
        y0=run.y0,
        x1=run.x1,
        y1=run.y1,
        tx=run.tx,
        ty=run.ty,
        font_size=run.font_size,
        space_width=run.space_width,
        order=run.order,
        stream_order=run.stream_order,
        xobject_depth=run.xobject_depth,
        font_name=run.font_name,
        is_vertical=run.is_vertical,
        rotation_angle=run.rotation_angle,
        visible=run.visible if visible is None else visible,
        inside_active_clip=run.inside_active_clip,
        line_break_before=run.line_break_before,
        seqno=run.seqno,
        fill_color=run.fill_color,
        advance_bbox=run.advance_bbox,
        ink_bbox=run.ink_bbox,
        baseline=run.baseline,
        provenance=(*run.provenance, *provenance),
        confidence=run.confidence,
        glyph_clusters=run.glyph_clusters,
    )


def internal_promote_hidden_run(run: TextRun) -> TextRun:
    """Create an extraction-only view without changing PDF paint visibility."""
    return internal_copy_run(
        run,
        visible=True,
        provenance=(("extraction_visibility", "trusted-hidden-layer"),),
    )


def internal_observations_from_runs(runs: tuple[TextRun, ...]) -> ObservationBatch:
    """Build the native observation columns shared by capture and hidden-layer promotion."""
    if not runs:
        return ObservationBatch.empty()
    n = len(runs)
    texts = [run.text for run in runs]
    polygons = numpy.full((n, 8), numpy.nan, dtype=numpy.float32)
    source = numpy.full(n, int(ObservationSource.NATIVE), dtype=numpy.uint8)

    # Build columns from Python lists in one pass; per-element numpy stores
    # cost a dispatch each.
    box_rows: list[tuple[float, float, float, float]] = []
    confidence_values: list[float] = []
    sequence_values: list[int] = []
    visible_values: list[bool] = []
    rotation_values: list[int] = []
    font_size_values: list[float] = []
    line_break_values: list[bool] = []
    for i, run in enumerate(runs):
        c = run.coords
        box_rows.append((c[0], c[1], c[2], c[3]))
        conf = run.confidence
        confidence_values.append(conf if conf is not None else math.nan)
        seq = run.seqno
        sequence_values.append(seq if seq >= 0 else i)
        visible_values.append(run.visible)
        rotation_values.append(run.rotation_angle)
        font_size_values.append(c[6])
        line_break_values.append(run.line_break_before)

    boxes = numpy.asarray(box_rows, dtype=numpy.float32)
    confidence = numpy.asarray(confidence_values, dtype=numpy.float32)
    sequence = numpy.asarray(sequence_values, dtype=numpy.int64)
    visible = numpy.asarray(visible_values, dtype=numpy.bool_)
    rotation = numpy.asarray(rotation_values, dtype=numpy.int64)
    font_size = numpy.asarray(font_size_values, dtype=numpy.float32)
    line_break_before = numpy.asarray(line_break_values, dtype=numpy.bool_)

    return ObservationBatch(
        text=tuple(texts),
        bbox=boxes,
        polygon=polygons,
        source=source,
        confidence=confidence,
        sequence=sequence,
        visible=visible,
        rotation=rotation,
        font_size=font_size,
        line_break_before=line_break_before,
        references=runs,
    )


def internal_promoted_hidden_runs(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    return tuple(internal_promote_hidden_run(run) if not run.visible else run for run in runs)


def internal_promoted_hidden_observations(capture: CapturedPage) -> ObservationBatch:
    """Expose a verified hidden layer while preserving its original geometry and ordering."""
    return internal_observations_from_runs(internal_promoted_hidden_runs(capture.runs))


def internal_learned_glyph_text(
    glyph: GlyphObservation,
    learned_unicode: LearnedUnicodeMap | None = None,
) -> str | None:
    if learned_unicode is None:
        return None
    learned = learned_unicode.get(glyph.font_decoder)
    if learned is None:
        return None
    text = learned.get(glyph.code_bytes)
    return text if isinstance(text, str) and len(text) == 1 else None


def internal_apply_learned_unicode_to_run(
    run: TextRun,
    learned_unicode: LearnedUnicodeMap | None = None,
) -> TextRun:
    if not run.glyph_clusters or not learned_unicode:
        return run
    source = run.text
    cursor = 0
    output: list[str] = []
    changed = False
    previous_decoder: object | None = None
    learned: Mapping[bytes, str] | None = None
    for cluster in run.glyph_clusters:
        for decoder, code_bytes, glyph_text in cluster.iter_decode_fields():
            if decoder is not previous_decoder:
                learned = learned_unicode.get(decoder)
                previous_decoder = decoder
            replacement = learned.get(code_bytes) if learned is not None else None
            original = glyph_text
            if not isinstance(replacement, str) or len(replacement) != 1 or not original:
                continue
            position = source.find(original, cursor)
            if position < 0:
                continue
            output.append(source[cursor:position])
            output.append(replacement)
            cursor = position + len(original)
            changed = changed or replacement != original
    if not changed:
        return run
    output.append(source[cursor:])
    return internal_copy_run(
        run,
        text="".join(output),
        provenance=(("unicode_source", "learned_ocr"),),
    )


def internal_vector_complexity(drawings: tuple[Any, ...], grid_lines: Any) -> int:
    """Estimate vector workload without depending on graphics-state bookkeeping.

    Every derived segment contributes geometric work. Paint operations carry a larger
    fixed dispatch and raster cost, while clips, groups, and state markers are control
    records rather than visible vector content.
    """
    paint_operations = sum(
        getattr(drawing, "kind", None) in VECTOR_PAINT_KINDS for drawing in drawings
    )
    return len(grid_lines) + paint_operations * VECTOR_PAINT_OPERATION_WEIGHT


STROKED_VECTOR_COMPACT_DIMENSION = 4.0
STROKED_VECTOR_RENDER_DIMENSION = 5.0
STROKED_VECTOR_MIN_DOMINANT_PATHS = 300
STROKED_VECTOR_MIN_STYLE_PATHS = 8
STROKED_VECTOR_MIN_COMPACT_RATIO = 0.60
STROKED_VECTOR_MIN_AXIS_COVERAGE = 0.35


def internal_stroked_vector_style(drawing: Any) -> tuple[object, ...] | None:
    """Return a stable paint-style key for an opaque, solid stroked path."""
    if (
        getattr(drawing, "kind", None) not in {"stroke", "fillstroke"}
        or getattr(drawing, "path", None) is None
        or getattr(drawing, "stroke_pattern", None) is not None
    ):
        return None
    raw_color = getattr(drawing, "stroke_color", None)
    if not isinstance(raw_color, (list, tuple)) or not raw_color:
        return None
    try:
        color = tuple(float(component) for component in raw_color)
        raw_opacity = getattr(drawing, "stroke_opacity", None)
        opacity = 1.0 if raw_opacity is None else float(raw_opacity)
        line_width = float(getattr(drawing, "line_width", 0.0))
    except (TypeError, ValueError):
        return None
    if opacity <= 0.0 or not (0.0 < line_width <= 1.5):
        return None
    raw_dash = getattr(drawing, "dash_pattern", None)
    if raw_dash:
        try:
            dash = (tuple(float(value) for value in raw_dash[0]), float(raw_dash[1]))
        except (IndexError, TypeError, ValueError):
            return None
        if dash[0]:
            return None
    else:
        dash = None
    return (
        getattr(drawing, "kind", None),
        color,
        opacity,
        line_width,
        int(getattr(drawing, "line_cap", 0) or 0),
        int(getattr(drawing, "line_join", 0) or 0),
        dash,
        getattr(drawing, "blend_mode", None),
        getattr(drawing, "soft_mask_alpha", None),
    )


def internal_stroked_vector_text_evidence(
    drawings: tuple[Any, ...],
    *,
    page_width: float,
    page_height: float,
    rotation: int = 0,
) -> StrokedVectorTextEvidence:
    """Detect distributed single-line fonts from repeated compact path styles.

    Flattened CAD exports typically paint every glyph stroke as a tiny path with
    one of a few repeated styles.  Wires, frames, and component outlines are much
    larger in at least one axis.  Retaining the qualifying drawing indexes lets
    OCR rasterize only the likely text layer instead of recomposing the page.
    """
    if len(drawings) < 180 or rotation % 360 or page_width <= 0.0 or page_height <= 0.0:
        return StrokedVectorTextEvidence()

    # Values are total paths, compact paths, and compact-path bounds.
    styles: dict[tuple[object, ...], list[float]] = {}
    indexed: list[tuple[int, tuple[object, ...], tuple[float, float, float, float], float]] = []
    for index, drawing in enumerate(drawings):
        style = internal_stroked_vector_style(drawing)
        if style is None:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
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

    dominant_style, dominant = max(styles.items(), key=lambda item: item[1][1])
    del dominant_style
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
    if len(selected) < STROKED_VECTOR_MIN_DOMINANT_PATHS:
        return StrokedVectorTextEvidence()
    bbox = (
        min(box[0] for _, box in selected),
        min(box[1] for _, box in selected),
        max(box[2] for _, box in selected),
        max(box[3] for _, box in selected),
    )
    return StrokedVectorTextEvidence(
        trusted=True,
        drawing_indexes=tuple(index for index, _ in selected),
        bbox=bbox,
        candidate_paths=len(selected),
    )


def internal_uncovered_vector_area(
    drawings: tuple[Any, ...],
    observations: ObservationBatch,
    *,
    page_area: float | None = None,
) -> float | None:
    """Estimate filled vector area not represented by native text.

    This expensive signal is only used on text-bearing, vector-heavy pages. It
    is intentionally conservative: overlap with any native text subtracts from
    the filled path, so false OCR escalation is preferred over missing vector
    text.
    """
    if not drawings or not len(observations):
        return None
    if len(drawings) < 180:
        return None
    native = observations.bbox
    rectangles: list[tuple[float, float, float, float, float]] = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke"}:
            continue
        rect = rect_tuple(getattr(drawing, "rect", None))
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        # Page backgrounds and exporter-generated white canvases are not text.
        # Charging their bounding boxes as filled glyph area forces otherwise
        # sparse vector documents into a full-page OCR route.
        if page_area is not None and area >= max(1.0, page_area) * 0.80:
            continue
        if area > 0.0:
            rectangles.append((x0, y0, x1, y1, area))
    if not rectangles:
        return 0.0
    if len(rectangles) * len(native) > 65_536:
        native_index = SpatialIndex.from_boxes(native)
        uncovered = 0.0
        for x0, y0, x1, y1, area in rectangles:
            box = (x0, y0, x1, y1)
            covered = sum(
                bbox_intersection_area(box, hit.bbox) for hit in native_index.intersecting_hits(box)
            )
            uncovered += max(0.0, area - min(area, covered))
        return uncovered

    # Evaluate a bounded batch at a time. This keeps the overlap calculation
    # vectorized without allocating a drawings-by-observations matrix for the
    # entire page.
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
    capture: CapturedPage,
    decoded: NewstrokeDecode,
) -> CapturedPage:
    """Promote a page-level, template-verified vector font into native observations."""
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
            capture.drawings,
            observations,
            page_area=capture.evidence.page_area,
        ),
        text_quality=text_quality,
        all_text_quality=text_quality,
        painted_native_characters=characters,
        painted_text_coverage=text_coverage,
        vector_text_characters=characters,
        vector_text_candidate_segments=decoded.candidate_segments,
        vector_text_matched_segments=decoded.matched_segments,
        vector_text_sequences=decoded.sequences,
        vector_text_maximum_error=decoded.maximum_error,
        vector_text_trusted=True,
    )
    return replace(capture, observations=observations, runs=runs, evidence=evidence)


def internal_capture_from_program(
    page: Any,
    program: PageProgram,
    *,
    learned_unicode: LearnedUnicodeMap | None = None,
) -> CapturedPage:
    products = program.products
    program_runs = tuple(products.runs)
    glyphs_by_seqno: dict[int, list[str]] = defaultdict(list)
    for glyph_seqno, glyph_font_name in products.glyphs.iter_font_names():
        if glyph_font_name:
            glyphs_by_seqno[int(glyph_seqno)].append(glyph_font_name)
    glyph_seqnos = tuple(sorted(glyphs_by_seqno))
    enriched_runs: list[TextRun] = []
    for index, run in enumerate(program_runs):
        next_seqno = (
            program_runs[index + 1].seqno if index + 1 < len(program_runs) else float("inf")
        )
        lo = bisect_left(glyph_seqnos, run.seqno)
        hi = bisect_left(glyph_seqnos, next_seqno)
        # Runs are overwhelmingly single-font: find the majority name without
        # a Counter unless a second distinct name actually appears, and skip
        # the run copy when the name would not change.
        majority: str | None = None
        mixed = False
        for seqno in glyph_seqnos[lo:hi]:
            for font_name in glyphs_by_seqno[seqno]:
                if majority is None:
                    majority = font_name
                elif font_name != majority:
                    mixed = True
                    break
            if mixed:
                break
        if mixed:
            font_counts = Counter(
                font_name for seqno in glyph_seqnos[lo:hi] for font_name in glyphs_by_seqno[seqno]
            )
            majority = font_counts.most_common(1)[0][0]
        if majority is None or majority == run.font_name:
            enriched_runs.append(run)
        else:
            enriched = internal_copy_run(run)
            enriched.font_name = majority
            enriched_runs.append(enriched)
    structured_runs = internal_apply_structure_actual_text(page, tuple(enriched_runs))
    raw_runs = tuple(
        internal_apply_learned_unicode_to_run(run, learned_unicode)
        for run in internal_extractable_runs(structured_runs)
    )
    raw_text = "".join(run.text for run in raw_runs)
    painted_text = "".join(run.text for run in raw_runs if run.visible)
    raw_analysis = internal_analyze_text(raw_text)
    suspicious_characters = raw_analysis.suspicious_characters
    all_text_quality = raw_analysis.quality
    native_characters = raw_analysis.characters
    if painted_text == raw_text:
        painted_text_quality = all_text_quality
        painted_native_characters = native_characters
    else:
        painted_analysis = internal_analyze_text(painted_text)
        painted_text_quality = painted_analysis.quality
        painted_native_characters = painted_analysis.characters
    glyph_evidence = internal_glyph_evidence_fields(
        products.glyphs.iter_evidence_rows(),
        raw_runs,
        learned_unicode,
    )
    trusted_hidden_text = internal_hidden_text_is_trusted(
        native_characters=native_characters,
        painted_characters=painted_native_characters,
        suspicious_characters=suspicious_characters,
        quality=all_text_quality,
        glyphs=glyph_evidence,
    )
    runs = internal_promoted_hidden_runs(raw_runs) if trusted_hidden_text else raw_runs
    observations = internal_observations_from_runs(runs)
    visible_text = "".join(run.text for run in runs if run.visible)
    if visible_text == raw_text:
        visible_native_characters = native_characters
        visible_text_quality = all_text_quality
    elif visible_text == painted_text:
        visible_native_characters = painted_native_characters
        visible_text_quality = painted_text_quality
    else:
        visible_analysis = internal_analyze_text(visible_text)
        visible_text_quality = visible_analysis.quality
        visible_native_characters = visible_analysis.characters
    drawings = tuple(products.drawings)
    inline_images = tuple(products.inline_images)
    image_filters = tuple(
        filter_name
        for drawing in drawings
        if getattr(drawing, "kind", None) in {"image", "inline-image"}
        and isinstance(getattr(drawing, "dictionary", None), dict)
        for filter_name in image_filter_names(
            lookup_dict_key(getattr(drawing, "dictionary", None), "Filter")
        )
    )
    if inline_images:
        image_filters = (
            *image_filters,
            *(
                filter_name
                for image in inline_images
                for filter_name in image_filter_names(lookup_dict_key(image.dictionary, "Filter"))
            ),
        )
    page_width = float(page.width)
    page_height = float(page.height)
    page_area = max(1.0, page_width * page_height)
    visible = observations.visible
    boxes = observations.bbox
    visible_widths = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    visible_heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    text_coverage = min(
        1.0,
        float(numpy.sum(visible_widths * visible_heights * visible, dtype=numpy.float64))
        / page_area,
    )
    painted_mask = numpy.fromiter(
        (run.visible for run in raw_runs),
        dtype=numpy.bool_,
        count=len(raw_runs),
    )
    painted_text_coverage = min(
        1.0,
        float(numpy.sum(visible_widths * visible_heights * painted_mask, dtype=numpy.float64))
        / page_area,
    )
    visible_image_areas: list[float] = []
    visible_image_boxes: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        if getattr(drawing, "kind", None) != "image":
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        width = max(0.0, min(page_width, box[2]) - max(0.0, box[0]))
        height = max(0.0, min(page_height, box[3]) - max(0.0, box[1]))
        if width > 0.0 and height > 0.0:
            visible_image_areas.append(width * height)
            visible_image_boxes.append(
                (
                    max(0.0, box[0]),
                    max(0.0, box[1]),
                    min(page_width, box[2]),
                    min(page_height, box[3]),
                )
            )
    image_count = len(inline_images) + sum(
        area >= page_area * 0.001 for area in visible_image_areas
    )
    grid_lines = products.lines
    full_page_image = any(
        width >= page_width * 0.90 and height >= page_height * 0.90
        for width, height in (
            (
                max(0.0, box[2] - box[0]),
                max(0.0, box[3] - box[1]),
            )
            for box in visible_image_boxes
        )
    )
    uncovered_vector_area = internal_uncovered_vector_area(
        drawings,
        observations,
        page_area=page_area,
    )
    captured = CapturedPage(
        page=page,
        program=program,
        observations=observations,
        runs=runs,
        drawings=drawings,
        grid_lines=grid_lines,
        inline_images=inline_images,
        evidence=PageEvidence(
            page_area=page_area,
            native_characters=native_characters,
            visible_native_characters=visible_native_characters,
            suspicious_characters=suspicious_characters,
            image_count=image_count,
            image_area_ratio=min(1.0, sum(visible_image_areas) / page_area),
            vector_complexity=internal_vector_complexity(drawings, grid_lines),
            image_boxes=tuple(
                box
                for box, area in zip(
                    visible_image_boxes,
                    visible_image_areas,
                    strict=True,
                )
                if area >= page_area * 0.001
            ),
            image_filters=image_filters,
            text_coverage=text_coverage,
            full_page_image=full_page_image,
            uncovered_vector_area=uncovered_vector_area,
            text_quality=visible_text_quality,
            all_text_quality=all_text_quality,
            glyphs=glyph_evidence,
            painted_native_characters=painted_native_characters,
            painted_text_coverage=painted_text_coverage,
            trusted_hidden_text=trusted_hidden_text,
        ),
    )
    newstroke_decode: NewstrokeDecode | None = None
    newstroke_seconds = 0.0
    if not captured.runs and internal_requires_high_resolution_vector_ocr(captured):
        newstroke_started = time.perf_counter()
        newstroke_decode = decode_newstroke_drawings(captured.drawings)
        newstroke_seconds = time.perf_counter() - newstroke_started
        if newstroke_decode.trusted:
            captured = internal_capture_with_newstroke_text(captured, newstroke_decode)
    if not captured.evidence.vector_text_trusted:
        stroked_vector_text = internal_stroked_vector_text_evidence(
            drawings,
            page_width=page_width,
            page_height=page_height,
            rotation=int(getattr(page, "rotation", 0) or 0),
        )
        captured = replace(
            captured,
            evidence=replace(captured.evidence, stroked_vector_text=stroked_vector_text),
        )
    if newstroke_decode is not None:
        captured = replace(
            captured,
            newstroke_report={
                "accepted": newstroke_decode.trusted,
                "candidate_segments": newstroke_decode.candidate_segments,
                "matched_segments": newstroke_decode.matched_segments,
                "matched_coverage": newstroke_decode.matched_coverage,
                "glyphs": newstroke_decode.glyphs,
                "characters": newstroke_decode.characters,
                "sequences": newstroke_decode.sequences,
                "maximum_error": newstroke_decode.maximum_error,
                "seconds": newstroke_seconds,
            },
        )
    return captured


def capture_page(page: Any) -> CapturedPage:
    """Build the canonical page products once and derive routing evidence from them."""
    return internal_capture_from_program(page, page.get_page_program())


def internal_requires_high_resolution_vector_ocr(capture: CapturedPage) -> bool:
    """Identify pure-vector diagrams whose tiny stroked labels need the maximum raster."""
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
    for drawing in getattr(capture, "drawings", ()):
        kind = getattr(drawing, "kind", None)
        if kind not in VECTOR_PAINT_KINDS:
            continue
        paint_count += 1
        if kind != "stroke":
            continue
        stroke_count += 1
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is not None and max(box[2] - box[0], box[3] - box[1]) <= 6.0:
            compact_stroke_count += 1
    return (
        stroke_count >= 10_000
        and stroke_count >= paint_count * 0.95
        and compact_stroke_count >= stroke_count * 0.95
    )
