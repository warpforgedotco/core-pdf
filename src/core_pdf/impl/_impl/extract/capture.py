# SPDX-License-Identifier: AGPL-3.0-only
"""Run the content stream and turn it into page evidence."""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, cast

import numpy

from core_pdf.impl._impl.extract.contracts import (
    FULL_PAGE_IMAGE_COVERAGE,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    PageAnalysis,
    PageEvidence,
    TextQualityStats,
)
from core_pdf.impl._impl.extract.quality import internal_analyze_text
from core_pdf.impl._impl.model.geometry import (
    bbox_union,
    interval_overlap,
    rect_tuple,
    union_bbox,
)
from core_pdf.impl._impl.model.glyphs import (
    GlyphUnicodeSemantics,
    glyph_unicode_semantics,
)
from core_pdf.impl._impl.model.runs import TextRun
from core_pdf.impl.spec.s_07_content.marked_content import (
    extend_baseline,
    min_optional_confidence,
)
from core_pdf.impl.spec.s_07_content.page_program import PageProgram
from core_pdf.impl.spec.s_07_content.stream_state import LayoutFormId
from core_pdf.impl.types import Rectangle


class internal_StructureUnset:
    __slots__ = ()


internal_STRUCTURE_UNSET = internal_StructureUnset()
WORD_TOKEN_RE = re.compile(r"\w+")

# Thresholds for discarding a text layer that merely repeats another one. A layer needs
# enough matching tokens to distinguish duplication from incidental repetition. Nested
# streams require full local coverage of each discarded run; clipped layers retain their
# overlap threshold because separately clipped boxes can repeat only a localized region.
DUPLICATE_LAYER_MIN_TOKENS = 24
DUPLICATE_CLIPPED_LAYER_MIN_OVERLAP = 0.50


def internal_normalized_tokens(runs: Iterable[TextRun]) -> tuple[str, ...]:
    return tuple(token.casefold() for run in runs for token in WORD_TOKEN_RE.findall(run.text))


def internal_token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    matched = (Counter(left) & Counter(right)).total()
    return matched / min(len(left), len(right))


def internal_clip_bbox(run: TextRun) -> tuple[float, float, float, float] | None:
    for key, value in reversed(run.provenance):
        if key != "clip_bbox" or not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(cast(Any, part)) for part in value)
        except (TypeError, ValueError):
            return None
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    return None


def internal_discard_duplicate_nested_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    page_runs: list[TextRun] = []
    by_form: dict[tuple[int, LayoutFormId | int], list[int]] = {}
    for index, run in enumerate(runs):
        if run.xobject_depth == 0:
            page_runs.append(run)
        else:
            form_id = next(
                (value for key, value in reversed(run.provenance) if key == "layout_form_id"),
                None,
            )
            # The form path and transformed bounds survive child invocations;
            # stream_order advances when entering a child and is not restored.
            group = cast(LayoutFormId, form_id) if isinstance(form_id, tuple) else run.stream_order
            by_form.setdefault((run.xobject_depth, group), []).append(index)
    if not page_runs or not by_form:
        return runs
    page_geometry = numpy.asarray(
        [(run.x0, run.y0, run.x1, run.y1) for run in page_runs], dtype=numpy.float64
    )
    page_tokens = [internal_normalized_tokens((run,)) for run in page_runs]
    duplicate_indices: set[int] = set()
    for indices in by_form.values():
        tokens_by_index = {index: internal_normalized_tokens((runs[index],)) for index in indices}
        if sum(map(len, tokens_by_index.values())) < DUPLICATE_LAYER_MIN_TOKENS:
            continue
        matched_indices: list[int] = []
        matched_tokens = 0
        for index, tokens in tokens_by_index.items():
            if not tokens:
                continue
            run = runs[index]
            intersects = (
                (page_geometry[:, 0] < run.x1)
                & (page_geometry[:, 2] > run.x0)
                & (page_geometry[:, 1] < run.y1)
                & (page_geometry[:, 3] > run.y0)
            )
            local_tokens = Counter(
                token
                for page_index in numpy.flatnonzero(intersects)
                for token in page_tokens[int(page_index)]
            )
            if Counter(tokens) <= local_tokens:
                matched_indices.append(index)
                matched_tokens += len(tokens)
        # A stream may mix a duplicate overlay with unique text. Discard only
        # locally matched runs, never every form at the same nesting depth or
        # an entire run merely because some of its words repeat page text.
        if matched_tokens >= DUPLICATE_LAYER_MIN_TOKENS:
            duplicate_indices.update(matched_indices)
    return tuple(run for index, run in enumerate(runs) if index not in duplicate_indices)


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
    candidate_boxes = [
        box
        for box in groups
        if box != primary_box and len(tokens_by_box[box]) >= DUPLICATE_LAYER_MIN_TOKENS
    ]
    if not candidate_boxes:
        return runs
    candidate_geometry = numpy.asarray(candidate_boxes, dtype=numpy.float64)
    primary_geometry = numpy.asarray(
        [(run.x0, run.y0, run.x1, run.y1) for run in primary_runs],
        dtype=numpy.float64,
    )
    intersections = (
        (candidate_geometry[:, None, 0] < primary_geometry[None, :, 2])
        & (candidate_geometry[:, None, 2] > primary_geometry[None, :, 0])
        & (candidate_geometry[:, None, 1] < primary_geometry[None, :, 3])
        & (candidate_geometry[:, None, 3] > primary_geometry[None, :, 1])
    )
    duplicate_boxes: set[tuple[float, float, float, float]] = set()
    for box, intersects in zip(candidate_boxes, intersections, strict=True):
        # Compare with primary-layer text painted in the same region. A page's
        # table cells legitimately have separate clipping boxes and repeat much
        # of the surrounding vocabulary; comparing each cell with every token
        # on the page deleted non-duplicate rows from ISO 32000-2:2020 Table 22.
        local_tokens = internal_normalized_tokens(
            tuple(primary_runs[int(index)] for index in numpy.flatnonzero(intersects))
        )
        if (
            len(local_tokens) >= DUPLICATE_LAYER_MIN_TOKENS
            and internal_token_overlap(tokens_by_box[box], local_tokens)
            >= DUPLICATE_CLIPPED_LAYER_MIN_OVERLAP
        ):
            duplicate_boxes.add(box)
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
    structure: Any = internal_STRUCTURE_UNSET,
) -> tuple[TextRun, ...]:
    if not any(internal_run_mcid(run) is not None for run in runs):
        return runs
    if structure is internal_STRUCTURE_UNSET:
        try:
            structure = page.structure
        except (IndexError, TypeError, ValueError):
            return runs
    if structure is None:
        return runs
    replacements: dict[int, TextRun] = {}
    output: list[TextRun] = []
    for run in runs:
        mcid = internal_run_mcid(run)
        if mcid is None:
            output.append(run)
            continue
        try:
            element = structure[mcid] if 0 <= mcid < len(structure) else None
            actual_text = getattr(element, "actual_text", None)
        except (IndexError, TypeError, ValueError):
            actual_text = None
        if not isinstance(actual_text, str):
            output.append(run)
            continue
        # Several MCIDs may belong to one structure element. Its ActualText
        # replaces the entire element, and an empty string deliberately removes it.
        marker = id(element)
        replacement = replacements.get(marker)
        if replacement is None:
            replacement = run.replace(
                text=actual_text,
                provenance=(*run.provenance, ("unicode_source", "structure_actual_text")),
                glyph_clusters=run.glyph_clusters,
            )
            replacements[marker] = replacement
            output.append(replacement)
            continue
        replacement.x0 = min(replacement.x0, run.x0)
        replacement.y0 = min(replacement.y0, run.y0)
        replacement.x1 = max(replacement.x1, run.x1)
        replacement.y1 = max(replacement.y1, run.y1)
        replacement.advance_bbox = cast(
            Rectangle, union_bbox(replacement.advance_bbox, run.advance_bbox)
        )
        replacement.union_ink_bbox(run.ink_bbox)
        replacement.baseline = extend_baseline(replacement.baseline, run.baseline)
        replacement.confidence = min_optional_confidence(replacement.confidence, run.confidence)
        replacement.glyph_clusters += run.glyph_clusters
        replacement.visible = replacement.visible or run.visible
        replacement.inside_active_clip = replacement.inside_active_clip or run.inside_active_clip
    return tuple(output)


def internal_glyph_evidence_fields(
    glyph_fields: Iterable[tuple[str, bool, object, bytes, str, float | None]],
    runs: tuple[TextRun, ...],
) -> GlyphEvidence:
    authoritative = 0
    heuristic = 0
    unknown = 0
    unsupported = 0
    low_confidence = 0
    semantic_characters = 0
    glyph_count = 0
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
        text = glyph_text
        semantics = glyph_unicode_semantics(glyph_text, unicode_source)
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
        if confidence is None or confidence < 0.50:
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


def internal_layout_bbox_for_run(run: TextRun) -> tuple[float, float, float, float]:
    """Choose geometry that describes this occurrence rather than every font glyph."""
    bbox = (run.x0, run.y0, run.x1, run.y1)
    if run.is_vertical or run.rotation_angle % 180:
        return bbox
    font_size = abs(run.font_size)
    advance_height = run.y1 - run.y0
    if font_size <= 0.0 or advance_height <= font_size * 2.5:
        return bbox
    cluster_ink = bbox_union(
        cluster.ink_bbox for cluster in run.glyph_clusters if cluster.text.strip()
    )
    _, ink_y0, _, ink_y1 = cluster_ink or run.ink_bbox
    ink_height = ink_y1 - ink_y0
    if ink_height <= 0.0 or advance_height <= ink_height * 2.5:
        return bbox

    # ISO 32000-1:2008 and ISO 32000-2:2020, 9.8.1 define Ascent and
    # Descent over the entire font. A rare outlier glyph can therefore make a
    # valid descriptor box several lines tall. When actual ink proves that
    # happened, use a conventional one-em line box around the captured
    # baseline while retaining any ink that extends beyond it.
    baseline = run.baseline
    if baseline is None:
        return (run.x0, ink_y0, run.x1, ink_y1)
    baseline_y = (baseline[1] + baseline[3]) * 0.5
    return (
        run.x0,
        min(ink_y0, baseline_y - font_size * 0.2),
        run.x1,
        max(ink_y1, baseline_y + font_size * 0.8),
    )


def internal_promote_hidden_run(run: TextRun) -> TextRun:
    """Create an extraction-only view without changing PDF paint visibility."""
    return run.replace(
        visible=True,
        provenance=(*run.provenance, ("extraction_visibility", "trusted-hidden-layer")),
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
        box_rows.append(internal_layout_bbox_for_run(run))
        conf = run.confidence
        confidence_values.append(conf if conf is not None else math.nan)
        seq = run.seqno
        sequence_values.append(seq if seq >= 0 else i)
        visible_values.append(run.visible)
        rotation_values.append(run.rotation_angle)
        font_size_values.append(run.font_size)
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


def internal_capture_runs(
    page: Any,
    program: PageProgram,
    structure: Any = internal_STRUCTURE_UNSET,
) -> tuple[TextRun, ...]:
    """Normalize captured PDF runs before any optional extraction enrichment."""
    program_runs = program.runs
    glyphs_by_seqno: dict[int, list[str]] = defaultdict(list)
    for glyph in program.glyphs:
        if glyph.font_name:
            glyphs_by_seqno[glyph.seqno].append(glyph.font_name)
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
        for glyph_position in range(lo, hi):
            seqno = glyph_seqnos[glyph_position]
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
                font_name
                for glyph_position in range(lo, hi)
                for font_name in glyphs_by_seqno[glyph_seqnos[glyph_position]]
            )
            majority = font_counts.most_common(1)[0][0]
        if majority is None or majority == run.font_name:
            enriched_runs.append(run)
        else:
            enriched_runs.append(run.replace(font_name=majority))
    structured_runs = internal_apply_structure_actual_text(page, tuple(enriched_runs), structure)
    return internal_extractable_runs(structured_runs)


def internal_capture_from_program(
    page: Any,
    program: PageProgram,
    *,
    structure: Any = internal_STRUCTURE_UNSET,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
    runs: tuple[TextRun, ...] | None = None,
    glyph_evidence: GlyphEvidence | None = None,
) -> PageAnalysis:
    raw_runs = runs if runs is not None else internal_capture_runs(page, program, structure)
    painted_mask = numpy.fromiter(
        (run.visible for run in raw_runs),
        dtype=numpy.bool_,
        count=len(raw_runs),
    )
    raw_text = "".join(run.text for run in raw_runs)
    painted_text = (
        raw_text
        if bool(numpy.all(painted_mask))
        else "".join(run.text for run in raw_runs if run.visible)
    )
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
    glyph_evidence = glyph_evidence or internal_glyph_evidence_fields(
        (
            (
                glyph.text,
                glyph.visible,
                glyph.font_decoder,
                glyph.code_bytes,
                glyph.unicode_source,
                glyph.confidence,
            )
            for glyph in program.glyphs
        ),
        raw_runs,
    )
    trusted_hidden_text = internal_hidden_text_is_trusted(
        native_characters=native_characters,
        painted_characters=painted_native_characters,
        suspicious_characters=suspicious_characters,
        quality=all_text_quality,
        glyphs=glyph_evidence,
    )
    if trusted_hidden_text:
        runs = internal_promoted_hidden_runs(raw_runs)
        visible_native_characters = native_characters
        visible_text_quality = all_text_quality
    else:
        runs = raw_runs
        visible_native_characters = painted_native_characters
        visible_text_quality = painted_text_quality
    observations = internal_observations_from_runs(runs)
    drawings = program.drawings
    inline_images = program.inline_images
    page_width = float(page.width)
    page_height = float(page.height)
    page_rotation = int(getattr(page, "rotation", 0) or 0)
    page_area = max(1.0, page_width * page_height)
    visible = observations.visible
    boxes = observations.bbox
    box_areas = numpy.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    box_heights = numpy.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    numpy.multiply(box_areas, box_heights, out=box_areas)
    coverage_areas = numpy.multiply(box_areas, visible)
    text_coverage = min(
        1.0,
        float(numpy.sum(coverage_areas, dtype=numpy.float64)) / page_area,
    )
    numpy.multiply(box_areas, painted_mask, out=coverage_areas)
    painted_text_coverage = min(
        1.0,
        float(numpy.sum(coverage_areas, dtype=numpy.float64)) / page_area,
    )
    visible_image_areas: list[float] = []
    visible_image_boxes: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        if drawing.kind != "image":
            continue
        box = rect_tuple(drawing.rect)
        if box is None:
            continue
        width = interval_overlap(0.0, page_width, box[0], box[2])
        height = interval_overlap(0.0, page_height, box[1], box[3])
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
    full_page_image = any(
        width >= page_width * FULL_PAGE_IMAGE_COVERAGE
        and height >= page_height * FULL_PAGE_IMAGE_COVERAGE
        for width, height in (
            (
                max(0.0, box[2] - box[0]),
                max(0.0, box[3] - box[1]),
            )
            for box in visible_image_boxes
        )
    )
    captured = PageAnalysis(
        page=page,
        width=page_width,
        height=page_height,
        rotation=page_rotation,
        fields=fields or (),
        annotations=annotations or (),
        program=program,
        observations=observations,
        evidence=PageEvidence(
            page_area=page_area,
            native_characters=native_characters,
            visible_native_characters=visible_native_characters,
            suspicious_characters=suspicious_characters,
            image_count=image_count,
            image_area_ratio=min(1.0, sum(visible_image_areas) / page_area),
            image_boxes=tuple(
                box
                for box, area in zip(
                    visible_image_boxes,
                    visible_image_areas,
                    strict=True,
                )
                if area >= page_area * 0.001
            ),
            text_coverage=text_coverage,
            full_page_image=full_page_image,
            text_quality=visible_text_quality,
            all_text_quality=all_text_quality,
            glyphs=glyph_evidence,
            painted_native_characters=painted_native_characters,
            painted_text_coverage=painted_text_coverage,
            trusted_hidden_text=trusted_hidden_text,
        ),
    )
    return captured


def capture_page(
    page: Any,
    *,
    structure: Any = internal_STRUCTURE_UNSET,
    hidden_layers: frozenset[str] | None = None,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
) -> PageAnalysis:
    """Build the canonical page products once and derive routing evidence from them."""
    capture_options: dict[str, object] = {}
    if hidden_layers is not None:
        capture_options["hidden_layers"] = hidden_layers
    if fields is not None:
        capture_options["fields"] = fields
    if annotations is not None:
        capture_options["annotations"] = annotations
    program = page.get_page_program(**capture_options)
    return internal_capture_from_program(
        page,
        program,
        structure=structure,
        fields=fields,
        annotations=annotations,
    )
