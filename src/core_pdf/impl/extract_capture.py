# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy

from core_pdf.impl.capture_program import EXTRACTION_CAPTURE, CaptureOptions, PageProgram
from core_pdf.impl.capture_records import LayoutFormId
from core_pdf.impl.extract_contracts import (
    FULL_PAGE_IMAGE_COVERAGE,
    GlyphEvidence,
    ObservationBatch,
    ObservationSource,
    PageAnalysis,
    PageEvidence,
    TextQualityStats,
)
from core_pdf.impl.extract_quality import analyze_text
from core_pdf.impl.geometry import (
    bbox_union,
    interval_overlap,
    rect_tuple,
)
from core_pdf.impl.glyphs import (
    GlyphUnicodeSemantics,
    glyph_unicode_semantics,
)
from core_pdf.impl.runs import TextRun
from core_pdf.impl.text import collapse_ws


class StructureUnset:
    __slots__ = ()


STRUCTURE_UNSET = StructureUnset()
WORD_TOKEN_RE = re.compile(r"\w+")

DUPLICATE_LAYER_MIN_TOKENS = 24


def normalized_tokens(runs: Iterable[TextRun]) -> tuple[str, ...]:
    return tuple(token.casefold() for run in runs for token in WORD_TOKEN_RE.findall(run.text))


def clip_bbox(run: TextRun) -> tuple[float, float, float, float] | None:
    for key, value in reversed(run.provenance):
        if key != "clip_bbox" or not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(part) for part in value)
        except TypeError, ValueError:
            return None
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    return None


def glyphs_covered_by_extended_run(run: TextRun, extended: TextRun) -> bool:
    if not run.glyph_clusters or len(extended.text) <= len(run.text):
        return False
    if "".join(cluster.text for cluster in run.glyph_clusters) != run.text:
        return False
    if "".join(cluster.text for cluster in extended.glyph_clusters) != extended.text:
        return False
    return Counter(
        (cluster.text, cluster.advance_bbox) for cluster in run.glyph_clusters
    ) <= Counter((cluster.text, cluster.advance_bbox) for cluster in extended.glyph_clusters)


def discard_duplicate_layer_runs(
    runs: tuple[TextRun, ...],
    primary_indices: list[int],
    candidate_groups: Iterable[list[int]],
) -> tuple[TextRun, ...]:
    primary_runs = [runs[index] for index in primary_indices]
    primary_geometry = numpy.asarray(
        [(run.x0, run.y0, run.x1, run.y1) for run in primary_runs], dtype=numpy.float64
    )
    # Primaries sorted by left edge: only a prefix of them can start left of a
    # candidate's right edge, so each candidate is tested against that prefix.
    by_left = numpy.argsort(primary_geometry[:, 0], kind="stable")
    sorted_geometry = primary_geometry[by_left]
    sorted_left = sorted_geometry[:, 0]
    primary_tokens = [normalized_tokens((run,)) for run in primary_runs]
    primary_text = [collapse_ws(run.text) for run in primary_runs]
    duplicate_indices: set[int] = set()
    for indices in candidate_groups:
        tokens_by_index = {index: normalized_tokens((runs[index],)) for index in indices}
        if sum(map(len, tokens_by_index.values())) < DUPLICATE_LAYER_MIN_TOKENS:
            continue
        matched_indices: list[int] = []
        matched_tokens = 0
        covered_primary: set[int] = set()
        for index, tokens in tokens_by_index.items():
            if not tokens:
                continue
            run = runs[index]
            prefix = sorted_geometry[: numpy.searchsorted(sorted_left, run.x1, side="left")]
            intersects = (
                (prefix[:, 0] < run.x1)
                & (prefix[:, 2] > run.x0)
                & (prefix[:, 1] < run.y1)
                & (prefix[:, 3] > run.y0)
            )
            # Back in primary order, as the text below is joined in it.
            nearby = numpy.sort(by_left[: len(prefix)][intersects])
            local_text = " ".join(primary_text[int(position)] for position in nearby)
            candidate_text = collapse_ws(run.text)
            if f" {candidate_text} " in f" {local_text} ":
                matched_indices.append(index)
                matched_tokens += len(tokens)
            else:
                covered_primary.update(
                    int(position)
                    for position in nearby
                    if glyphs_covered_by_extended_run(primary_runs[int(position)], run)
                )
        if matched_tokens >= DUPLICATE_LAYER_MIN_TOKENS:
            duplicate_indices.update(matched_indices)
        if sum(len(primary_tokens[position]) for position in covered_primary) >= (
            DUPLICATE_LAYER_MIN_TOKENS
        ):
            duplicate_indices.update(primary_indices[position] for position in covered_primary)
    return tuple(run for index, run in enumerate(runs) if index not in duplicate_indices)


def discard_duplicate_nested_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    page_indices: list[int] = []
    by_form: dict[tuple[int, LayoutFormId | int], list[int]] = {}
    for index, run in enumerate(runs):
        if run.xobject_depth == 0:
            page_indices.append(index)
        else:
            form_id = next(
                (value for key, value in reversed(run.provenance) if key == "layout_form_id"),
                None,
            )
            group = form_id if isinstance(form_id, tuple) else run.stream_order
            by_form.setdefault((run.xobject_depth, group), []).append(index)
    if not page_indices or not by_form:
        return runs
    return discard_duplicate_layer_runs(runs, page_indices, by_form.values())


def discard_duplicate_clipped_layers(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    groups: dict[tuple[float, float, float, float], list[int]] = {}
    for index, run in enumerate(runs):
        box = clip_bbox(run)
        if box is not None:
            groups.setdefault(box, []).append(index)
    if len(groups) < 2:
        return runs
    primary_box, primary_indices = max(
        groups.items(), key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])
    )
    return discard_duplicate_layer_runs(
        runs, primary_indices, (indices for box, indices in groups.items() if box != primary_box)
    )


def extractable_runs(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    active = tuple(run for run in runs if run.text and run.inside_active_clip)
    return discard_duplicate_clipped_layers(discard_duplicate_nested_layers(active))


def run_uses_actual_text(run: TextRun) -> bool:
    return any(
        key == "unicode_source" and value in {"actual_text", "structure_actual_text"}
        for key, value in run.provenance
    )


def run_mcid(run: TextRun) -> int | None:
    for key, value in reversed(run.provenance):
        if key == "mcid" and type(value) is int:
            return value
    return None


def structure_actual_text_owner(element: Any) -> tuple[int, str] | None:
    owner: tuple[int, str] | None = None
    visited: set[int] = set()
    while element is not None:
        try:
            props = getattr(element, "props", None)
            marker = id(props if isinstance(props, dict) else element)
            if marker in visited:
                break
            visited.add(marker)
            actual_text = getattr(element, "actual_text", None)
            if isinstance(actual_text, str):
                owner = (marker, actual_text)
            element = getattr(element, "parent", None)
        except IndexError, TypeError, ValueError:
            break
    return owner


def apply_structure_actual_text(
    page: Any,
    runs: tuple[TextRun, ...],
    structure: Any = STRUCTURE_UNSET,
) -> tuple[TextRun, ...]:
    if not any(run_mcid(run) is not None for run in runs):
        return runs
    if structure is STRUCTURE_UNSET:
        try:
            structure = page.structure
        except IndexError, TypeError, ValueError:
            return runs
    if structure is None:
        return runs
    replacements: dict[int, TextRun] = {}
    output: list[TextRun] = []
    # The owner is a walk up from the MCID's element, and a page's runs share
    # few MCIDs, so each is walked once.
    owners: dict[int, tuple[int, str] | None] = {}
    for run in runs:
        mcid = run_mcid(run)
        if mcid is None:
            output.append(run)
            continue
        if mcid in owners:
            owner = owners[mcid]
        else:
            try:
                element = structure[mcid] if 0 <= mcid < len(structure) else None
            except IndexError, TypeError, ValueError:
                element = None
            owner = owners[mcid] = structure_actual_text_owner(element)
        if owner is None:
            output.append(run)
            continue
        marker, actual_text = owner
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
        replacement.absorb_extent(run)
        replacement.union_ink_bbox(run.ink_bbox)
        replacement.glyph_clusters += run.glyph_clusters
        replacement.visible = replacement.visible or run.visible
        replacement.inside_active_clip = replacement.inside_active_clip or run.inside_active_clip
    return tuple(output)


def glyph_evidence_fields(
    glyph_fields: Iterable[tuple[str, str, float | None]],
    runs: tuple[TextRun, ...],
) -> GlyphEvidence:
    authoritative = 0
    heuristic = 0
    unknown = 0
    unsupported = 0
    low_confidence = 0
    semantic_characters = 0
    glyph_count = 0
    for glyph_text, unicode_source, confidence in glyph_fields:
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
        if run_uses_actual_text(run)
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


def hidden_text_is_trusted(
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


def layout_bbox_for_run(run: TextRun) -> tuple[float, float, float, float]:
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


def promote_hidden_run(run: TextRun) -> TextRun:
    return run.replace(
        visible=True,
        provenance=(*run.provenance, ("extraction_visibility", "trusted-hidden-layer")),
    )


def observations_from_runs(runs: tuple[TextRun, ...]) -> ObservationBatch:
    if not runs:
        return ObservationBatch.empty()
    n = len(runs)
    texts = [run.text for run in runs]
    source = numpy.full(n, int(ObservationSource.NATIVE), dtype=numpy.uint8)

    box_rows: list[tuple[float, float, float, float]] = []
    confidence_values: list[float] = []
    sequence_values: list[int] = []
    visible_values: list[bool] = []
    rotation_values: list[int] = []
    font_size_values: list[float] = []
    line_break_values: list[bool] = []
    for i, run in enumerate(runs):
        box_rows.append(layout_bbox_for_run(run))
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
        source=source,
        confidence=confidence,
        sequence=sequence,
        visible=visible,
        rotation=rotation,
        font_size=font_size,
        line_break_before=line_break_before,
        references=runs,
    )


def promoted_hidden_runs(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    return tuple(promote_hidden_run(run) if not run.visible else run for run in runs)


def capture_runs(
    page: Any,
    program: PageProgram,
    structure: Any = STRUCTURE_UNSET,
) -> tuple[TextRun, ...]:
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
            enriched_runs.append(run.with_font_name(majority))
    structured_runs = apply_structure_actual_text(page, tuple(enriched_runs), structure)
    return extractable_runs(structured_runs)


def capture_from_program(
    page: Any,
    program: PageProgram,
    *,
    structure: Any = STRUCTURE_UNSET,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
    runs: tuple[TextRun, ...] | None = None,
    glyph_evidence: GlyphEvidence | None = None,
) -> PageAnalysis:
    raw_runs = runs if runs is not None else capture_runs(page, program, structure)
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
    raw_analysis = analyze_text(raw_text)
    suspicious_characters = raw_analysis.suspicious_characters
    all_text_quality = raw_analysis.quality
    native_characters = raw_analysis.characters
    if painted_text == raw_text:
        painted_text_quality = all_text_quality
        painted_native_characters = native_characters
    else:
        painted_analysis = analyze_text(painted_text)
        painted_text_quality = painted_analysis.quality
        painted_native_characters = painted_analysis.characters
    glyph_evidence = glyph_evidence or glyph_evidence_fields(
        ((glyph.text, glyph.unicode_source, glyph.confidence) for glyph in program.glyphs),
        raw_runs,
    )
    trusted_hidden_text = hidden_text_is_trusted(
        native_characters=native_characters,
        painted_characters=painted_native_characters,
        suspicious_characters=suspicious_characters,
        quality=all_text_quality,
        glyphs=glyph_evidence,
    )
    if trusted_hidden_text:
        runs = promoted_hidden_runs(raw_runs)
        visible_native_characters = native_characters
        visible_text_quality = all_text_quality
    else:
        runs = raw_runs
        visible_native_characters = painted_native_characters
        visible_text_quality = painted_text_quality
    observations = observations_from_runs(runs)
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
    return PageAnalysis(
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
            trusted_hidden_text=trusted_hidden_text,
        ),
    )


def capture_page(
    page: Any,
    *,
    structure: Any = STRUCTURE_UNSET,
    hidden_layers: frozenset[str] | None = None,
    fields: tuple[Any, ...] | None = None,
    annotations: tuple[Any, ...] | None = None,
    options: CaptureOptions = EXTRACTION_CAPTURE,
) -> PageAnalysis:
    # Extraction never rasterizes, so it defaults to EXTRACTION_CAPTURE. OCR
    # asks for the render payload back, because it renders the page it extracted.
    capture_options: dict[str, object] = {"options": options}
    if hidden_layers is not None:
        capture_options["hidden_layers"] = hidden_layers
    if fields is not None:
        capture_options["fields"] = fields
    if annotations is not None:
        capture_options["annotations"] = annotations
    program = page.get_page_program(**capture_options)
    return capture_from_program(
        page,
        program,
        structure=structure,
        fields=fields,
        annotations=annotations,
    )
