# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable, Literal

from core_pdf.impl.engine.extraction.common import observation_resolver, page_geometry
from core_pdf.impl.engine.extraction.ocr import (
    postprocess as ocr_postprocess,
    selection as ocr_selection,
    table_regions as ocr_table_regions,
    text_analysis as ocr_text_analysis,
)
from core_pdf.impl.engine.extraction.ocr.candidates import (
    OcrCandidate,
    OcrPageTextResult,
)
from core_pdf.impl.engine.extraction.common.render import render_resolved_text_lines
from core_pdf.impl.engine.extraction.ocr.vector_text import VectorStrokeOcrResult
from core_pdf.impl.engine.layout.word_frequencies import word_rank

OcrLineSourceFamily = Literal[
    "native",
    "broad_page",
    "table",
    "figure",
    "embedded_image",
    "vector_stroke",
]
OcrLineType = Literal[
    "body_prose",
    "table_row",
    "chemical_symbolic",
    "chemical_hybrid",
    "diagram_label",
    "header_footer",
    "page_marker",
    "junk",
]


@dataclass(frozen=True)
class OcrLineTokenStats:
    token_count: int
    short_token_ratio: float
    punctuation_ratio: float
    numeric_ratio: float
    uppercase_ratio: float
    chemical_signal_count: int
    ocr_confusion_count: int


@dataclass(frozen=True)
class OcrLineVariant:
    family: OcrLineSourceFamily
    source_name: str
    bbox: tuple[float, float, float, float] | None
    text: str
    confidence: float | None
    quality: float
    artifact: float
    token_stats: OcrLineTokenStats
    line_type: OcrLineType
    observation: page_geometry.PageObservation
    break_before: int = 1
    provenance: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class OcrLineCluster:
    anchor_bbox: tuple[float, float, float, float] | None
    variants: tuple[OcrLineVariant, ...]
    selected: OcrLineVariant | None
    base_variant: OcrLineVariant | None = None
    operation: str = "keep"


@dataclass(frozen=True)
class OcrLineMatchSignals:
    geometry: float
    overlap_ratio: float
    anchor_overlap: float
    token_shape: float
    token_count_similarity: float
    score: float


@dataclass(frozen=True)
class OcrLineReplacementDecision:
    should_replace: bool
    confidence: float
    reasons: tuple[str, ...]
    signals: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class OcrLineSelection:
    text_lines: tuple[observation_resolver.ResolvedTextLine, ...]


@dataclass(frozen=True)
class OcrLineReconciliationSources:
    broad_page_result: OcrPageTextResult | None = None
    figure_result: OcrPageTextResult | None = None
    embedded_image_result: OcrPageTextResult | None = None
    vector_result: VectorStrokeOcrResult | None = None


@dataclass(frozen=True)
class DiagramGeometryContext:
    figure_variants: tuple[OcrLineVariant, ...]
    figure_observations: tuple[page_geometry.PageObservation, ...]
    region_bbox: tuple[float, float, float, float] | None
    page_bbox: tuple[float, float, float, float] | None
    figure_area_ratio: float
    figure_cluster_count: int
    short_figure_line_count: int
    figure_fragment_count: int
    broad_page_line_count: int
    broad_page_region_count: int
    broad_page_region_ratio: float
    diagram_heavy: bool


@dataclass(frozen=True)
class OcrBaseWindow:
    start_index: int
    end_index: int
    variant: OcrLineVariant


@dataclass(frozen=True)
class OcrBaseWindowEligibility:
    start_index: int
    end_index: int
    eligible: bool
    reason: str
    gap: float
    gap_ratio: float
    left_line_type: OcrLineType
    right_line_type: OcrLineType
    merged_line_type: OcrLineType | None = None
    merged_bbox: tuple[float, float, float, float] | None = None
    merged_text: str = ""


def reconcile_page_text_lines(
    base_text: str,
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    sources: OcrLineReconciliationSources,
) -> OcrLineSelection:
    resolved_base_lines = ensure_base_lines(base_text, base_lines)
    base_variants = tuple(
        variant_from_resolved_line(line, index)
        for index, line in enumerate(resolved_base_lines)
    )
    if not base_variants:
        return OcrLineSelection(())
    clusters = [
        OcrLineCluster(
            anchor_bbox=variant.bbox,
            variants=(variant,),
            selected=variant,
            base_variant=variant,
        )
        for variant in base_variants
    ]
    cluster_variants(
        clusters,
        broad_page_selected_output_variants(sources.broad_page_result),
    )
    cluster_variants(
        clusters,
        broad_page_variants(sources.broad_page_result),
    )
    cluster_variants(
        clusters,
        table_variants(sources.broad_page_result),
    )
    cluster_variants(
        clusters,
        figure_variants(sources.figure_result),
    )
    cluster_variants(
        clusters,
        embedded_image_variants(sources.embedded_image_result),
    )
    cluster_variants(
        clusters,
        vector_variants(sources.vector_result),
    )
    geometry_context = build_diagram_geometry_context(base_variants, sources)
    output_lines: list[observation_resolver.ResolvedTextLine] = list(
        resolved_base_lines
    )
    seen_tokens = set(ocr_text_analysis.normalized_text_tokens(base_text))
    replaced_indexes: set[int] = set()
    for index, cluster in enumerate(clusters):
        base_variant = cluster.base_variant
        if base_variant is None:
            continue
        winner, runner_up = choose_cluster_winner(cluster)
        selected = winner
        replacement_decision = (
            evaluate_replacement_decision(cluster, base_variant, selected)
            if selected is not None
            else None
        )
        if (
            selected is not None
            and replacement_decision is not None
            and replacement_decision.should_replace
        ):
            output_lines[index] = resolved_line_from_variant(
                selected,
                original=output_lines[index],
                contributing_observations=cluster_support_observations(
                    cluster,
                    selected,
                ),
            )
            seen_tokens.update(ocr_text_analysis.normalized_text_tokens(selected.text))
            replaced_indexes.add(index)
        elif selected is not None:
            output_lines[index] = resolved_line_with_cluster_support(
                output_lines[index],
                cluster,
                base_variant,
            )
    merge_proposals = select_merge_replacement_variants(
        resolved_base_lines,
        sources,
        replaced_indexes,
    )
    if merge_proposals:
        output_lines = apply_merge_replacement_variants(
            output_lines,
            merge_proposals,
        )
    output_lines = suppress_geometry_noise_lines(
        output_lines,
        geometry_context,
    )
    output_lines = apply_figure_source_priority_replacements(
        output_lines,
        sources.figure_result,
        geometry_context,
    )
    output_lines = compose_dominant_figure_page_lines(
        output_lines,
        sources,
        geometry_context,
    )
    seen_tokens = set(
        ocr_text_analysis.normalized_text_tokens(
            render_resolved_text_lines(tuple(output_lines))
        )
    )
    supplements = select_supplement_variants(
        clusters,
        seen_tokens,
        geometry_context,
    )
    if supplements:
        output_lines = insert_supplement_variants(output_lines, supplements)
        for variant in supplements:
            seen_tokens.update(ocr_text_analysis.normalized_text_tokens(variant.text))
    output_lines = upgrade_broad_page_fragments_with_figure_lines(
        output_lines,
        sources.figure_result,
        geometry_context,
    )
    return OcrLineSelection(tuple(output_lines))


def build_diagram_geometry_context(
    base_variants: tuple[OcrLineVariant, ...],
    sources: OcrLineReconciliationSources,
) -> DiagramGeometryContext:
    figure_lines = deduped_figure_context_variants(
        figure_variants(sources.figure_result)
    )
    figure_observations = tuple(variant.observation for variant in figure_lines)
    region_bbox = page_geometry.observation_union_bbox(figure_observations)
    page_bbox = page_geometry.observation_union_bbox(
        variant.observation for variant in base_variants
    )
    figure_area_ratio = 0.0
    if region_bbox is not None and page_bbox is not None:
        page_area = page_geometry.rect_area(page_bbox)
        if page_area > 0.0:
            figure_area_ratio = page_geometry.rect_area(region_bbox) / page_area
    figure_cluster_count = geometry_line_cluster_count(figure_lines)
    short_figure_line_count = sum(
        1 for variant in figure_lines if variant.token_stats.token_count <= 4
    )
    figure_fragment_count = sum(
        1 for variant in figure_lines if figure_variant_is_fragment_like(variant)
    )
    broad_page_lines = [
        variant for variant in base_variants if variant.family == "broad_page"
    ]
    broad_page_region_count = sum(
        1
        for variant in broad_page_lines
        if diagram_region_membership_score_for_bbox(variant.bbox, region_bbox) >= 0.55
    )
    broad_page_region_ratio = (
        broad_page_region_count / len(broad_page_lines) if broad_page_lines else 0.0
    )
    diagram_heavy = bool(
        len(figure_lines) >= 3
        and (
            figure_area_ratio >= 0.08
            or figure_cluster_count >= 2
            or short_figure_line_count >= 3
        )
    )
    return DiagramGeometryContext(
        figure_variants=figure_lines,
        figure_observations=figure_observations,
        region_bbox=region_bbox,
        page_bbox=page_bbox,
        figure_area_ratio=figure_area_ratio,
        figure_cluster_count=figure_cluster_count,
        short_figure_line_count=short_figure_line_count,
        figure_fragment_count=figure_fragment_count,
        broad_page_line_count=len(broad_page_lines),
        broad_page_region_count=broad_page_region_count,
        broad_page_region_ratio=broad_page_region_ratio,
        diagram_heavy=diagram_heavy,
    )


def deduped_figure_context_variants(
    variants: tuple[OcrLineVariant, ...],
) -> tuple[OcrLineVariant, ...]:
    selected: list[OcrLineVariant] = []
    seen: set[tuple[str, tuple[float, float, float, float] | None]] = set()
    for variant in variants:
        if not figure_variant_contributes_geometry_context(variant):
            continue
        key = (variant.text, variant.bbox)
        if key in seen:
            continue
        seen.add(key)
        selected.append(variant)
    return tuple(selected)


def figure_variant_contributes_geometry_context(variant: OcrLineVariant) -> bool:
    if variant.line_type in {"diagram_label", "chemical_symbolic", "chemical_hybrid"}:
        return True
    if variant.line_type != "body_prose":
        return False
    if variant.bbox is None:
        return False
    confidence = (
        float(variant.confidence)
        if isinstance(variant.confidence, int | float)
        else 0.0
    )
    if confidence < 55.0:
        return False
    text = variant.text.strip()
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if not text or not tokens:
        return False
    if figure_variant_looks_like_page_metadata(tokens):
        return False
    if len(tokens) <= 1 or len(tokens) > 8:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if figure_text_like_noise(variant):
        return False
    return readable_content_token_count(text) >= 1 or any(
        token.isdigit() for token in tokens
    )


def geometry_line_cluster_count(variants: tuple[OcrLineVariant, ...]) -> int:
    if not variants:
        return 0
    clusters: list[list[OcrLineVariant]] = []
    for variant in sorted(variants, key=variant_order_key):
        target: list[OcrLineVariant] | None = None
        for cluster in clusters:
            if any(
                variants_share_diagram_region(variant, existing) for existing in cluster
            ):
                target = cluster
                break
        if target is None:
            clusters.append([variant])
        else:
            target.append(variant)
    return len(clusters)


def variants_share_diagram_region(left: OcrLineVariant, right: OcrLineVariant) -> bool:
    if left.bbox is None or right.bbox is None:
        return False
    geometry = page_geometry.observation_geometry_match_score(
        left.observation,
        right.observation,
    )
    overlap = page_geometry.observation_overlap_ratio(
        left.observation,
        right.observation,
        denominator="smaller",
    )
    if geometry >= 0.35 or overlap >= 0.18:
        return True
    left_center = page_geometry.observation_center(left.observation)
    right_center = page_geometry.observation_center(right.observation)
    if left_center is None or right_center is None:
        return False
    left_height = max(1.0, page_geometry.observation_height(left.observation))
    right_height = max(1.0, page_geometry.observation_height(right.observation))
    left_width = max(1.0, page_geometry.observation_width(left.observation))
    right_width = max(1.0, page_geometry.observation_width(right.observation))
    return (
        abs(left_center[0] - right_center[0]) <= max(left_width, right_width) * 2.8
        and abs(left_center[1] - right_center[1])
        <= max(left_height, right_height) * 3.0
    )


def ensure_base_lines(
    base_text: str,
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    if base_lines and base_text == render_resolved_text_lines(base_lines):
        return base_lines
    text_lines = tuple(line.strip() for line in base_text.splitlines() if line.strip())
    if base_lines and len(text_lines) == len(base_lines):
        return replace_base_line_text(base_lines, text_lines)
    return ocr_postprocess.resolved_text_lines_from_strings(
        tuple(base_text.splitlines()),
        source="base_text",
        kind="base_text_line",
    )


def replace_base_line_text(
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    text_lines: tuple[str, ...],
) -> tuple[observation_resolver.ResolvedTextLine, ...]:
    replaced: list[observation_resolver.ResolvedTextLine] = []
    for line, text in zip(base_lines, text_lines, strict=True):
        observation = page_geometry.PageObservation(
            kind=line.observation.kind,
            source=line.observation.source,
            bbox=line.observation.bbox,
            advance_bbox=line.observation.advance_bbox,
            ink_bbox=line.observation.ink_bbox,
            confidence=line.observation.confidence,
            text=text,
            baseline=line.observation.baseline,
            provenance=line.observation.provenance,
        )
        replaced.append(
            observation_resolver.ResolvedTextLine(
                text,
                observation,
                break_before=line.break_before,
                contributing_observations=(observation,),
            )
        )
    return tuple(replaced)


def variant_from_resolved_line(
    line: observation_resolver.ResolvedTextLine,
    line_index: int,
) -> OcrLineVariant:
    observation = line.observation
    family = source_family_from_name(observation.source)
    stats = token_stats(line.text)
    line_type = classify_line_type(line.text, stats, family)
    return OcrLineVariant(
        family=family,
        source_name=observation.source,
        bbox=observation.bbox,
        text=line.text,
        confidence=observation.confidence,
        quality=ocr_text_analysis.text_ocr_quality_score(line.text),
        artifact=ocr_text_analysis.scanned_ocr_artifact_score(line.text),
        token_stats=stats,
        line_type=line_type,
        observation=page_geometry.PageObservation(
            kind=observation.kind,
            source=observation.source,
            bbox=observation.bbox,
            advance_bbox=observation.advance_bbox,
            ink_bbox=observation.ink_bbox,
            confidence=observation.confidence,
            text=line.text,
            baseline=observation.baseline,
            provenance=(
                *observation.provenance,
                *page_geometry.provenance_tuple(line_index=line_index),
            ),
        ),
        break_before=line.break_before,
        provenance=observation.provenance,
    )


def candidate_variants(
    candidate: OcrCandidate,
    *,
    family: OcrLineSourceFamily,
    kind: str,
) -> tuple[OcrLineVariant, ...]:
    geometry = page_geometry.image_space_from_ocr_candidate(None, candidate)
    variants: list[OcrLineVariant] = []
    for row_index, row in enumerate(candidate.result.line_rows):
        observation = page_geometry.page_observation_from_ocr_candidate_row(
            row,
            candidate=candidate,
            geometry=geometry,
            kind=kind,
        )
        if observation is None:
            continue
        text = observation.text.strip()
        if not text:
            continue
        stats = token_stats(text)
        variants.append(
            OcrLineVariant(
                family=family,
                source_name=candidate.name,
                bbox=observation.bbox,
                text=text,
                confidence=observation.confidence,
                quality=ocr_text_analysis.text_ocr_quality_score(text),
                artifact=ocr_text_analysis.scanned_ocr_artifact_score(text),
                token_stats=stats,
                line_type=classify_line_type(text, stats, family),
                observation=page_geometry.PageObservation(
                    kind=observation.kind,
                    source=observation.source,
                    bbox=observation.bbox,
                    advance_bbox=observation.advance_bbox,
                    ink_bbox=observation.ink_bbox,
                    confidence=observation.confidence,
                    text=text,
                    baseline=observation.baseline,
                    provenance=(
                        *observation.provenance,
                        *page_geometry.provenance_tuple(row_index=row_index),
                    ),
                ),
                provenance=observation.provenance,
            )
        )
    return tuple(variants)


def broad_page_variants(result: OcrPageTextResult | None) -> tuple[OcrLineVariant, ...]:
    if result is None:
        return ()
    variants: list[OcrLineVariant] = []
    seen: set[str] = set()
    for candidate in result.candidates:
        source_name = ocr_selection.ocr_variant_source_name(candidate.name)
        if not ocr_selection.broad_page_candidate_name(source_name):
            continue
        if candidate.region_count != 0 or not candidate.result.line_rows:
            continue
        if candidate.name in seen:
            continue
        variants.extend(
            candidate_variants(candidate, family="broad_page", kind="ocr_textline")
        )
        seen.add(candidate.name)
    return tuple(variants)


def broad_page_selected_output_variants(
    result: OcrPageTextResult | None,
) -> tuple[OcrLineVariant, ...]:
    if result is None or not result.selected_output_lines:
        return ()
    selected = result.candidate
    source_name = (
        f"{selected.name}:selected_output"
        if selected is not None
        else "broad_page:selected_output"
    )
    return resolved_line_variants(
        result.selected_output_lines,
        family="broad_page",
        source_name=source_name,
    )


def broad_page_replacement_variants(
    result: OcrPageTextResult | None,
) -> tuple[OcrLineVariant, ...]:
    variants = [
        *broad_page_selected_output_variants(result),
        *broad_page_variants(result),
    ]
    deduped: list[OcrLineVariant] = []
    seen: set[tuple[str, str, tuple[float, float, float, float] | None]] = set()
    for variant in variants:
        key = (variant.text, variant.source_name, variant.bbox)
        if key in seen:
            continue
        deduped.append(variant)
        seen.add(key)
    return tuple(deduped)


def table_variants(result: OcrPageTextResult | None) -> tuple[OcrLineVariant, ...]:
    if result is None:
        return ()
    variants: list[OcrLineVariant] = []
    for candidate in result.candidates:
        if candidate.name not in ocr_table_regions.OCR_TABLE_CANDIDATE_NAMES:
            continue
        if not candidate.result.line_rows:
            continue
        variants.extend(
            candidate_variants(candidate, family="table", kind="table_ocr_line")
        )
    return tuple(variants)


def figure_variants(result: OcrPageTextResult | None) -> tuple[OcrLineVariant, ...]:
    if result is None or result.candidate is None:
        return ()
    return candidate_variants(
        result.candidate, family="figure", kind="figure_text_line"
    )


def embedded_image_variants(
    result: OcrPageTextResult | None,
) -> tuple[OcrLineVariant, ...]:
    if result is None or result.candidate is None:
        return ()
    return candidate_variants(
        result.candidate, family="embedded_image", kind="ocr_textline"
    )


def vector_variants(result: VectorStrokeOcrResult | None) -> tuple[OcrLineVariant, ...]:
    if result is None or not result.lines:
        return ()
    variants: list[OcrLineVariant] = []
    for line_index, line in enumerate(result.lines):
        observation = page_geometry.page_observation_from_text_line(
            line,
            source="vector_stroke",
            kind="vector_text_line",
            line_index=line_index,
        )
        if observation is None:
            continue
        text = observation.text.strip()
        if not text:
            continue
        stats = token_stats(text)
        variants.append(
            OcrLineVariant(
                family="vector_stroke",
                source_name="vector_stroke",
                bbox=observation.bbox,
                text=text,
                confidence=observation.confidence,
                quality=ocr_text_analysis.text_ocr_quality_score(text),
                artifact=ocr_text_analysis.scanned_ocr_artifact_score(text),
                token_stats=stats,
                line_type=classify_line_type(text, stats, "vector_stroke"),
                observation=observation,
                provenance=observation.provenance,
            )
        )
    return tuple(variants)


def resolved_line_variants(
    lines: tuple[observation_resolver.ResolvedTextLine, ...],
    *,
    family: OcrLineSourceFamily,
    source_name: str | None = None,
) -> tuple[OcrLineVariant, ...]:
    variants: list[OcrLineVariant] = []
    for line_index, line in enumerate(lines):
        variant = variant_from_resolved_line(line, line_index)
        if family == variant.family and source_name is None:
            variants.append(variant)
            continue
        observation = page_geometry.PageObservation(
            kind=variant.observation.kind,
            source=source_name or variant.observation.source,
            bbox=variant.observation.bbox,
            advance_bbox=variant.observation.advance_bbox,
            ink_bbox=variant.observation.ink_bbox,
            confidence=variant.observation.confidence,
            text=variant.text,
            baseline=variant.observation.baseline,
            provenance=variant.observation.provenance,
        )
        variants.append(
            OcrLineVariant(
                family=family,
                source_name=source_name or variant.source_name,
                bbox=variant.bbox,
                text=variant.text,
                confidence=variant.confidence,
                quality=variant.quality,
                artifact=variant.artifact,
                token_stats=variant.token_stats,
                line_type=classify_line_type(variant.text, variant.token_stats, family),
                observation=observation,
                break_before=variant.break_before,
                provenance=variant.provenance,
            )
        )
    return tuple(variants)


def cluster_variants(
    clusters: list[OcrLineCluster],
    variants: Iterable[OcrLineVariant],
) -> None:
    for variant in variants:
        index = best_cluster_index(clusters, variant)
        if index is None:
            clusters.append(
                OcrLineCluster(
                    anchor_bbox=variant.bbox,
                    variants=(variant,),
                    selected=None,
                    base_variant=None,
                    operation="candidate_only",
                )
            )
            continue
        cluster = clusters[index]
        if any(
            existing.source_name == variant.source_name
            and existing.text == variant.text
            and existing.bbox == variant.bbox
            for existing in cluster.variants
        ):
            continue
        clusters[index] = OcrLineCluster(
            anchor_bbox=cluster.anchor_bbox or variant.bbox,
            variants=(*cluster.variants, variant),
            selected=cluster.selected,
            base_variant=cluster.base_variant,
            operation=cluster.operation,
        )


def best_cluster_index(
    clusters: list[OcrLineCluster],
    variant: OcrLineVariant,
) -> int | None:
    if variant.bbox is None:
        return None
    best_index: int | None = None
    best_score = 0.0
    for index, cluster in enumerate(clusters):
        base = cluster.base_variant
        if base is None or base.bbox is None:
            continue
        if not cluster_line_types_compatible(base, variant):
            continue
        if not cluster_bbox_match_is_plausible(base.bbox, variant.bbox):
            continue
        signals = cluster_match_signals(base, variant)
        if variant.family == "embedded_image" and signals.anchor_overlap <= 0.0:
            continue
        if not cluster_match_is_eligible_for_variants(base, variant, signals):
            continue
        if signals.score > best_score:
            best_index = index
            best_score = signals.score
    if best_score < 0.58:
        return None
    return best_index


def cluster_bbox_match_is_plausible(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_width = max(1.0, left[2] - left[0])
    left_height = max(1.0, left[3] - left[1])
    right_width = max(1.0, right[2] - right[0])
    right_height = max(1.0, right[3] - right[1])
    max_width = max(left_width, right_width)
    max_height = max(left_height, right_height)
    horizontal_gap = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical_gap = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    if vertical_gap > max_height * 1.6:
        return False
    if horizontal_gap > max_width * 2.4:
        return False
    left_center_x = (left[0] + left[2]) * 0.5
    left_center_y = (left[1] + left[3]) * 0.5
    right_center_x = (right[0] + right[2]) * 0.5
    right_center_y = (right[1] + right[3]) * 0.5
    if abs(left_center_y - right_center_y) > max_height * 1.9:
        return False
    return abs(left_center_x - right_center_x) <= max_width * 3.1


def cluster_match_signals(
    base: OcrLineVariant,
    variant: OcrLineVariant,
) -> OcrLineMatchSignals:
    geometry = page_geometry.observation_geometry_match_score(
        base.observation,
        variant.observation,
    )
    overlap_ratio = page_geometry.observation_overlap_ratio(
        base.observation,
        variant.observation,
        denominator="smaller",
    )
    anchor_overlap = anchor_token_overlap_score(base, variant)
    token_shape = token_shape_agreement_score(base.text, variant.text)
    token_count_similarity = token_count_similarity_score(base, variant)
    if base.line_type in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        score = (
            geometry * 0.42
            + overlap_ratio * 0.28
            + anchor_overlap * 0.12
            + token_shape * 0.14
            + token_count_similarity * 0.04
        )
    elif base.line_type == "table_row":
        score = (
            geometry * 0.40
            + overlap_ratio * 0.34
            + anchor_overlap * 0.18
            + token_shape * 0.05
            + token_count_similarity * 0.03
        )
    else:
        score = (
            geometry * 0.38
            + overlap_ratio * 0.32
            + anchor_overlap * 0.18
            + token_shape * 0.08
            + token_count_similarity * 0.04
        )
    return OcrLineMatchSignals(
        geometry=geometry,
        overlap_ratio=overlap_ratio,
        anchor_overlap=anchor_overlap,
        token_shape=token_shape,
        token_count_similarity=token_count_similarity,
        score=score,
    )


def cluster_match_is_eligible(
    line_type: OcrLineType,
    signals: OcrLineMatchSignals,
) -> bool:
    if line_type in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        if signals.geometry < 0.68 or signals.overlap_ratio < 0.42:
            return False
        if max(signals.anchor_overlap, signals.token_shape) < 0.34:
            return False
        return signals.score >= 0.58
    if line_type == "table_row":
        if signals.geometry < 0.74 or signals.overlap_ratio < 0.50:
            return False
        if signals.anchor_overlap < 0.24:
            return False
        return signals.score >= 0.62
    if signals.geometry < 0.72 or signals.overlap_ratio < 0.48:
        return False
    if signals.anchor_overlap < 0.28:
        return False
    return signals.score >= 0.60


def cluster_match_is_eligible_for_variants(
    base: OcrLineVariant,
    variant: OcrLineVariant,
    signals: OcrLineMatchSignals,
) -> bool:
    if figure_label_may_replace_weak_broad_page_line(base, variant):
        return (
            signals.geometry >= 0.78
            and signals.overlap_ratio >= 0.55
            and signals.token_shape >= 0.45
            and signals.score >= 0.72
        )
    return cluster_match_is_eligible(base.line_type, signals)


def choose_cluster_winner(
    cluster: OcrLineCluster,
) -> tuple[OcrLineVariant | None, OcrLineVariant | None]:
    variants = tuple(
        variant for variant in cluster.variants if variant.line_type != "header_footer"
    )
    if not variants:
        return (cluster.base_variant, None)
    support = Counter(consensus_key(variant.text) for variant in variants)
    ranked = sorted(
        variants,
        key=lambda variant: variant_sort_key(
            variant, support[consensus_key(variant.text)]
        ),
    )
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return winner, runner_up


def variant_sort_key(variant: OcrLineVariant, support: int) -> tuple[float, ...]:
    stats = variant.token_stats
    return (
        -float(support),
        variant.quality,
        variant.artifact,
        line_type_penalty(variant.line_type),
        stats.short_token_ratio,
        stats.punctuation_ratio,
        -chemical_plausibility_score(variant),
        -source_family_priority(variant.family, variant.line_type),
        -(variant.confidence or 0.0),
    )


def line_type_penalty(line_type: OcrLineType) -> float:
    return {
        "body_prose": 0.0,
        "table_row": 0.03,
        "chemical_hybrid": 0.02,
        "chemical_symbolic": 0.01,
        "diagram_label": 0.04,
        "page_marker": 0.20,
        "header_footer": 0.30,
        "junk": 0.40,
    }[line_type]


def source_family_priority(
    family: OcrLineSourceFamily, line_type: OcrLineType
) -> float:
    if line_type == "table_row":
        return {
            "table": 6.0,
            "broad_page": 5.0,
            "native": 4.0,
            "figure": 1.0,
            "embedded_image": 0.0,
            "vector_stroke": 2.0,
        }[family]
    if line_type in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        return {
            "figure": 6.0,
            "broad_page": 5.0,
            "vector_stroke": 4.0,
            "embedded_image": 3.0,
            "table": 2.0,
            "native": 1.0,
        }[family]
    return {
        "native": 6.0,
        "broad_page": 5.0,
        "table": 3.0,
        "figure": 2.0,
        "embedded_image": 1.0,
        "vector_stroke": 2.0,
    }[family]


def should_replace_base_variant(
    base: OcrLineVariant,
    selected: OcrLineVariant,
) -> bool:
    return evaluate_replacement_decision(
        OcrLineCluster(
            anchor_bbox=base.bbox,
            variants=(base, selected),
            selected=selected,
            base_variant=base,
        ),
        base,
        selected,
    ).should_replace


def variant_provenance_float(variant: OcrLineVariant, key: str) -> float:
    for name, value in variant.provenance:
        if name != key:
            continue
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def evaluate_replacement_decision(
    cluster: OcrLineCluster,
    base: OcrLineVariant,
    selected: OcrLineVariant,
) -> OcrLineReplacementDecision:
    reasons: list[str] = []
    if selected.text == base.text:
        return rejected_replacement("same_text")
    if base.bbox is None or selected.bbox is None:
        return rejected_replacement("missing_bbox")
    if selected.family == "embedded_image":
        return rejected_replacement("embedded_image_source")
    if selected.line_type in {"header_footer", "page_marker", "junk"}:
        return rejected_replacement("selected_line_type")
    if not replacement_line_types_compatible(base, selected):
        return rejected_replacement("line_type_mismatch")
    cluster_signals = cluster_match_signals(base, selected)
    if not cluster_match_is_eligible_for_variants(base, selected, cluster_signals):
        return rejected_replacement(
            "cluster_identity_weak",
            geometry=cluster_signals.geometry,
            overlap_ratio=cluster_signals.overlap_ratio,
            anchor_overlap=cluster_signals.anchor_overlap,
            token_shape=cluster_signals.token_shape,
            token_count_similarity=cluster_signals.token_count_similarity,
            cluster_score=cluster_signals.score,
        )
    base_tokens_normalized = ocr_text_analysis.normalized_text_tokens(base.text)
    selected_tokens_normalized = ocr_text_analysis.normalized_text_tokens(selected.text)
    if ocr_line_replacement_drops_connector_payload(
        base.text,
        base_tokens_normalized,
        selected_tokens_normalized,
    ):
        return rejected_replacement(
            "connector_payload_dropped",
            geometry=cluster_signals.geometry,
            overlap_ratio=cluster_signals.overlap_ratio,
            anchor_overlap=cluster_signals.anchor_overlap,
            token_shape=cluster_signals.token_shape,
        )
    if ocr_line_replacement_extends_short_numeric_token(
        base_tokens_normalized,
        selected_tokens_normalized,
        base_confidence=base.confidence or 0.0,
        selected_confidence=selected.confidence or 0.0,
    ):
        return rejected_replacement(
            "short_numeric_extension",
            geometry=cluster_signals.geometry,
            overlap_ratio=cluster_signals.overlap_ratio,
            anchor_overlap=cluster_signals.anchor_overlap,
            token_shape=cluster_signals.token_shape,
        )
    base_tokens = max(1, base.token_stats.token_count)
    selected_tokens = selected.token_stats.token_count
    if selected_tokens < max(2, int(base_tokens * 0.55)):
        return rejected_replacement("selected_too_short")
    if selected_tokens > max(base_tokens + 16, int(base_tokens * 1.9)):
        return rejected_replacement("selected_too_long")
    if selected.quality > min(0.48, base.quality + 0.06):
        return rejected_replacement("quality_not_better")
    if selected.line_type == "body_prose" and selected.family not in {
        "native",
        "broad_page",
    }:
        return rejected_replacement("body_prose_source")
    if selected.line_type == "table_row" and selected.family not in {
        "table",
        "broad_page",
    }:
        return rejected_replacement("table_row_source")
    if (
        base.line_type == "body_prose"
        and selected.family == "broad_page"
        and selected.token_stats.numeric_ratio > base.token_stats.numeric_ratio + 0.12
        and selected.token_stats.token_count >= base.token_stats.token_count
    ):
        return rejected_replacement(
            "broad_page_numeric_noise",
            geometry=cluster_signals.geometry,
            overlap_ratio=cluster_signals.overlap_ratio,
            anchor_overlap=cluster_signals.anchor_overlap,
            token_shape=cluster_signals.token_shape,
            selected_numeric_ratio=selected.token_stats.numeric_ratio,
            base_numeric_ratio=base.token_stats.numeric_ratio,
        )
    base_readable = readable_content_token_count(base.text)
    selected_readable = readable_content_token_count(selected.text)
    if (
        base.line_type == "body_prose"
        and selected.line_type == "body_prose"
        and base_readable >= 1
        and selected_readable == 0
        and selected.token_stats.short_token_ratio >= 0.60
        and selected.token_stats.token_count <= min(4, base.token_stats.token_count)
        and all(token.isalpha() for token in selected_tokens_normalized)
    ):
        return rejected_replacement("readability_regressed")
    if (
        base.family == "native"
        and base.line_type == "body_prose"
        and selected.family == "broad_page"
        and "psm11" in selected.source_name
        and base_readable >= 8
        and selected_readable < base_readable * 0.65
    ):
        return rejected_replacement(
            "native_prose_readable_token_loss",
            geometry=cluster_signals.geometry,
            overlap_ratio=cluster_signals.overlap_ratio,
            anchor_overlap=cluster_signals.anchor_overlap,
            token_shape=cluster_signals.token_shape,
            base_readable=float(base_readable),
            selected_readable=float(selected_readable),
        )

    geometry_score = cluster_signals.geometry
    overlap_ratio = cluster_signals.overlap_ratio
    anchor_score = cluster_signals.anchor_overlap
    token_shape_score = cluster_signals.token_shape
    support_score = cluster_support_score(cluster, selected)
    plausibility_gain = chemical_plausibility_score(
        selected
    ) - chemical_plausibility_score(base)
    artifact_gain = base.artifact - selected.artifact
    quality_gain = base.quality - selected.quality
    punctuation_gain = (
        base.token_stats.punctuation_ratio - selected.token_stats.punctuation_ratio
    )
    short_token_gain = (
        base.token_stats.short_token_ratio - selected.token_stats.short_token_ratio
    )
    confusion_gain = float(
        base.token_stats.ocr_confusion_count - selected.token_stats.ocr_confusion_count
    )
    confidence_gain = ((selected.confidence or 0.0) - (base.confidence or 0.0)) / 100.0
    language_prior_gain = variant_provenance_float(
        selected,
        "language_prior_gain",
    )
    token_delta_ratio = abs(selected_tokens - base_tokens) / base_tokens
    if base.line_type == "body_prose" and selected.family == "broad_page":
        token_ratio = selected_tokens / base_tokens
        if token_ratio < 0.90 and quality_gain < 0.05:
            return rejected_replacement(
                "broad_page_body_prose_too_short",
                geometry=cluster_signals.geometry,
                overlap_ratio=cluster_signals.overlap_ratio,
                anchor_overlap=cluster_signals.anchor_overlap,
                token_shape=cluster_signals.token_shape,
                selected_tokens=float(selected_tokens),
                base_tokens=float(base_tokens),
                token_ratio=token_ratio,
                quality_gain=quality_gain,
            )
        if quality_gain < 0.02 and selected.text.casefold() in base.text.casefold():
            return rejected_replacement(
                "broad_page_body_prose_substring_truncation",
                geometry=cluster_signals.geometry,
                overlap_ratio=cluster_signals.overlap_ratio,
                anchor_overlap=cluster_signals.anchor_overlap,
                token_shape=cluster_signals.token_shape,
                quality_gain=quality_gain,
                selected_tokens=float(selected_tokens),
                base_tokens=float(base_tokens),
                token_ratio=token_ratio,
            )
    geometry_identity = (
        geometry_score * 0.56
        + overlap_ratio * 0.29
        + anchor_score * 0.10
        + token_shape_score * 0.05
    )

    if geometry_identity < 0.72:
        return rejected_replacement(
            "geometry_not_decisive",
            geometry=geometry_score,
            overlap_ratio=overlap_ratio,
            anchor_overlap=anchor_score,
            token_shape=token_shape_score,
            geometry_identity=geometry_identity,
        )

    confidence = (
        geometry_score * 0.24
        + overlap_ratio * 0.20
        + anchor_score * 0.10
        + token_shape_score * 0.05
        + geometry_identity * 0.13
        + max(0.0, min(1.0, quality_gain / 0.18)) * 0.08
        + max(0.0, min(1.0, artifact_gain / 0.30)) * 0.05
        + max(0.0, min(1.0, punctuation_gain / 0.18)) * 0.02
        + max(0.0, min(1.0, short_token_gain / 0.35)) * 0.03
        + max(0.0, min(1.0, confusion_gain / 6.0)) * 0.02
        + max(0.0, min(1.0, plausibility_gain / 2.5)) * 0.04
        + max(0.0, min(1.0, confidence_gain / 0.18)) * 0.02
        + max(0.0, min(1.0, language_prior_gain / 0.25)) * 0.04
        + support_score * 0.06
        - min(0.12, token_delta_ratio * 0.10)
    )

    if geometry_score >= 0.82:
        reasons.append("strong_geometry")
    if overlap_ratio >= 0.62:
        reasons.append("bbox_overlap")
    if anchor_score >= 0.50:
        reasons.append("anchor_match")
    if token_shape_score >= 0.68:
        reasons.append("token_shape_match")
    if quality_gain >= 0.03:
        reasons.append("quality_gain")
    if artifact_gain >= 0.08:
        reasons.append("artifact_gain")
    if support_score >= 0.50:
        reasons.append("multi_source_support")
    if plausibility_gain > 0.0:
        reasons.append("plausibility_gain")
    if language_prior_gain >= 0.07:
        reasons.append("language_prior_gain")
    required_confidence = replacement_threshold(base.line_type)
    if (
        language_prior_gain >= 0.07
        and geometry_score >= 0.85
        and anchor_score >= 0.65
        and token_shape_score >= 0.75
    ):
        required_confidence -= 0.03

    if (
        base.line_type in {"chemical_symbolic", "chemical_hybrid"}
        and plausibility_gain <= 0.0
    ):
        return rejected_replacement(
            "plausibility_not_improved",
            geometry=geometry_score,
            overlap_ratio=overlap_ratio,
            anchor_overlap=anchor_score,
            token_shape=token_shape_score,
            support=support_score,
            geometry_identity=geometry_identity,
            confidence=confidence,
        )
    if (
        base.token_stats.short_token_ratio >= 0.72
        and base.token_stats.punctuation_ratio >= 0.22
        and short_token_gain <= 0.10
    ):
        return rejected_replacement(
            "noise_not_reduced",
            geometry=geometry_score,
            overlap_ratio=overlap_ratio,
            anchor_overlap=anchor_score,
            token_shape=token_shape_score,
            support=support_score,
            geometry_identity=geometry_identity,
            confidence=confidence,
        )
    if confidence < required_confidence:
        return rejected_replacement(
            "replacement_confidence_low",
            geometry=geometry_score,
            overlap_ratio=overlap_ratio,
            anchor_overlap=anchor_score,
            token_shape=token_shape_score,
            support=support_score,
            quality_gain=quality_gain,
            artifact_gain=artifact_gain,
            plausibility_gain=plausibility_gain,
            language_prior_gain=language_prior_gain,
            geometry_identity=geometry_identity,
            required_confidence=required_confidence,
            confidence=confidence,
        )
    return OcrLineReplacementDecision(
        should_replace=True,
        confidence=confidence,
        reasons=tuple(reasons),
        signals=(
            ("geometry", geometry_score),
            ("overlap_ratio", overlap_ratio),
            ("anchor_overlap", anchor_score),
            ("token_shape", token_shape_score),
            ("geometry_identity", geometry_identity),
            ("support", support_score),
            ("quality_gain", quality_gain),
            ("artifact_gain", artifact_gain),
            ("punctuation_gain", punctuation_gain),
            ("short_token_gain", short_token_gain),
            ("confusion_gain", confusion_gain),
            ("plausibility_gain", plausibility_gain),
            ("confidence_gain", confidence_gain),
            ("language_prior_gain", language_prior_gain),
            ("required_confidence", required_confidence),
            ("replacement_confidence", confidence),
        ),
    )


def ocr_line_replacement_drops_connector_payload(
    base_text: str,
    base_tokens: list[str],
    selected_tokens: list[str],
) -> bool:
    if not any(ch in base_text for ch in "-‐‑‒–—−"):
        return False
    base_counts = Counter(base_tokens)
    selected_counts = Counter(selected_tokens)
    if not selected_counts or not all(
        count <= base_counts[token] for token, count in selected_counts.items()
    ):
        return False
    if base_counts == selected_counts:
        return False
    return any(
        count > selected_counts[token] and ocr_line_connector_payload_token(token)
        for token, count in base_counts.items()
    )


def ocr_line_connector_payload_token(token: str) -> bool:
    if token == "_":
        return False
    if token.isdigit():
        return True
    return token.isalpha() and len(token) >= 2


def ocr_line_replacement_extends_short_numeric_token(
    base_tokens: list[str],
    selected_tokens: list[str],
    *,
    base_confidence: float,
    selected_confidence: float,
) -> bool:
    if len(base_tokens) != len(selected_tokens):
        return False
    changed = [
        (base, selected)
        for base, selected in zip(base_tokens, selected_tokens, strict=False)
        if base != selected
    ]
    if len(changed) != 1:
        return False
    base, selected = changed[0]
    if not (base.isdigit() and selected.isdigit()):
        return False
    if not (len(base) <= 2 and selected.startswith(base)):
        return False
    if len(selected) <= len(base):
        return False
    return base_confidence >= 50.0 or selected_confidence <= base_confidence + 5.0


def select_merge_replacement_variants(
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    sources: OcrLineReconciliationSources,
    replaced_indexes: set[int],
) -> list[tuple[OcrBaseWindow, OcrLineVariant, OcrLineReplacementDecision]]:
    windows = base_merge_windows(base_lines, replaced_indexes)
    if not windows:
        return []
    variants = broad_page_replacement_variants(sources.broad_page_result)
    proposals: list[
        tuple[OcrBaseWindow, OcrLineVariant, OcrLineReplacementDecision]
    ] = []
    used_variant_keys: set[
        tuple[str, str, tuple[float, float, float, float] | None]
    ] = set()
    for window in windows:
        best_variant: OcrLineVariant | None = None
        best_decision: OcrLineReplacementDecision | None = None
        for variant in variants:
            if (
                variant.line_type == "body_prose"
                and window.variant.line_type != "body_prose"
            ):
                continue
            decision = evaluate_replacement_decision(
                OcrLineCluster(
                    anchor_bbox=window.variant.bbox,
                    variants=(window.variant, variant),
                    selected=variant,
                    base_variant=window.variant,
                ),
                window.variant,
                variant,
            )
            if not decision.should_replace:
                continue
            if decision.confidence < 0.60:
                continue
            if best_decision is None or decision.confidence > best_decision.confidence:
                best_variant = variant
                best_decision = decision
        if best_variant is None or best_decision is None:
            continue
        variant_key = (best_variant.text, best_variant.source_name, best_variant.bbox)
        if variant_key in used_variant_keys:
            continue
        proposals.append((window, best_variant, best_decision))
        used_variant_keys.add(variant_key)
    proposals.sort(key=lambda item: item[2].confidence, reverse=True)
    return proposals


def base_merge_windows(
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    replaced_indexes: set[int],
) -> tuple[OcrBaseWindow, ...]:
    windows: list[OcrBaseWindow] = []
    for index in range(len(base_lines) - 1):
        if index in replaced_indexes or index + 1 in replaced_indexes:
            continue
        left = base_lines[index]
        right = base_lines[index + 1]
        if not merge_window_is_eligible(left, right):
            continue
        merged_variant = merged_variant_from_resolved_lines(left, right, index)
        if merged_variant is None:
            continue
        windows.append(
            OcrBaseWindow(
                start_index=index,
                end_index=index + 1,
                variant=merged_variant,
            )
        )
    return tuple(windows)


def base_merge_window_eligibility(
    base_lines: tuple[observation_resolver.ResolvedTextLine, ...],
    index: int,
    replaced_indexes: set[int],
) -> OcrBaseWindowEligibility:
    left = base_lines[index]
    right = base_lines[index + 1]
    left_variant = variant_from_resolved_line(left, index)
    right_variant = variant_from_resolved_line(right, index + 1)
    if index in replaced_indexes or index + 1 in replaced_indexes:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="already_replaced",
            gap=0.0,
            gap_ratio=0.0,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    if right.break_before > 1:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="break_before",
            gap=0.0,
            gap_ratio=0.0,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    if left.observation.bbox is None or right.observation.bbox is None:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="missing_bbox",
            gap=0.0,
            gap_ratio=0.0,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    gap = max(0.0, left.observation.bbox[1] - right.observation.bbox[3])
    left_height = max(1.0, left.observation.bbox[3] - left.observation.bbox[1])
    right_height = max(1.0, right.observation.bbox[3] - right.observation.bbox[1])
    gap_ratio = gap / max(left_height, right_height)
    if gap > max(left_height, right_height) * 1.25:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="gap_too_large",
            gap=gap,
            gap_ratio=gap_ratio,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    if left_variant.line_type in {"header_footer", "page_marker", "junk"}:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="left_line_type",
            gap=gap,
            gap_ratio=gap_ratio,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    if right_variant.line_type in {"header_footer", "page_marker", "junk"}:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="right_line_type",
            gap=gap,
            gap_ratio=gap_ratio,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    merged_variant = merged_variant_from_resolved_lines(left, right, index)
    if merged_variant is None:
        return OcrBaseWindowEligibility(
            start_index=index,
            end_index=index + 1,
            eligible=False,
            reason="merge_failed",
            gap=gap,
            gap_ratio=gap_ratio,
            left_line_type=left_variant.line_type,
            right_line_type=right_variant.line_type,
        )
    return OcrBaseWindowEligibility(
        start_index=index,
        end_index=index + 1,
        eligible=True,
        reason="eligible",
        gap=gap,
        gap_ratio=gap_ratio,
        left_line_type=left_variant.line_type,
        right_line_type=right_variant.line_type,
        merged_line_type=merged_variant.line_type,
        merged_bbox=merged_variant.bbox,
        merged_text=merged_variant.text,
    )


def merge_window_is_eligible(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
) -> bool:
    eligibility = base_merge_window_eligibility((left, right), 0, set())
    return eligibility.eligible


def merged_variant_from_resolved_lines(
    left: observation_resolver.ResolvedTextLine,
    right: observation_resolver.ResolvedTextLine,
    line_index: int,
) -> OcrLineVariant | None:
    left_variant = variant_from_resolved_line(left, line_index)
    right_variant = variant_from_resolved_line(right, line_index + 1)
    if left_variant.bbox is None or right_variant.bbox is None:
        return None
    bbox = page_geometry.observation_union_bbox(
        (left_variant.observation, right_variant.observation)
    )
    if bbox is None:
        return None
    text = f"{left.text.rstrip()} {right.text.lstrip()}".strip()
    if not text:
        return None
    stats = token_stats(text)
    family = left_variant.family
    line_type = classify_line_type(text, stats, family)
    confidence_values = [
        value
        for value in (left_variant.confidence, right_variant.confidence)
        if value is not None
    ]
    confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else None
    )
    observation = page_geometry.PageObservation(
        kind=left_variant.observation.kind,
        source=f"{left_variant.source_name}:merged_window",
        bbox=bbox,
        advance_bbox=bbox,
        ink_bbox=bbox,
        confidence=confidence,
        text=text,
        provenance=(
            *left_variant.observation.provenance,
            *right_variant.observation.provenance,
            *page_geometry.provenance_tuple(merged_window=True),
        ),
    )
    return OcrLineVariant(
        family=family,
        source_name=f"{left_variant.source_name}:merged_window",
        bbox=bbox,
        text=text,
        confidence=confidence,
        quality=ocr_text_analysis.text_ocr_quality_score(text),
        artifact=ocr_text_analysis.scanned_ocr_artifact_score(text),
        token_stats=stats,
        line_type=line_type,
        observation=observation,
        break_before=left.break_before,
        provenance=observation.provenance,
    )


def apply_merge_replacement_variants(
    output_lines: list[observation_resolver.ResolvedTextLine],
    proposals: list[tuple[OcrBaseWindow, OcrLineVariant, OcrLineReplacementDecision]],
) -> list[observation_resolver.ResolvedTextLine]:
    consumed_indexes: set[int] = set()
    merged_by_start: dict[
        int, tuple[OcrBaseWindow, OcrLineVariant, OcrLineReplacementDecision]
    ] = {}
    for window, variant, decision in proposals:
        if (
            window.start_index in consumed_indexes
            or window.end_index in consumed_indexes
        ):
            continue
        consumed_indexes.add(window.start_index)
        consumed_indexes.add(window.end_index)
        merged_by_start[window.start_index] = (window, variant, decision)
    if not merged_by_start:
        return output_lines
    merged_lines: list[observation_resolver.ResolvedTextLine] = []
    index = 0
    while index < len(output_lines):
        proposal = merged_by_start.get(index)
        if proposal is None:
            if index not in consumed_indexes:
                merged_lines.append(output_lines[index])
            index += 1
            continue
        window, variant, _decision = proposal
        merged_lines.append(
            resolved_line_from_variant(
                variant,
                original=output_lines[window.start_index],
            )
        )
        index = window.end_index + 1
    return merged_lines


def rejected_replacement(reason: str, **signals: float) -> OcrLineReplacementDecision:
    return OcrLineReplacementDecision(
        should_replace=False,
        confidence=signals.get("confidence", 0.0),
        reasons=(reason,),
        signals=tuple((name, value) for name, value in signals.items()),
    )


def replacement_threshold(line_type: OcrLineType) -> float:
    if line_type in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        return 0.54
    if line_type == "table_row":
        return 0.56
    return 0.58


def select_supplement_variants(
    clusters: list[OcrLineCluster],
    seen_tokens: set[str],
    geometry_context: DiagramGeometryContext,
) -> list[OcrLineVariant]:
    supplements: list[OcrLineVariant] = []
    for cluster in clusters:
        if cluster.base_variant is not None:
            continue
        winner, _ = choose_cluster_winner(cluster)
        if winner is None:
            continue
        if not should_insert_variant(winner, seen_tokens, geometry_context):
            continue
        supplements.append(winner)
    supplements.sort(key=variant_order_key)
    deduped: list[OcrLineVariant] = []
    used_keys: set[tuple[str, tuple[str, ...]]] = set()
    for variant in supplements:
        key = (variant.family, consensus_key(variant.text))
        if key in used_keys:
            continue
        deduped.append(variant)
        used_keys.add(key)
    return deduped


def should_insert_variant(
    variant: OcrLineVariant,
    seen_tokens: set[str],
    geometry_context: DiagramGeometryContext,
) -> bool:
    if variant.line_type in {"header_footer", "page_marker", "junk"}:
        return False
    if variant.family == "broad_page":
        return False
    if variant.family == "vector_stroke":
        return False
    if variant.family == "table" and variant.line_type != "table_row":
        return False
    if variant.family == "figure":
        return figure_variant_should_insert(variant, seen_tokens, geometry_context)
    if variant.line_type == "body_prose":
        return False
    if variant.family == "embedded_image":
        confidence = (
            int(variant.confidence)
            if isinstance(variant.confidence, int | float)
            else None
        )
        return ocr_postprocess.embedded_image_text_line_should_append(
            variant.text,
            seen_tokens,
            confidence=confidence,
        )
    tokens = ocr_text_analysis.normalized_text_tokens(variant.text)
    new_tokens = sum(1 for token in tokens if token not in seen_tokens)
    if variant.line_type in {"chemical_symbolic", "chemical_hybrid"}:
        return new_tokens >= 1
    return new_tokens >= 2


def figure_variant_should_insert(
    variant: OcrLineVariant,
    seen_tokens: set[str],
    geometry_context: DiagramGeometryContext,
) -> bool:
    confidence = (
        float(variant.confidence)
        if isinstance(variant.confidence, int | float)
        else 0.0
    )
    if confidence < 80.0 and not figure_variant_qualifies_for_contextual_insert(
        variant,
        geometry_context,
    ):
        return False
    text = variant.text.strip()
    if not text or not text[0].isalnum():
        return False
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if len(tokens) > 4 or len(text) > 48:
        return False
    if figure_variant_looks_like_page_metadata(tokens):
        return False
    if not ocr_postprocess.figure_ocr_line_has_diagram_signal(text, tokens):
        return False
    if figure_variant_is_contextual_compact_label(
        variant,
        seen_tokens,
        geometry_context,
    ):
        return True
    return ocr_postprocess.figure_ocr_line_should_append(text, seen_tokens)


def figure_variant_qualifies_for_contextual_insert(
    variant: OcrLineVariant,
    geometry_context: DiagramGeometryContext,
) -> bool:
    if not geometry_context.diagram_heavy:
        return False
    if not figure_variant_contributes_geometry_context(variant):
        return False
    return diagram_region_membership_score(variant, geometry_context) >= 0.55


def figure_variant_is_contextual_compact_label(
    variant: OcrLineVariant,
    seen_tokens: set[str],
    geometry_context: DiagramGeometryContext,
) -> bool:
    if not figure_variant_qualifies_for_contextual_insert(variant, geometry_context):
        return False
    text = variant.text.strip()
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if not tokens or len(tokens) > 5:
        return False
    if variant.token_stats.ocr_confusion_count > 0 and not any(
        ch.isdigit() for ch in text
    ):
        return False
    if "vs" not in tokens and not any(ch.isdigit() for ch in text):
        return False
    if not any(
        figure_variant_token_is_informative_and_new(raw_token, seen_tokens)
        for raw_token in text.split()
    ):
        return False
    return readable_content_token_count(text) >= 1 or any(ch.isdigit() for ch in text)


def figure_variant_token_is_informative_and_new(
    raw_token: str,
    seen_tokens: set[str],
) -> bool:
    token = raw_token.strip().strip(".,;:()[]{}")
    normalized = token.casefold()
    if not normalized or normalized in seen_tokens:
        return False
    if normalized.isdigit():
        return True
    if token.isalpha():
        if len(token) >= 3:
            return True
        return token.isupper() and len(token) >= 2
    if (
        len(token) <= 5
        and not any(ch.isdigit() for ch in token)
        and any(not ch.isalnum() for ch in token)
        and any(ch.islower() for ch in token)
        and any(ch.isupper() for ch in token)
    ):
        return False
    if len(token) >= 4 and any(ch.isalpha() for ch in token):
        return True
    return any(ch in token for ch in "_/+-")


def figure_variant_looks_like_page_metadata(tokens: list[str]) -> bool:
    token_set = set(tokens)
    if {"patent", "sheet"} & token_set:
        return True
    if "oct" in token_set and any(token.isdigit() for token in tokens):
        return True
    if "us" in token_set and any(token.isdigit() for token in tokens):
        return True
    return False


def suppress_geometry_noise_lines(
    output_lines: list[observation_resolver.ResolvedTextLine],
    context: DiagramGeometryContext,
) -> list[observation_resolver.ResolvedTextLine]:
    if not context.diagram_heavy or not context.figure_observations:
        return output_lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    variants = [
        variant_from_resolved_line(line, index)
        for index, line in enumerate(output_lines)
    ]
    for index, (line, variant) in enumerate(zip(output_lines, variants, strict=True)):
        suppression = geometry_suppression_reason(
            variants,
            index,
            variant,
            context,
        )
        if suppression is None:
            kept.append(line)
            continue
    return kept


def apply_figure_source_priority_replacements(
    output_lines: list[observation_resolver.ResolvedTextLine],
    figure_result: OcrPageTextResult | None,
    context: DiagramGeometryContext,
) -> list[observation_resolver.ResolvedTextLine]:
    figure_candidates = prioritized_figure_source_variants(figure_result, context)
    if not context.diagram_heavy or not figure_candidates:
        return output_lines
    updated = list(output_lines)
    for index, line in enumerate(output_lines):
        base = variant_from_resolved_line(line, index)
        replacement, signals = best_figure_source_priority_replacement(
            base,
            figure_candidates,
            context,
        )
        if replacement is None or signals is None:
            continue
        updated[index] = resolved_line_from_variant(
            replacement,
            original=line,
            contributing_observations=(base.observation, replacement.observation),
        )
    return updated


def upgrade_broad_page_fragments_with_figure_lines(
    output_lines: list[observation_resolver.ResolvedTextLine],
    figure_result: OcrPageTextResult | None,
    context: DiagramGeometryContext,
) -> list[observation_resolver.ResolvedTextLine]:
    figure_candidates = figure_fragment_upgrade_variants(figure_result, context)
    if not figure_candidates:
        return output_lines
    updated = list(output_lines)
    for index, line in enumerate(output_lines):
        base = variant_from_resolved_line(line, index)
        if base.family != "broad_page" or base.bbox is None:
            continue
        replacement, signals = best_figure_fragment_upgrade(
            base, figure_candidates, context
        )
        if replacement is None or signals is None:
            continue
        updated[index] = resolved_line_from_variant(
            replacement,
            original=line,
            contributing_observations=(base.observation, replacement.observation),
        )
    return updated


def figure_fragment_upgrade_variants(
    result: OcrPageTextResult | None,
    context: DiagramGeometryContext,
) -> list[OcrLineVariant]:
    selected: list[OcrLineVariant] = []
    seen: set[tuple[tuple[str, ...], tuple[float, float, float, float] | None]] = set()
    for variant in figure_variants(result):
        if variant.bbox is None:
            continue
        if figure_variant_looks_like_page_metadata(
            ocr_text_analysis.normalized_text_tokens(variant.text)
        ):
            continue
        if diagram_region_membership_score(variant, context) < 0.10:
            continue
        if variant.token_stats.token_count > 6:
            continue
        if readable_content_token_count(variant.text) < 1:
            continue
        key = (consensus_key(variant.text), variant.bbox)
        if key in seen:
            continue
        seen.add(key)
        selected.append(variant)
    selected.sort(key=variant_order_key)
    return selected


def best_figure_fragment_upgrade(
    base: OcrLineVariant,
    figure_candidates: list[OcrLineVariant],
    context: DiagramGeometryContext,
) -> tuple[OcrLineVariant | None, dict[str, float] | None]:
    if diagram_region_membership_score(base, context) < 0.42:
        return (None, None)
    base_tokens = ocr_text_analysis.normalized_text_tokens(base.text)
    base_readable = readable_content_token_count(base.text)
    best_variant: OcrLineVariant | None = None
    best_signals: dict[str, float] | None = None
    best_score = 0.0
    for candidate in figure_candidates:
        signals = figure_source_priority_signals(base, candidate)
        if signals["row_alignment"] < 0.56:
            continue
        if max(signals["geometry"], signals["overlap"]) < 0.10:
            continue
        anchor_overlap = signals["anchor_overlap"]
        if anchor_overlap < 0.16:
            continue
        candidate_tokens = ocr_text_analysis.normalized_text_tokens(candidate.text)
        if len(candidate_tokens) <= len(base_tokens):
            continue
        candidate_readable = readable_content_token_count(candidate.text)
        if candidate_readable < base_readable:
            continue
        score = (
            anchor_overlap * 0.34
            + signals["row_alignment"] * 0.22
            + signals["geometry"] * 0.18
            + signals["overlap"] * 0.14
            + min(1.0, (len(candidate_tokens) - len(base_tokens)) / 3.0) * 0.12
        )
        if score > best_score:
            best_score = score
            best_variant = candidate
            best_signals = {
                **signals,
                "token_gain": float(len(candidate_tokens) - len(base_tokens)),
                "score": score,
            }
    if best_variant is None or best_signals is None:
        return (None, None)
    return (best_variant, best_signals)


def prioritized_figure_source_variants(
    result: OcrPageTextResult | None,
    context: DiagramGeometryContext,
) -> list[OcrLineVariant]:
    selected: list[OcrLineVariant] = []
    seen: set[tuple[tuple[str, ...], tuple[float, float, float, float] | None]] = set()
    for variant in figure_variants(result):
        if not figure_variant_is_source_priority_candidate(variant, context):
            continue
        key = (consensus_key(variant.text), variant.bbox)
        if key in seen:
            continue
        seen.add(key)
        selected.append(variant)
    selected.sort(key=variant_order_key)
    return selected


def figure_variant_is_source_priority_candidate(
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> bool:
    if variant.bbox is None:
        return False
    if figure_variant_looks_like_page_metadata(
        ocr_text_analysis.normalized_text_tokens(variant.text)
    ):
        return False
    if figure_variant_is_fragment_like(variant):
        return False
    if diagram_region_membership_score(variant, context) < 0.10:
        return False
    return readable_content_token_count(variant.text) >= 1 or any(
        token.isdigit()
        for token in ocr_text_analysis.normalized_text_tokens(variant.text)
    )


def best_figure_source_priority_replacement(
    base: OcrLineVariant,
    figure_candidates: list[OcrLineVariant],
    context: DiagramGeometryContext,
) -> tuple[OcrLineVariant | None, dict[str, float] | None]:
    if base.family != "broad_page" or base.bbox is None:
        return (None, None)
    if base.line_type in {"header_footer", "page_marker", "table_row"}:
        return (None, None)
    region_score = diagram_region_membership_score(base, context)
    if region_score < 0.42:
        return (None, None)
    base_readable = readable_content_token_count(base.text)
    base_weak = broad_page_variant_is_geom_weak(base, readable_tokens=base_readable)
    best_variant: OcrLineVariant | None = None
    best_signals: dict[str, float] | None = None
    best_score = 0.0
    for candidate in figure_candidates:
        signals = figure_source_priority_signals(base, candidate)
        if not figure_source_priority_is_eligible(
            base,
            candidate,
            signals,
            base_readable=base_readable,
            base_weak=base_weak,
        ):
            continue
        score = (
            signals["geometry"] * 0.40
            + signals["overlap"] * 0.28
            + signals["row_alignment"] * 0.16
            + signals["readable_gain"] * 0.16
        )
        if score > best_score:
            best_score = score
            best_variant = candidate
            best_signals = {**signals, "score": score, "region_score": region_score}
    if best_variant is None or best_signals is None:
        return (None, None)
    return (best_variant, best_signals)


def figure_source_priority_signals(
    base: OcrLineVariant,
    candidate: OcrLineVariant,
) -> dict[str, float]:
    geometry = page_geometry.observation_geometry_match_score(
        base.observation,
        candidate.observation,
    )
    overlap = page_geometry.observation_overlap_ratio(
        base.observation,
        candidate.observation,
        denominator="smaller",
    )
    left_y = page_geometry.observation_mid_y(base.observation)
    right_y = page_geometry.observation_mid_y(candidate.observation)
    height = max(
        1.0,
        page_geometry.observation_height(base.observation),
        page_geometry.observation_height(candidate.observation),
    )
    row_alignment = max(0.0, 1.0 - abs(left_y - right_y) / (height * 1.2))
    readable_gain = max(
        0.0,
        float(
            readable_content_token_count(candidate.text)
            - readable_content_token_count(base.text)
        ),
    )
    anchor_overlap = anchor_token_overlap_score(base, candidate)
    return {
        "geometry": geometry,
        "overlap": overlap,
        "row_alignment": row_alignment,
        "readable_gain": min(1.0, readable_gain / 2.0),
        "anchor_overlap": anchor_overlap,
    }


def figure_source_priority_is_eligible(
    base: OcrLineVariant,
    candidate: OcrLineVariant,
    signals: dict[str, float],
    *,
    base_readable: int,
    base_weak: bool,
) -> bool:
    if signals["geometry"] < 0.18 and signals["overlap"] < 0.12:
        return False
    if signals["row_alignment"] < 0.58:
        return False
    candidate_readable = readable_content_token_count(candidate.text)
    candidate_tokens = ocr_text_analysis.normalized_text_tokens(candidate.text)
    base_tokens = ocr_text_analysis.normalized_text_tokens(base.text)
    candidate_has_digit = any(token.isdigit() for token in candidate_tokens)
    base_has_digit = any(token.isdigit() for token in base_tokens)
    if base_readable == 0 and candidate_readable >= 1:
        return signals["geometry"] >= 0.22 or signals["overlap"] >= 0.18
    if (
        base_readable <= 1
        and candidate_readable >= 1
        and candidate_has_digit
        and not base_has_digit
        and signals["anchor_overlap"] >= 0.16
        and candidate.token_stats.token_count <= base.token_stats.token_count + 3
    ):
        return signals["geometry"] >= 0.14 or signals["overlap"] >= 0.12
    if (
        base.token_stats.token_count <= 2
        and candidate.token_stats.token_count >= base.token_stats.token_count + 2
        and candidate_readable >= base_readable + 1
        and candidate_has_digit
        and signals["anchor_overlap"] >= 0.24
        and signals["row_alignment"] >= 0.70
    ):
        return signals["geometry"] >= 0.12 or signals["overlap"] >= 0.10
    if not base_weak:
        return False
    return candidate_readable >= base_readable + 1 and (
        signals["geometry"] >= 0.24 or signals["overlap"] >= 0.20
    )


def geometry_suppression_reason(
    variants: list[OcrLineVariant],
    index: int,
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> dict[str, Any] | None:
    if variant.family != "broad_page" or variant.bbox is None:
        return None
    if variant.line_type in {"table_row", "header_footer", "page_marker"}:
        return None
    max_overlap = max_figure_overlap_ratio(variant, context.figure_observations)
    covered = page_geometry.observation_is_covered_by(
        variant.observation,
        context.figure_observations,
        single_overlap_ratio=0.42,
        cumulative_overlap_ratio=0.55,
    )
    continuity = broad_page_line_continuity_score(variants, index)
    readable = readable_content_token_count(variant.text)
    cluster_noise = variant_is_dense_label_cluster_noise(variant, context)
    weak = broad_page_variant_is_geom_weak(variant, readable_tokens=readable)
    region_score = diagram_region_membership_score(variant, context)
    if covered and weak:
        return {
            "reason": "covered_by_figure_region",
            "signals": {
                "max_overlap": max_overlap,
                "continuity": continuity,
                "readable_tokens": float(readable),
                "region_score": region_score,
            },
        }
    if region_score >= 0.55 and cluster_noise and continuity <= 0.18:
        return {
            "reason": "dense_label_cluster_noise",
            "signals": {
                "max_overlap": max_overlap,
                "continuity": continuity,
                "readable_tokens": float(readable),
                "region_score": region_score,
            },
        }
    if region_score >= 0.78 and weak and continuity <= 0.22:
        return {
            "reason": "low_continuity_diagram_region",
            "signals": {
                "max_overlap": max_overlap,
                "continuity": continuity,
                "readable_tokens": float(readable),
                "region_score": region_score,
            },
        }
    return None


def max_figure_overlap_ratio(
    variant: OcrLineVariant,
    figure_observations: tuple[page_geometry.PageObservation, ...],
) -> float:
    return max(
        (
            page_geometry.observation_overlap_ratio(
                variant.observation,
                observation,
                denominator="smaller",
            )
            for observation in figure_observations
        ),
        default=0.0,
    )


def broad_page_line_continuity_score(
    variants: list[OcrLineVariant],
    index: int,
) -> float:
    current = variants[index]
    if current.bbox is None:
        return 0.0
    scores: list[float] = []
    for direction in (-1, 1):
        neighbor_index = index + direction
        while 0 <= neighbor_index < len(variants):
            neighbor = variants[neighbor_index]
            neighbor_index += direction
            if neighbor.family != current.family or neighbor.bbox is None:
                continue
            scores.append(adjacent_line_continuity(current, neighbor))
            break
    return max(scores, default=0.0)


def adjacent_line_continuity(
    current: OcrLineVariant,
    neighbor: OcrLineVariant,
) -> float:
    if current.bbox is None or neighbor.bbox is None:
        return 0.0
    current_width = max(1.0, current.bbox[2] - current.bbox[0])
    neighbor_width = max(1.0, neighbor.bbox[2] - neighbor.bbox[0])
    current_height = max(1.0, current.bbox[3] - current.bbox[1])
    neighbor_height = max(1.0, neighbor.bbox[3] - neighbor.bbox[1])
    x_overlap = max(
        0.0,
        min(current.bbox[2], neighbor.bbox[2]) - max(current.bbox[0], neighbor.bbox[0]),
    )
    x_overlap_ratio = x_overlap / min(current_width, neighbor_width)
    center_gap = abs(
        page_geometry.observation_mid_x(current.observation)
        - page_geometry.observation_mid_x(neighbor.observation)
    )
    center_alignment = max(0.0, 1.0 - center_gap / max(current_width, neighbor_width))
    vertical_gap = max(
        0.0,
        min(
            abs(current.bbox[1] - neighbor.bbox[3]),
            abs(neighbor.bbox[1] - current.bbox[3]),
        ),
    )
    gap_score = max(
        0.0, 1.0 - vertical_gap / max(current_height, neighbor_height) / 2.5
    )
    return min(1.0, x_overlap_ratio * 0.45 + center_alignment * 0.30 + gap_score * 0.25)


def readable_content_token_count(text: str) -> int:
    count = 0
    for token in ocr_text_analysis.normalized_text_tokens(text):
        if token.isdigit():
            count += 1
            continue
        if (
            len(token) >= 4
            and token.isalpha()
            and not ocr_text_analysis.alpha_token_looks_ocr_garbled(token)
            and (rank := word_rank(token)) is not None
            and rank <= 150_000
        ):
            count += 1
    return count


def broad_page_variant_is_geom_weak(
    variant: OcrLineVariant,
    *,
    readable_tokens: int,
) -> bool:
    if readable_tokens >= 2:
        return False
    stats = variant.token_stats
    if stats.token_count <= 2 and readable_tokens == 0:
        return True
    if stats.short_token_ratio >= 0.60:
        return True
    if stats.punctuation_ratio >= 0.18:
        return True
    if stats.ocr_confusion_count >= 2:
        return True
    if ocr_text_analysis.alphabetic_gibberish_line_score(variant.text) >= 0.40:
        return True
    return False


def variant_is_dense_label_cluster_noise(
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> bool:
    if variant.bbox is None:
        return False
    if context.figure_cluster_count < 2 and context.short_figure_line_count < 3:
        return False
    if variant.token_stats.token_count > 4:
        return False
    return readable_content_token_count(variant.text) == 0


def diagram_region_membership_score(
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> float:
    return diagram_region_membership_score_for_bbox(variant.bbox, context.region_bbox)


def diagram_region_membership_score_for_bbox(
    bbox: tuple[float, float, float, float] | None,
    region_bbox: tuple[float, float, float, float] | None,
) -> float:
    if bbox is None or region_bbox is None:
        return 0.0
    region_observation = page_geometry.PageObservation(
        kind="diagram_region",
        source="figure_region_mask",
        bbox=region_bbox,
        advance_bbox=region_bbox,
        ink_bbox=region_bbox,
    )
    return page_geometry.observation_overlap_ratio(
        page_geometry.PageObservation(
            kind="diagram_line",
            source="figure_region_mask",
            bbox=bbox,
            advance_bbox=bbox,
            ink_bbox=bbox,
        ),
        region_observation,
        denominator="left",
    )


def compose_dominant_figure_page_lines(
    output_lines: list[observation_resolver.ResolvedTextLine],
    sources: OcrLineReconciliationSources,
    context: DiagramGeometryContext,
) -> list[observation_resolver.ResolvedTextLine]:
    if not dominant_figure_page_context(context):
        return output_lines
    figure_candidates = dominant_figure_region_variants(sources.figure_result, context)
    if len(figure_candidates) < 5:
        return output_lines
    kept: list[observation_resolver.ResolvedTextLine] = []
    for index, line in enumerate(output_lines):
        variant = variant_from_resolved_line(line, index)
        if not dominant_figure_region_line_should_drop(variant, context):
            kept.append(line)
            continue
    merged = list(kept)
    seen_keys = {consensus_key(line.text) for line in merged if line.text.strip()}
    for variant in figure_candidates:
        key = consensus_key(variant.text)
        if key in seen_keys:
            continue
        line = resolved_line_from_variant(variant)
        insert_at = supplement_insert_index(merged, variant)
        if insert_at >= len(merged):
            merged.append(line)
        else:
            merged.insert(insert_at, line)
        seen_keys.add(key)
    return merged


def dominant_figure_page_context(context: DiagramGeometryContext) -> bool:
    if not context.diagram_heavy or context.region_bbox is None:
        return False
    if context.broad_page_line_count < 6 or context.broad_page_region_count < 4:
        return False
    if context.broad_page_region_ratio < 0.62:
        return False
    descriptive_count = max(
        0, len(context.figure_variants) - context.figure_fragment_count
    )
    if descriptive_count < 5:
        return False
    if context.figure_fragment_count > max(
        4,
        int(len(context.figure_variants) * 0.34),
    ):
        return False
    return context.short_figure_line_count <= max(
        6,
        int(len(context.figure_variants) * 0.55),
    )


def dominant_figure_region_variants(
    result: OcrPageTextResult | None,
    context: DiagramGeometryContext,
) -> list[OcrLineVariant]:
    selected: list[OcrLineVariant] = []
    seen: set[tuple[tuple[str, ...], tuple[float, float, float, float] | None]] = set()
    for variant in figure_variants(result):
        if not figure_variant_should_compose(variant, context):
            continue
        key = (consensus_key(variant.text), variant.bbox)
        if key in seen:
            continue
        seen.add(key)
        selected.append(variant)
    selected.sort(key=variant_order_key)
    return selected


def figure_variant_should_compose(
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> bool:
    confidence = (
        float(variant.confidence)
        if isinstance(variant.confidence, int | float)
        else 0.0
    )
    if confidence < 55.0 or variant.bbox is None:
        return False
    text = variant.text.strip()
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if not text or figure_variant_looks_like_page_metadata(tokens):
        return False
    if diagram_region_membership_score(variant, context) < 0.12:
        return False
    if figure_variant_is_fragment_like(variant):
        return False
    if text.startswith("FIG"):
        return False
    if len(tokens) == 1:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if variant.token_stats.token_count > 8:
        return False
    if figure_text_like_noise(variant):
        return False
    return (
        variant.line_type in {"diagram_label", "chemical_symbolic", "chemical_hybrid"}
        or readable_content_token_count(text) >= 1
        or any(token.isdigit() for token in tokens)
    )


def figure_text_like_noise(variant: OcrLineVariant) -> bool:
    if variant.token_stats.ocr_confusion_count >= 3:
        return True
    if variant.token_stats.punctuation_ratio >= 0.26:
        return True
    if (
        variant.token_stats.token_count <= 3
        and readable_content_token_count(variant.text) == 0
        and not any(ch.isdigit() for ch in variant.text)
    ):
        return True
    return ocr_text_analysis.alphabetic_gibberish_line_score(variant.text) >= 0.45


def figure_variant_is_fragment_like(variant: OcrLineVariant) -> bool:
    text = variant.text.strip()
    if not text:
        return True
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    if not tokens:
        return True
    if figure_variant_looks_like_page_metadata(tokens):
        return True
    if all(token.isdigit() for token in tokens):
        return True
    if len(tokens) == 1:
        token = tokens[0]
        if token.isdigit():
            return True
        if token.isalpha() and len(token) < 5:
            return True
    readable_tokens = readable_content_token_count(text)
    if readable_tokens == 0 and len(tokens) <= 2:
        return True
    if figure_text_like_noise(variant):
        return True
    return False


def dominant_figure_region_line_should_drop(
    variant: OcrLineVariant,
    context: DiagramGeometryContext,
) -> bool:
    if variant.family != "broad_page" or variant.bbox is None:
        return False
    if variant.line_type in {"header_footer", "page_marker"}:
        return False
    region_score = diagram_region_membership_score(variant, context)
    if region_score < 0.42:
        return False
    return True


def insert_supplement_variants(
    output_lines: list[observation_resolver.ResolvedTextLine],
    supplements: list[OcrLineVariant],
) -> list[observation_resolver.ResolvedTextLine]:
    merged = list(output_lines)
    for variant in supplements:
        line = resolved_line_from_variant(variant)
        insert_at = supplement_insert_index(merged, variant)
        if insert_at >= len(merged):
            merged.append(line)
        else:
            merged.insert(insert_at, line)
    return merged


def supplement_insert_index(
    lines: list[observation_resolver.ResolvedTextLine],
    variant: OcrLineVariant,
) -> int:
    if variant.bbox is None:
        return len(lines)
    order = variant_order_key(variant)
    for index, line in enumerate(lines):
        observation = line.observation
        if observation.bbox is None:
            continue
        existing = variant_order_key(
            OcrLineVariant(
                family=source_family_from_name(observation.source),
                source_name=observation.source,
                bbox=observation.bbox,
                text=line.text,
                confidence=observation.confidence,
                quality=ocr_text_analysis.text_ocr_quality_score(line.text),
                artifact=ocr_text_analysis.scanned_ocr_artifact_score(line.text),
                token_stats=token_stats(line.text),
                line_type=classify_line_type(
                    line.text,
                    token_stats(line.text),
                    source_family_from_name(observation.source),
                ),
                observation=observation,
                break_before=line.break_before,
                provenance=observation.provenance,
            )
        )
        if order < existing:
            return index
    return len(lines)


def resolved_line_from_variant(
    variant: OcrLineVariant,
    *,
    original: observation_resolver.ResolvedTextLine | None = None,
    contributing_observations: tuple[page_geometry.PageObservation, ...] | None = None,
) -> observation_resolver.ResolvedTextLine:
    break_before = (
        original.break_before if original is not None else variant.break_before
    )
    return observation_resolver.ResolvedTextLine(
        variant.text,
        variant.observation,
        break_before=break_before,
        contributing_observations=contributing_observations
        if contributing_observations is not None
        else (variant.observation,),
    )


def resolved_line_with_cluster_support(
    line: observation_resolver.ResolvedTextLine,
    cluster: OcrLineCluster,
    selected: OcrLineVariant,
) -> observation_resolver.ResolvedTextLine:
    contributing_observations = cluster_support_observations(cluster, selected)
    if not contributing_observations:
        return line
    return observation_resolver.ResolvedTextLine(
        line.text,
        line.observation,
        break_before=line.break_before,
        contributing_observations=contributing_observations,
        resolution=line.resolution,
    )


def cluster_support_observations(
    cluster: OcrLineCluster,
    selected: OcrLineVariant,
) -> tuple[page_geometry.PageObservation, ...]:
    selected_key = consensus_key(selected.text)
    observations: list[page_geometry.PageObservation] = [selected.observation]
    seen = {observation_support_key(selected.observation)}
    for variant in cluster.variants:
        if consensus_key(variant.text) != selected_key:
            continue
        key = observation_support_key(variant.observation)
        if key in seen:
            continue
        seen.add(key)
        observations.append(variant.observation)
    return tuple(observations)


def observation_support_key(
    observation: page_geometry.PageObservation,
) -> tuple[str, str, page_geometry.Rect | None, str]:
    return (
        observation.kind,
        observation.source,
        observation.bbox,
        observation.text,
    )


def source_family_from_name(source_name: str) -> OcrLineSourceFamily:
    normalized = ocr_selection.ocr_variant_source_name(source_name)
    if source_name == "vector_stroke":
        return "vector_stroke"
    if source_name == "embedded_image_text":
        return "embedded_image"
    if source_name == "figure_ocr_regions" or source_name.startswith("figure_"):
        return "figure"
    if source_name in ocr_table_regions.OCR_TABLE_CANDIDATE_NAMES:
        return "table"
    if source_name == "table_fusion_text":
        return "table"
    if ocr_selection.broad_page_candidate_name(normalized):
        return "broad_page"
    return "native"


def token_stats(text: str) -> OcrLineTokenStats:
    tokens = ocr_text_analysis.normalized_text_tokens(text)
    uppercase_chars = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = (
        sum(1 for ch in uppercase_chars if ch.isupper()) / len(uppercase_chars)
        if uppercase_chars
        else 0.0
    )
    return OcrLineTokenStats(
        token_count=len(tokens),
        short_token_ratio=ocr_text_analysis.ocr_line_short_token_ratio(text),
        punctuation_ratio=ocr_text_analysis.ocr_line_punctuation_ratio(text),
        numeric_ratio=ocr_text_analysis.numeric_token_ratio(text),
        uppercase_ratio=uppercase_ratio,
        chemical_signal_count=ocr_text_analysis.chemical_signal_count(text),
        ocr_confusion_count=ocr_text_analysis.ocr_confusion_char_count(text),
    )


def classify_line_type(
    text: str,
    stats: OcrLineTokenStats,
    family: OcrLineSourceFamily,
) -> OcrLineType:
    stripped = text.strip()
    if not stripped:
        return "junk"
    if stats.token_count <= 2 and stripped.replace(",", "").replace(".", "").isdigit():
        return "page_marker"
    if family == "embedded_image":
        return "diagram_label"
    if family == "table":
        return "table_row"
    tokens = ocr_text_analysis.normalized_text_tokens(stripped)
    if (
        stats.chemical_signal_count >= 3
        or ocr_text_analysis.line_has_readable_technical_notation(
            stripped,
            tokens,
        )
    ):
        if stats.short_token_ratio >= 0.40 or stats.punctuation_ratio >= 0.16:
            return "chemical_symbolic"
        return "chemical_hybrid"
    if compact_diagram_label_line(stripped, tokens, stats, family):
        return "diagram_label"
    if stats.short_token_ratio >= 0.80 and stats.token_count >= 6:
        return "junk"
    return "body_prose"


def compact_diagram_label_line(
    text: str,
    tokens: list[str],
    stats: OcrLineTokenStats,
    family: OcrLineSourceFamily,
) -> bool:
    if stats.token_count == 0 or stats.token_count > 8:
        return False
    if family in {"figure", "vector_stroke"}:
        return stats.uppercase_ratio >= 0.45
    if family != "broad_page":
        return False
    if len(text) > 64:
        return False
    if stats.uppercase_ratio < 0.65:
        return False
    if stats.punctuation_ratio > 0.18:
        return False
    if stats.ocr_confusion_count > 2:
        return False
    if ocr_text_analysis.alphabetic_gibberish_line_score(text) >= 0.50:
        return False
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return False
    readable_alpha = sum(
        1 for token in alpha_tokens if compact_label_token_is_readable(token)
    )
    has_long_alpha = any(len(token) >= 8 for token in alpha_tokens)
    has_numeric = any(any(ch.isdigit() for ch in token) for token in tokens)
    if has_numeric:
        return readable_alpha >= 1 or has_long_alpha
    if stats.token_count <= 4:
        return readable_alpha >= 1 or has_long_alpha
    return readable_alpha >= 2


def compact_label_token_is_readable(token: str) -> bool:
    if len(token) < 4 or not token.isalpha():
        return False
    rank = word_rank(token.casefold())
    return rank is not None and rank <= 150_000


def line_types_compatible(left: OcrLineType, right: OcrLineType) -> bool:
    if left == right:
        return True
    chemical_types = {"chemical_symbolic", "chemical_hybrid", "diagram_label"}
    if left in chemical_types and right in chemical_types:
        return True
    if {left, right} <= {"body_prose", "table_row"}:
        return True
    return False


def cluster_line_types_compatible(
    base: OcrLineVariant,
    variant: OcrLineVariant,
) -> bool:
    if line_types_compatible(base.line_type, variant.line_type):
        return True
    return figure_label_may_replace_weak_broad_page_line(base, variant)


def replacement_line_types_compatible(
    base: OcrLineVariant,
    selected: OcrLineVariant,
) -> bool:
    if line_types_compatible(base.line_type, selected.line_type):
        return True
    return figure_label_may_replace_weak_broad_page_line(base, selected)


def figure_label_may_replace_weak_broad_page_line(
    base: OcrLineVariant,
    variant: OcrLineVariant,
) -> bool:
    if base.family != "broad_page" or variant.family != "figure":
        return False
    if base.line_type != "body_prose" or variant.line_type != "diagram_label":
        return False
    if base.bbox is None or variant.bbox is None:
        return False
    return broad_page_variant_is_geom_weak(
        base,
        readable_tokens=readable_content_token_count(base.text),
    )


def chemical_plausibility_score(variant: OcrLineVariant) -> float:
    return chemical_plausibility_score_for_text(
        variant.text,
        variant.token_stats,
        variant.line_type,
    )


def chemical_plausibility_score_for_text(
    text: str,
    stats: OcrLineTokenStats,
    line_type: OcrLineType,
) -> float:
    if line_type not in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        return 0.0
    score = float(stats.chemical_signal_count)
    score += max(0.0, 1.0 - stats.short_token_ratio)
    score += max(0.0, 0.35 - stats.punctuation_ratio) * 4.0
    score -= stats.ocr_confusion_count * 0.15
    if "wherein" in text.casefold():
        score += 0.5
    return score


@lru_cache(maxsize=32_768)
def consensus_key(text: str) -> tuple[str, ...]:
    canonical = ocr_text_analysis.canonicalized_ocr_consensus_tokens(text)
    if canonical:
        return canonical
    return tuple(ocr_text_analysis.normalized_text_tokens(text))


@lru_cache(maxsize=32_768)
def anchor_tokens(text: str, line_type: OcrLineType) -> tuple[str, ...]:
    tokens = consensus_key(text)
    if not tokens:
        return ()
    if line_type == "table_row":
        return tuple(token for token in tokens if len(token) >= 2)[:8]
    if line_type in {"chemical_symbolic", "chemical_hybrid", "diagram_label"}:
        return tuple(token for token in tokens if len(token) >= 1)[:8]
    informative = [token for token in tokens if len(token) >= 3]
    if not informative:
        informative = list(tokens[:4])
    if len(informative) <= 4:
        return tuple(informative)
    return tuple((*informative[:2], *informative[-2:]))


def anchor_token_overlap_score(base: OcrLineVariant, variant: OcrLineVariant) -> float:
    left = anchor_tokens(base.text, base.line_type)
    right = anchor_tokens(variant.text, variant.line_type)
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts)
    denominator = max(len(left), len(right))
    coverage = overlap / denominator
    if overlap == 0:
        return 0.0
    return min(1.0, coverage + min(overlap, 3) * 0.08)


@lru_cache(maxsize=32_768)
def token_shape_signature(text: str) -> str:
    chunks: list[str] = []
    for token in consensus_key(text):
        shape = []
        previous = ""
        for ch in token:
            if ch.isupper():
                current = "A"
            elif ch.islower():
                current = "a"
            elif ch.isdigit():
                current = "9"
            else:
                current = "-"
            if current != previous:
                shape.append(current)
                previous = current
        if shape:
            chunks.append("".join(shape))
    return " ".join(chunks)


@lru_cache(maxsize=65_536)
def _cached_token_shape_agreement_score(left_shape: str, right_shape: str) -> float:
    return SequenceMatcher(a=left_shape, b=right_shape).ratio()


def token_shape_agreement_score(left_text: str, right_text: str) -> float:
    left = token_shape_signature(left_text)
    right = token_shape_signature(right_text)
    if not left or not right:
        return 0.0
    if left <= right:
        return _cached_token_shape_agreement_score(left, right)
    return _cached_token_shape_agreement_score(right, left)


def token_count_similarity_score(
    base: OcrLineVariant, variant: OcrLineVariant
) -> float:
    base_count = max(1, base.token_stats.token_count)
    variant_count = max(1, variant.token_stats.token_count)
    return max(
        0.0, 1.0 - abs(base_count - variant_count) / max(base_count, variant_count)
    )


def cluster_support_score(cluster: OcrLineCluster, selected: OcrLineVariant) -> float:
    selected_key = consensus_key(selected.text)
    supporters = [
        variant
        for variant in cluster.variants
        if variant is not selected and consensus_key(variant.text) == selected_key
    ]
    if not supporters:
        return 0.0
    family_count = len({variant.family for variant in supporters} | {selected.family})
    source_count = len(
        {variant.source_name for variant in supporters} | {selected.source_name}
    )
    return min(1.0, family_count * 0.22 + source_count * 0.16)


def variant_order_key(variant: OcrLineVariant) -> tuple[float, float]:
    bbox = variant.bbox
    if bbox is None:
        return (float("inf"), float("inf"))
    return (-bbox[3], bbox[0])


def preview_text(text: str) -> str:
    return text[:160]
