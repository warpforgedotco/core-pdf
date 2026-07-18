# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass, replace

from core_ocr.impl import page_geometry
from core_ocr.impl.text_analysis import normalized_text_tokens


@dataclass(frozen=True)
class ObservationResolution:
    action: str
    reason: str
    candidate: page_geometry.PageObservation
    matched: page_geometry.PageObservation | None = None
    geometry_score: float = 0.0
    coverage_ratio: float = 0.0
    text_overlap: float = 0.0
    useful_new_tokens: int = 0


@dataclass(frozen=True)
class ResolvedTextLine:
    text: str
    observation: page_geometry.PageObservation
    break_before: int = 1
    contributing_observations: tuple[page_geometry.PageObservation, ...] = ()
    resolution: ObservationResolution | None = None


def resolve_observation_append(
    candidate: page_geometry.PageObservation,
    accepted: Iterable[page_geometry.PageObservation],
    *,
    existing_text: str = "",
    existing_tokens: Set[str] | None = None,
) -> ObservationResolution:
    accepted_observations = tuple(
        observation for observation in accepted if observation.bbox is not None
    )
    if candidate.bbox is None:
        if observation_has_useful_new_text(
            candidate,
            existing_text,
            existing_tokens=existing_tokens,
        ):
            return ObservationResolution("append", "no_geometry", candidate)
        return ObservationResolution("skip", "duplicate_text", candidate)
    if not candidate.text.strip():
        return ObservationResolution("skip", "empty_text", candidate)

    matched, geometry_score, coverage_ratio = observation_geometry_resolution(
        candidate, accepted_observations
    )
    text_overlap = observation_text_overlap(candidate, matched) if matched is not None else 0.0
    useful_new_tokens = observation_useful_new_token_count(
        candidate,
        existing_text,
        existing_tokens=existing_tokens,
    )

    if coverage_ratio >= 0.72 and text_overlap >= 0.65:
        return ObservationResolution(
            "skip",
            "covered_geometry",
            candidate,
            matched,
            geometry_score,
            coverage_ratio,
            text_overlap,
            useful_new_tokens,
        )
    if geometry_score >= 0.86 and text_overlap >= 0.65:
        return ObservationResolution(
            "skip",
            "same_geometry",
            candidate,
            matched,
            geometry_score,
            coverage_ratio,
            text_overlap,
            useful_new_tokens,
        )
    if geometry_score >= 0.70 and text_overlap >= 0.45:
        return ObservationResolution(
            "skip",
            "same_text_region",
            candidate,
            matched,
            geometry_score,
            coverage_ratio,
            text_overlap,
            useful_new_tokens,
        )
    return ObservationResolution(
        "append",
        "uncovered",
        candidate,
        matched,
        geometry_score,
        coverage_ratio,
        text_overlap,
        useful_new_tokens,
    )


def resolve_text_lines(
    lines: Iterable[ResolvedTextLine],
    *,
    existing_text: str = "",
) -> tuple[ResolvedTextLine, ...]:
    accepted_lines: list[ResolvedTextLine] = []
    accepted_observations: list[page_geometry.PageObservation] = []
    accepted_tokens = set(normalized_text_tokens(existing_text))
    for line in lines:
        observation = line.observation
        if observation.text != line.text:
            observation = replace(observation, text=line.text)
            line = replace(line, observation=observation)
        resolution = resolve_observation_append(
            observation,
            accepted_observations,
            existing_tokens=accepted_tokens,
        )
        if resolution.action != "append":
            continue
        accepted_line = replace(line, resolution=resolution)
        accepted_lines.append(accepted_line)
        accepted_observations.append(observation)
        accepted_tokens.update(normalized_text_tokens(line.text))
    return tuple(accepted_lines)


def text_from_resolved_lines(lines: Iterable[ResolvedTextLine]) -> str:
    parts: list[str] = []
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if parts:
            parts.append("\n" * max(1, line.break_before))
        parts.append(text)
    return "".join(parts)


def best_observation_geometry_match(
    candidate: page_geometry.PageObservation,
    accepted: Iterable[page_geometry.PageObservation],
) -> tuple[page_geometry.PageObservation | None, float]:
    best: page_geometry.PageObservation | None = None
    best_score = 0.0
    for observation in accepted:
        score = page_geometry.observation_geometry_match_score(
            candidate,
            observation,
        )
        if score > best_score:
            best = observation
            best_score = score
    return best, best_score


def observation_geometry_resolution(
    candidate: page_geometry.PageObservation,
    accepted: Iterable[page_geometry.PageObservation],
) -> tuple[page_geometry.PageObservation | None, float, float]:
    best: page_geometry.PageObservation | None = None
    best_score = 0.0
    area = page_geometry.observation_area(candidate)
    covered_area = 0.0
    candidate_bbox = candidate.bbox
    if candidate_bbox is None:
        return (None, 0.0, 1.0)
    candidate_x0, candidate_y0, candidate_x1, candidate_y1 = candidate_bbox
    candidate_height = candidate_y1 - candidate_y0
    candidate_center_y = (candidate_y0 + candidate_y1) * 0.5
    for observation in accepted:
        observation_bbox = observation.bbox
        if observation_bbox is None:
            continue
        observation_x0, observation_y0, observation_x1, observation_y1 = observation_bbox
        if candidate_x1 <= observation_x0 or observation_x1 <= candidate_x0:
            continue
        if candidate_y1 <= observation_y0 or observation_y1 <= candidate_y0:
            observation_height = observation_y1 - observation_y0
            if candidate_height <= 0.0 or observation_height <= 0.0:
                continue
            observation_center_y = (observation_y0 + observation_y1) * 0.5
            if abs(candidate_center_y - observation_center_y) > (
                max(candidate_height, observation_height) * 0.55
            ):
                continue
        score, intersection_area = page_geometry.observation_geometry_match_metrics(
            candidate, observation
        )
        if score > best_score:
            best = observation
            best_score = score
        covered_area += intersection_area
    coverage_ratio = 1.0 if area <= 0.0 else min(1.0, covered_area / area)
    return best, best_score, coverage_ratio


def observation_coverage_ratio(
    candidate: page_geometry.PageObservation,
    accepted: Iterable[page_geometry.PageObservation],
) -> float:
    area = page_geometry.observation_area(candidate)
    if area <= 0.0:
        return 1.0
    covered_area = 0.0
    for observation in accepted:
        covered_area += page_geometry.observation_intersection_area(
            candidate,
            observation,
        )
    return min(1.0, covered_area / area)


def observation_text_overlap(
    left: page_geometry.PageObservation,
    right: page_geometry.PageObservation | None,
) -> float:
    if right is None:
        return 0.0
    left_tokens = set(normalized_text_tokens(left.text))
    right_tokens = set(normalized_text_tokens(right.text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def observation_has_useful_new_text(
    candidate: page_geometry.PageObservation,
    existing_text: str,
    *,
    existing_tokens: Set[str] | None = None,
) -> bool:
    return (
        observation_useful_new_token_count(
            candidate,
            existing_text,
            existing_tokens=existing_tokens,
        )
        > 0
    )


def observation_useful_new_token_count(
    candidate: page_geometry.PageObservation,
    existing_text: str,
    *,
    existing_tokens: Set[str] | None = None,
) -> int:
    seen = (
        existing_tokens
        if existing_tokens is not None
        else set(normalized_text_tokens(existing_text))
    )
    count = 0
    for token in normalized_text_tokens(candidate.text):
        if token in seen:
            continue
        if token_has_content_signal(token):
            count += 1
    return count


def token_has_content_signal(token: str) -> bool:
    alnum = "".join(ch for ch in token if ch.isalnum())
    if any(ch.isdigit() for ch in alnum):
        return True
    return len(alnum) >= 3
