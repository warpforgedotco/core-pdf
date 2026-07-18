# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from core_ocr.impl.candidates import OcrCandidate
from core_ocr.impl.services import get_candidate_services
from core_ocr.impl.text_analysis import normalized_text_tokens
from core_ocr.impl.types import OcrObservation, OcrRow, ocr_int_value


@dataclass(frozen=True)
class TextGeometryLine:
    text: str
    observation: Any

    @property
    def confidence(self) -> int | None:
        confidence = self.observation.confidence
        return int(round(confidence)) if confidence is not None else None


def text_geometry_line_from_bbox(
    text: str,
    bbox: tuple[float, float, float, float],
    confidence: int | float | None = None,
    *,
    source: str,
    kind: str,
    provenance: dict[str, object] | None = None,
) -> TextGeometryLine:
    page_geometry = get_candidate_services().page_geometry
    observation = page_geometry.page_observation_from_bbox(
        bbox,
        source=source,
        kind=kind,
        text=text,
        confidence=page_geometry.numeric_confidence(confidence),
        provenance=provenance,
    )
    if observation is None:
        raise ValueError("TextGeometryLine requires a valid bbox")
    return TextGeometryLine(text, observation)


def ocr_candidate_geometry_lines(
    page: Any,
    candidate: OcrCandidate | None,
    *,
    selected_tokens: set[str] | None = None,
) -> list[TextGeometryLine]:
    if candidate is None:
        return []
    rows = candidate.result.line_rows + selected_iterator_geometry_rows(
        candidate,
        selected_tokens,
    )
    if not rows:
        return []
    lines: list[TextGeometryLine] = []
    for row in rows:
        line = ocr_row_geometry_line(page, candidate, row)
        if line is not None:
            lines.append(line)
    return lines


def ocr_candidate_textline_geometry_lines(
    page: Any,
    candidate: OcrCandidate | None,
) -> list[TextGeometryLine]:
    if candidate is None:
        return []
    lines: list[TextGeometryLine] = []
    for row in candidate.result.line_rows:
        line = ocr_row_geometry_line(page, candidate, row)
        if line is not None:
            lines.append(line)
    return lines


def selected_iterator_geometry_rows(
    candidate: OcrCandidate,
    selected_tokens: set[str] | None,
) -> tuple[OcrRow, ...]:
    word_rows = candidate.result.word_rows
    symbol_rows = candidate.result.symbol_rows
    if selected_tokens is None:
        return (*word_rows, *symbol_rows)
    selected_word_rows = tuple(
        row
        for row in word_rows
        if selected_tokens.intersection(normalized_text_tokens(str(row.get("text", ""))))
    )
    selected_symbol_rows = tuple(
        row
        for row in symbol_rows
        if iterator_symbol_row_matches_selected_token(row, selected_tokens)
    )
    return (*selected_word_rows, *selected_symbol_rows)


def iterator_symbol_row_matches_selected_token(
    row: OcrRow,
    selected_tokens: set[str],
) -> bool:
    row_tokens = normalized_text_tokens(str(row.get("text", "")))
    if not row_tokens:
        return False
    for row_token in row_tokens:
        if row_token in selected_tokens:
            return True
        if len(row_token) == 1 and any(row_token in token for token in selected_tokens):
            return True
    return False


def ocr_row_geometry_line(
    page: Any,
    candidate: OcrCandidate,
    row: OcrRow,
) -> TextGeometryLine | None:
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    bbox = get_candidate_services().page_geometry.normalize_rect(row.get("page_bbox"))
    if not get_candidate_services().page_geometry.valid_rect(bbox):
        bbox = None
    try:
        left = ocr_int_value(row["left"])
        top = ocr_int_value(row["top"])
        right = left + ocr_int_value(row["width"])
        bottom = top + ocr_int_value(row["height"])
    except (KeyError, TypeError, ValueError):
        if bbox is None:
            return None
    else:
        bbox = bbox or ocr_pixel_bbox_to_page_bbox(
            page,
            candidate,
            (left, top, right, bottom),
        )
    if bbox is None:
        return None
    confidence = row.get("conf")
    try:
        confidence_int = ocr_int_value(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_int = None
    return text_geometry_line_from_bbox(
        text,
        bbox,
        confidence_int,
        source=candidate.name,
        kind="ocr_textline",
    )


def ocr_pixel_bbox_to_page_bbox(
    page: Any,
    candidate: OcrCandidate,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    if candidate.bbox is not None:
        bbox = (
            bbox[0] + candidate.bbox[0],
            bbox[1] + candidate.bbox[1],
            bbox[2] + candidate.bbox[0],
            bbox[3] + candidate.bbox[1],
        )
    geometry = ocr_candidate_page_geometry(page, candidate)
    if geometry is None:
        return None
    return get_candidate_services().page_geometry.image_bbox_to_page_bbox(bbox, geometry)


def ocr_candidate_page_geometry(
    page: Any,
    candidate: OcrCandidate,
) -> Any:
    return get_candidate_services().page_geometry.image_space_from_ocr_candidate(
        page,
        candidate,
        fallback_resolution=300,
    )


def ocr_row_page_bbox(
    row: Mapping[str, Any],
    geometry: Any,
) -> Any:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return get_candidate_services().page_geometry.image_bbox_to_page_bbox(
        (left, top, left + width, top + height), geometry
    )


def ocr_baseline_to_page(baseline: Any, geometry: Any) -> Any:
    baseline_type = type(baseline)
    if (baseline_type is not list and baseline_type is not tuple) or len(baseline) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in baseline)
    except (TypeError, ValueError):
        return None
    return get_candidate_services().page_geometry.image_segment_to_page_segment(
        (x1, y1), (x2, y2), geometry
    )


def normalize_ocr_row_to_page(
    row: Mapping[str, Any],
    geometry: Any,
    *,
    token_type_classifier: Any = None,
) -> dict[str, Any]:
    normalized = dict(row)
    page_bbox = ocr_row_page_bbox(row, geometry)
    if page_bbox is not None:
        normalized["page_bbox"] = page_bbox
    page_baseline = ocr_baseline_to_page(row.get("baseline"), geometry)
    if page_baseline is not None:
        normalized["page_baseline"] = page_baseline
    if token_type_classifier is not None:
        token_type = token_type_classifier(str(row.get("text", "")))
        if token_type is not None:
            normalized["token_type"] = token_type
    return normalized


def normalize_ocr_observation_to_page(
    observation: OcrObservation,
    geometry: Any,
    *,
    source: str = "",
    token_type_classifier: Any = None,
) -> OcrObservation:
    token_type = observation.token_type
    if token_type is None and token_type_classifier is not None:
        token_type = token_type_classifier(observation.text)
    page_geometry = get_candidate_services().page_geometry
    return replace(
        observation,
        source=source or observation.source,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        page_bbox=page_geometry.image_bbox_to_page_bbox(observation.bbox, geometry),
        page_baseline=ocr_baseline_to_page(observation.baseline, geometry),
        token_type=token_type,
        provenance=page_geometry.observation_provenance(geometry),
    )


def annotate_ocr_observation_page_geometry(
    observation: OcrObservation,
    geometry: Any,
    *,
    source: str = "",
) -> OcrObservation:
    page_geometry = get_candidate_services().page_geometry
    return replace(
        observation,
        source=source or observation.source,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        provenance=page_geometry.observation_provenance(geometry),
    )


def image_space_from_ocr_candidate(
    page: Any | None,
    candidate: Any,
    *,
    fallback_resolution: int | None = None,
) -> Any:
    page_geometry = get_candidate_services().page_geometry
    image_width = getattr(candidate, "image_width", None)
    image_height = getattr(candidate, "image_height", None)
    if image_width is None or image_height is None:
        return None
    image_resolution = getattr(candidate, "image_resolution", None) or fallback_resolution
    source = str(getattr(candidate, "name", "") or "")
    page_bbox = page_geometry.normalize_rect(getattr(candidate, "page_bbox", None))
    page_width = getattr(candidate, "page_width", None)
    page_height = getattr(candidate, "page_height", None)
    if page_bbox is not None or (page_width is not None and page_height is not None):
        return page_geometry.ImageSpace.from_dimensions(
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
            page_width=page_width,
            page_height=page_height,
            page_bbox=page_bbox,
            source=source,
        )
    if page is None:
        return page_geometry.ImageSpace.from_dimensions(
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
            source=source,
        )
    page_space = page_geometry.PageSpace.from_page(page, source="page")
    if page_space is None:
        return None
    return page_geometry.ImageSpace.from_dimensions(
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_width=page_space.width,
        page_height=page_space.height,
        page_bbox=page_space.bbox,
        clockwise_quarter_turns=page_geometry.page_rotation_to_clockwise_quarter_turns(
            page_space.rotation
        ),
        source=source,
    )


def page_observation_from_ocr_observation(observation: OcrObservation) -> Any:
    page_geometry = get_candidate_services().page_geometry
    return page_geometry.PageObservation(
        kind=ocr_observation_kind(observation),
        source=observation.source,
        bbox=observation.page_bbox,
        advance_bbox=observation.page_bbox,
        ink_bbox=observation.page_bbox,
        confidence=(float(observation.confidence) if observation.confidence is not None else None),
        text=observation.text,
        baseline=observation.page_baseline,
        provenance=observation.provenance,
    )


def page_observation_from_ocr_candidate_row(
    row: Mapping[str, Any],
    *,
    candidate: Any,
    geometry: Any,
    kind: str,
) -> Any:
    page_geometry = get_candidate_services().page_geometry
    source = str(getattr(candidate, "name", "") or "")
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    page_bbox = page_geometry.normalize_rect(row.get("page_bbox"))
    if page_bbox is None and geometry is not None:
        pixel_bbox = ocr_row_pixel_bbox(row)
        if pixel_bbox is not None:
            candidate_bbox = page_geometry.normalize_rect(getattr(candidate, "bbox", None))
            if candidate_bbox is not None:
                offset_x, offset_y = int(candidate_bbox[0]), int(candidate_bbox[1])
                pixel_bbox = (
                    pixel_bbox[0] + offset_x,
                    pixel_bbox[1] + offset_y,
                    pixel_bbox[2] + offset_x,
                    pixel_bbox[3] + offset_y,
                )
            page_bbox = page_geometry.image_bbox_to_page_bbox(pixel_bbox, geometry)
    if page_bbox is None:
        return None
    page_baseline = page_geometry.normalize_segment(row.get("page_baseline"))
    if page_baseline is None and geometry is not None:
        page_baseline = ocr_baseline_to_page(row.get("baseline"), geometry)
    provenance = dict(page_geometry.observation_provenance(geometry)) if geometry else {}
    provenance["row_kind"] = kind
    return page_geometry.PageObservation(
        kind=kind,
        source=source,
        bbox=page_bbox,
        confidence=page_geometry.numeric_confidence(row.get("conf")),
        text=text,
        baseline=page_baseline,
        provenance=page_geometry.provenance_tuple(provenance),
    )


def ocr_row_pixel_bbox(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return (left, top, left + width, top + height)


def page_observations_from_ocr_candidate(page: Any, candidate: Any) -> tuple[Any, ...]:
    result = getattr(candidate, "result", None)
    if result is None:
        return ()
    geometry = image_space_from_ocr_candidate(page, candidate)
    result_observations = tuple(getattr(result, "observations", ()) or ())
    if result_observations:
        observations = []
        for observation in result_observations:
            if getattr(observation, "page_bbox", None) is None and geometry is not None:
                observation = normalize_ocr_observation_to_page(
                    observation,
                    geometry,
                    source=str(getattr(candidate, "name", "") or ""),
                )
            page_observation = page_observation_from_ocr_observation(observation)
            if page_observation.bbox is not None:
                observations.append(page_observation)
        return tuple(observations)
    observations = []
    for row_attribute, kind in (
        ("line_rows", "ocr_textline"),
        ("word_rows", "ocr_word"),
        ("symbol_rows", "ocr_symbol"),
    ):
        for row in getattr(result, row_attribute, ()) or ():
            observation = page_observation_from_ocr_candidate_row(
                row,
                candidate=candidate,
                geometry=geometry,
                kind=kind,
            )
            if observation is not None:
                observations.append(observation)
    return tuple(observations)


def ocr_observation_kind(observation: OcrObservation) -> str:
    return {
        2: "ocr_textline",
        3: "ocr_word",
        4: "ocr_symbol",
    }.get(observation.level, "ocr")
