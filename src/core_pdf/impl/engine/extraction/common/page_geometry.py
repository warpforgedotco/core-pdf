# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_layout.impl.layout.geometry import rect_tuple as layout_rect_tuple

Rect = tuple[float, float, float, float]
Point = tuple[float, float]
Segment = tuple[float, float, float, float]
Provenance = tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class PageSpace:
    bbox: Rect
    rotation: int = 0
    media_box: Rect | None = None
    crop_box: Rect | None = None
    source: str = "page"

    @classmethod
    def from_page(cls, page: Any, *, source: str = "page") -> PageSpace | None:
        media_box = normalize_rect(getattr(page, "media_box", None))
        crop_box = normalize_rect(getattr(page, "crop_box", None))
        bbox = crop_box or media_box
        if bbox is None:
            return None
        return cls(
            bbox,
            rotation=page_rotation_degrees(getattr(page, "rotation", 0)),
            media_box=media_box,
            crop_box=crop_box,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class PageObservation:
    kind: str
    source: str
    bbox: Rect | None = None
    advance_bbox: Rect | None = None
    ink_bbox: Rect | None = None
    confidence: float | None = None
    text: str = ""
    baseline: Segment | None = None
    provenance: Provenance = ()


@dataclass(frozen=True, slots=True)
class PageObservationSet:
    page_space: PageSpace | None
    observations: tuple[PageObservation, ...] = ()

    def by_kind(self, *kinds: str) -> tuple[PageObservation, ...]:
        accepted = set(kinds)
        return tuple(item for item in self.observations if item.kind in accepted)

    def by_source(self, *sources: str) -> tuple[PageObservation, ...]:
        accepted = set(sources)
        return tuple(item for item in self.observations if item.source in accepted)


def rect_box_tuple(value: Any) -> Rect | None:
    return layout_rect_tuple(value)


def normalize_rect(value: Any) -> Rect | None:
    rect = rect_box_tuple(value)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def valid_rect(rect: Rect | None) -> bool:
    return rect is not None and rect[2] > rect[0] and rect[3] > rect[1]


def rect_area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def rect_intersection_area(left: Rect, right: Rect) -> float:
    return rect_area(
        (
            max(left[0], right[0]),
            max(left[1], right[1]),
            min(left[2], right[2]),
            min(left[3], right[3]),
        )
    )


def page_rotation_degrees(value: Any) -> int:
    try:
        return int(value) % 360
    except (TypeError, ValueError):
        return 0


def text_run_bbox(run: Any) -> Rect | None:
    return normalize_rect(
        getattr(run, "ink_bbox", None)
        or getattr(run, "advance_bbox", None)
        or (
            getattr(run, "x0", None),
            getattr(run, "y0", None),
            getattr(run, "x1", None),
            getattr(run, "y1", None),
        )
    )


def provenance_tuple(**values: object) -> Provenance:
    return tuple((key, value) for key, value in values.items() if value is not None)


def collect_page_observations(
    page: Any,
    *,
    include_images: bool = True,
    **_: object,
) -> PageObservationSet:
    observations: list[PageObservation] = []
    state = getattr(page, "state", None)
    glyphs = tuple(getattr(state, "glyphs", ()) or ())
    for index, glyph in enumerate(glyphs):
        bbox = text_run_bbox(glyph)
        if bbox is not None:
            observations.append(
                PageObservation(
                    kind="native_glyph",
                    source="native_text",
                    bbox=bbox,
                    advance_bbox=normalize_rect(getattr(glyph, "advance_bbox", None)),
                    ink_bbox=normalize_rect(getattr(glyph, "ink_bbox", None)),
                    text=str(getattr(glyph, "text", "")),
                    confidence=getattr(glyph, "confidence", None),
                    provenance=provenance_tuple(glyph_index=index),
                )
            )
    get_lines = getattr(page, "get_text_lines", None)
    lines = tuple(get_lines()) if callable(get_lines) else ()
    for index, line in enumerate(lines):
        bbox = normalize_rect((line.x0, line.y0, line.x1, line.y1))
        if bbox is not None:
            observations.append(
                PageObservation(
                    kind="native_line",
                    source="native_text",
                    bbox=bbox,
                    advance_bbox=bbox,
                    ink_bbox=bbox,
                    text=str(line.text()),
                    provenance=provenance_tuple(line_index=index),
                )
            )
    if include_images:
        # Image observations are intentionally omitted: image text is not extracted.
        pass
    return PageObservationSet(PageSpace.from_page(page), tuple(observations))


def observation_area(observation: PageObservation) -> float:
    return rect_area(observation.bbox) if observation.bbox is not None else 0.0


def observation_intersection_area(left: PageObservation, right: PageObservation) -> float:
    if left.bbox is None or right.bbox is None:
        return 0.0
    return rect_intersection_area(left.bbox, right.bbox)


def observation_geometry_match_score(left: PageObservation, right: PageObservation) -> float:
    return observation_geometry_match_metrics(left, right)[0]


def observation_geometry_match_metrics(
    left: PageObservation, right: PageObservation
) -> tuple[float, float]:
    if left.bbox is None or right.bbox is None:
        return (0.0, 0.0)
    left_box, right_box = left.bbox, right.bbox
    left_width = left_box[2] - left_box[0]
    left_height = left_box[3] - left_box[1]
    right_width = right_box[2] - right_box[0]
    right_height = right_box[3] - right_box[1]
    if min(left_width, left_height, right_width, right_height) <= 0:
        return (0.0, 0.0)
    intersection = rect_intersection_area(left_box, right_box)
    x_overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    y_overlap = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
    row_alignment = max(
        y_overlap / min(left_height, right_height),
        1.0
        - abs((left_box[1] + left_box[3]) - (right_box[1] + right_box[3]))
        / 2
        / max(left_height, right_height),
    )
    horizontal_overlap = x_overlap / min(left_width, right_width)
    if row_alignment < 0.45 or horizontal_overlap < 0.18:
        return (0.0, intersection)
    return (min(1.0, row_alignment * 0.72 + horizontal_overlap * 0.18 + 0.1), intersection)
