# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from core_pdf.impl.engine.extraction.ocr.types import OcrImage, OcrObservation
from core_pdf.impl.engine.layout.geometry import rect_tuple as layout_rect_tuple

Rect = tuple[float, float, float, float]
PixelRect = tuple[int, int, int, int]
Point = tuple[float, float]
Segment = tuple[float, float, float, float]
Provenance = tuple[tuple[str, object], ...]
TokenTypeClassifier = Callable[[str], str | None]


@dataclass(frozen=True)
class PageSpace:
    bbox: Rect
    rotation: int = 0
    media_box: Rect | None = None
    crop_box: Rect | None = None
    source: str = "page"

    @classmethod
    def from_page(cls, page: Any, *, source: str = "page") -> PageSpace | None:
        cache = getattr(page, "extraction_cache", None)
        cache_key = ("page_space", source)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        media_box = rect_box_tuple(getattr(page, "media_box", None))
        crop_box = rect_box_tuple(getattr(page, "crop_box", None))
        bbox = crop_box or media_box
        if bbox is None:
            return None
        normalized_bbox = normalize_rect(bbox)
        if normalized_bbox is None:
            return None
        page_space = cls(
            normalized_bbox,
            rotation=page_rotation_degrees(getattr(page, "rotation", 0)),
            media_box=normalize_rect(media_box) if media_box is not None else None,
            crop_box=normalize_rect(crop_box) if crop_box is not None else None,
            source=source,
        )
        if cache is not None:
            cache[cache_key] = page_space
        return page_space

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def display_width(self) -> float:
        return self.height if self.rotation % 180 else self.width

    @property
    def display_height(self) -> float:
        return self.width if self.rotation % 180 else self.height

    @property
    def display_bbox(self) -> Rect:
        return (0.0, 0.0, self.display_width, self.display_height)


@dataclass(frozen=True)
class ImageSpace:
    image_width: int | None
    image_height: int | None
    image_resolution: int | None = None
    page_width: float | None = None
    page_height: float | None = None
    page_bbox: Rect | None = None
    clockwise_quarter_turns: int = 0
    source: str = ""

    @classmethod
    def from_ocr_image(
        cls,
        image: OcrImage,
        *,
        source: str | None = None,
        page_width: float | None = None,
        page_height: float | None = None,
        page_bbox: Rect | None = None,
    ) -> ImageSpace:
        source_name = source if source is not None else image.source
        image_space = _cached_image_space_from_ocr_image(
            image_width=image.target_width or image.width,
            image_height=image.target_height or image.height,
            image_resolution=image.resolution,
            page_width=page_width,
            page_height=page_height,
            page_bbox=page_bbox or image.page_bbox,
            clockwise_quarter_turns=image.page_clockwise_quarter_turns,
            rendered_page_source=source_name.startswith("rendered_page_"),
        )
        return (
            image_space
            if source is None and image_space.source == source_name
            else replace(image_space, source=source_name)
        )

    @classmethod
    def from_dimensions(
        cls,
        *,
        image_width: int | None,
        image_height: int | None,
        image_resolution: int | None = None,
        page_width: float | None = None,
        page_height: float | None = None,
        page_bbox: Rect | None = None,
        clockwise_quarter_turns: int = 0,
        source: str = "",
    ) -> ImageSpace:
        image_space = _cached_image_space_from_dimensions(
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
            page_width=page_width,
            page_height=page_height,
            page_bbox=page_bbox,
            clockwise_quarter_turns=clockwise_quarter_turns,
        )
        if image_space.source == source:
            return image_space
        return replace(image_space, source=source)

    @property
    def valid_image_size(self) -> bool:
        return (
            self.image_width is not None
            and self.image_height is not None
            and self.image_width > 0
            and self.image_height > 0
        )

    @property
    def source_image_size(self) -> tuple[int, int] | None:
        if not self.valid_image_size:
            return None
        return source_image_size(
            self.image_width or 0,
            self.image_height or 0,
            self.clockwise_quarter_turns,
        )

    @property
    def effective_page_bbox(self) -> Rect | None:
        if self.page_bbox is not None:
            return self.page_bbox
        if (
            self.page_width is None
            or self.page_height is None
            or self.page_width <= 0.0
            or self.page_height <= 0.0
        ):
            return None
        return (0.0, 0.0, self.page_width, self.page_height)


@lru_cache(maxsize=8192)
def _cached_image_space_from_dimensions(
    *,
    image_width: int | None,
    image_height: int | None,
    image_resolution: int | None,
    page_width: float | None,
    page_height: float | None,
    page_bbox: Rect | None,
    clockwise_quarter_turns: int,
) -> ImageSpace:
    normalized_page_bbox = normalize_rect(page_bbox)
    if normalized_page_bbox is not None:
        page_width = max(0.0, normalized_page_bbox[2] - normalized_page_bbox[0])
        page_height = max(0.0, normalized_page_bbox[3] - normalized_page_bbox[1])
    return ImageSpace(
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_width=page_width,
        page_height=page_height,
        page_bbox=normalized_page_bbox,
        clockwise_quarter_turns=clockwise_quarter_turns,
        source="",
    )


@lru_cache(maxsize=8192)
def _cached_image_space_from_ocr_image(
    *,
    image_width: int | None,
    image_height: int | None,
    image_resolution: int | None,
    page_width: float | None,
    page_height: float | None,
    page_bbox: Rect | None,
    clockwise_quarter_turns: int,
    rendered_page_source: bool,
) -> ImageSpace:
    normalized_page_bbox = normalize_rect(page_bbox)
    if normalized_page_bbox is not None:
        page_width = max(0.0, normalized_page_bbox[2] - normalized_page_bbox[0])
        page_height = max(0.0, normalized_page_bbox[3] - normalized_page_bbox[1])
    elif rendered_page_source and (page_width is None or page_height is None):
        rendered_size = rendered_image_page_size_points(
            "rendered_page",
            image_width=image_width or 0,
            image_height=image_height or 0,
            image_resolution=image_resolution,
        )
        if rendered_size is not None:
            page_width, page_height = rendered_size
    return ImageSpace(
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_width=page_width,
        page_height=page_height,
        page_bbox=normalized_page_bbox,
        clockwise_quarter_turns=clockwise_quarter_turns,
        source="",
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


@dataclass(frozen=True)
class PageObservationSet:
    page_space: PageSpace | None
    observations: tuple[PageObservation, ...] = ()

    def by_kind(self, *kinds: str) -> tuple[PageObservation, ...]:
        accepted = set(kinds)
        return tuple(
            observation for observation in self.observations if observation.kind in accepted
        )

    def by_source(self, *sources: str) -> tuple[PageObservation, ...]:
        accepted = set(sources)
        return tuple(
            observation for observation in self.observations if observation.source in accepted
        )


def rect_box_tuple(value: Any) -> Rect | None:
    return layout_rect_tuple(value)


def normalize_rect(rect: Any) -> Rect | None:
    if type(rect) is tuple and len(rect) == 4:
        x0, y0, x1, y1 = rect
        if type(x0) is float and type(y0) is float and type(x1) is float and type(y1) is float:
            if x0 <= x1 and y0 <= y1:
                return rect
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    box = rect_box_tuple(rect)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def valid_rect(rect: Rect | None) -> bool:
    return rect is not None and rect[2] > rect[0] and rect[3] > rect[1]


def rect_area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def rect_intersection_area(left: Rect, right: Rect) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def text_run_bbox(run: Any) -> Rect | None:
    rect = rect_box_tuple(run)
    if not valid_rect(rect):
        return None
    return rect


def page_rotation_degrees(rotation: Any) -> int:
    try:
        return int(rotation) % 360
    except (TypeError, ValueError):
        return 0


def page_rotation_to_clockwise_quarter_turns(rotation: Any) -> int:
    return (page_rotation_degrees(rotation) // 90) % 4


def rendered_image_page_size_points(
    source: str,
    *,
    image_width: int | None,
    image_height: int | None,
    image_resolution: int | None,
) -> tuple[float, float] | None:
    if not source.startswith("rendered_page_"):
        return None
    if image_resolution is None or image_resolution <= 0:
        return None
    if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
        return None
    scale = 72.0 / float(image_resolution)
    return (float(image_width) * scale, float(image_height) * scale)


def source_image_size(
    image_width: int,
    image_height: int,
    clockwise_quarter_turns: int,
) -> tuple[int, int]:
    if clockwise_quarter_turns % 2:
        return image_height, image_width
    return image_width, image_height


def unrotate_image_point(
    x: float,
    y: float,
    geometry: ImageSpace,
) -> Point | None:
    if geometry.image_width is None or geometry.image_height is None:
        return None
    turns = geometry.clockwise_quarter_turns % 4
    if turns == 0:
        return float(x), float(y)
    if turns == 2:
        return (
            float(geometry.image_width) - float(x),
            float(geometry.image_height) - float(y),
        )
    source_width, source_height = source_image_size(
        geometry.image_width,
        geometry.image_height,
        turns,
    )
    if turns == 1:
        return float(source_width) - float(y), float(x)
    return float(y), float(source_height) - float(x)


def rotate_source_point(
    x: float,
    y: float,
    geometry: ImageSpace,
) -> Point | None:
    if geometry.image_width is None or geometry.image_height is None:
        return None
    turns = geometry.clockwise_quarter_turns % 4
    if turns == 0:
        return float(x), float(y)
    if turns == 2:
        return (
            float(geometry.image_width) - float(x),
            float(geometry.image_height) - float(y),
        )
    source_width, source_height = source_image_size(
        geometry.image_width,
        geometry.image_height,
        turns,
    )
    if turns == 1:
        return float(y), float(source_width) - float(x)
    return float(source_height) - float(y), float(x)


def unrotate_image_bbox(
    bbox: PixelRect | Rect,
    geometry: ImageSpace,
) -> Rect | None:
    rect = normalize_rect(bbox)
    if rect is None or not geometry.valid_image_size:
        return None
    left, top, right, bottom = rect
    if right <= left or bottom <= top:
        return None
    points = (
        unrotate_image_point(left, top, geometry),
        unrotate_image_point(right, top, geometry),
        unrotate_image_point(right, bottom, geometry),
        unrotate_image_point(left, bottom, geometry),
    )
    if any(point is None for point in points):
        return None
    xs = [point[0] for point in points if point is not None]
    ys = [point[1] for point in points if point is not None]
    return (min(xs), min(ys), max(xs), max(ys))


def image_point_to_page_point(point: Point, geometry: ImageSpace) -> Point | None:
    source_size = geometry.source_image_size
    page_bbox = geometry.effective_page_bbox
    if source_size is None or page_bbox is None:
        return None
    source_width, source_height = source_size
    page_x0, page_y0, page_x1, page_y1 = page_bbox
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    if source_width <= 0 or source_height <= 0 or page_width <= 0 or page_height <= 0:
        return None
    source_point = unrotate_image_point(point[0], point[1], geometry)
    if source_point is None:
        return None
    x_scale = page_width / float(source_width)
    y_scale = page_height / float(source_height)
    return (
        page_x0 + source_point[0] * x_scale,
        page_y1 - source_point[1] * y_scale,
    )


def page_point_to_image_point(point: Point, geometry: ImageSpace) -> Point | None:
    source_size = geometry.source_image_size
    page_bbox = geometry.effective_page_bbox
    if source_size is None or page_bbox is None:
        return None
    source_width, source_height = source_size
    page_x0, page_y0, page_x1, page_y1 = page_bbox
    page_width = page_x1 - page_x0
    page_height = page_y1 - page_y0
    if source_width <= 0 or source_height <= 0 or page_width <= 0 or page_height <= 0:
        return None
    source_x = (float(point[0]) - page_x0) * float(source_width) / page_width
    source_y = (page_y1 - float(point[1])) * float(source_height) / page_height
    return rotate_source_point(source_x, source_y, geometry)


def image_bbox_to_page_bbox(
    bbox: PixelRect | Rect | None,
    geometry: ImageSpace,
) -> Rect | None:
    if bbox is None:
        return None
    rect = normalize_rect(bbox)
    if rect is None:
        return None
    left, top, right, bottom = rect
    points = (
        image_point_to_page_point((left, top), geometry),
        image_point_to_page_point((right, top), geometry),
        image_point_to_page_point((right, bottom), geometry),
        image_point_to_page_point((left, bottom), geometry),
    )
    if any(point is None for point in points):
        return None
    xs = [point[0] for point in points if point is not None]
    ys = [point[1] for point in points if point is not None]
    return (min(xs), min(ys), max(xs), max(ys))


def page_bbox_to_image_bbox(
    bbox: Rect | None,
    geometry: ImageSpace,
    *,
    padding: float = 0.0,
) -> Rect | None:
    rect = normalize_rect(bbox)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    if padding:
        x0 -= padding
        y0 -= padding
        x1 += padding
        y1 += padding
    points = (
        page_point_to_image_point((x0, y0), geometry),
        page_point_to_image_point((x1, y0), geometry),
        page_point_to_image_point((x1, y1), geometry),
        page_point_to_image_point((x0, y1), geometry),
    )
    if any(point is None for point in points):
        return None
    xs = [point[0] for point in points if point is not None]
    ys = [point[1] for point in points if point is not None]
    return (min(xs), min(ys), max(xs), max(ys))


def page_bbox_to_image_pixel_bbox(
    bbox: Rect | None,
    geometry: ImageSpace,
    *,
    padding: float = 0.0,
    clamp: bool = True,
) -> PixelRect | None:
    image_bbox = page_bbox_to_image_bbox(bbox, geometry, padding=padding)
    if image_bbox is None:
        return None
    left = int(math.floor(image_bbox[0]))
    top = int(math.floor(image_bbox[1]))
    right = int(math.ceil(image_bbox[2]))
    bottom = int(math.ceil(image_bbox[3]))
    if clamp:
        if geometry.image_width is None or geometry.image_height is None:
            return None
        left = max(0, min(geometry.image_width, left))
        top = max(0, min(geometry.image_height, top))
        right = max(0, min(geometry.image_width, right))
        bottom = max(0, min(geometry.image_height, bottom))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def image_segment_to_page_segment(
    start: Point,
    end: Point,
    geometry: ImageSpace,
) -> Segment | None:
    page_start = image_point_to_page_point(start, geometry)
    page_end = image_point_to_page_point(end, geometry)
    if page_start is None or page_end is None:
        return None
    return (page_start[0], page_start[1], page_end[0], page_end[1])


def image_axis_length_to_page_length(
    value: float,
    geometry: ImageSpace,
    *,
    axis: str,
) -> float | None:
    source_size = geometry.source_image_size
    page_bbox = geometry.effective_page_bbox
    if source_size is None or page_bbox is None:
        return None
    source_width, source_height = source_size
    page_width = page_bbox[2] - page_bbox[0]
    page_height = page_bbox[3] - page_bbox[1]
    if source_width <= 0 or source_height <= 0 or page_width <= 0 or page_height <= 0:
        return None
    if axis == "x":
        return float(value) * page_width / float(source_width)
    return float(value) * page_height / float(source_height)


def ocr_row_page_bbox(
    row: Mapping[str, Any],
    geometry: ImageSpace,
) -> Rect | None:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return image_bbox_to_page_bbox((left, top, left + width, top + height), geometry)


def ocr_baseline_to_page(
    baseline: Any,
    geometry: ImageSpace,
) -> Segment | None:
    baseline_type = type(baseline)
    if (baseline_type is not list and baseline_type is not tuple) or len(baseline) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in baseline)
    except (TypeError, ValueError):
        return None
    return image_segment_to_page_segment((x1, y1), (x2, y2), geometry)


def normalize_ocr_row_to_page(
    row: Mapping[str, Any],
    geometry: ImageSpace,
    *,
    token_type_classifier: TokenTypeClassifier | None = None,
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
    geometry: ImageSpace,
    *,
    source: str = "",
    token_type_classifier: TokenTypeClassifier | None = None,
) -> OcrObservation:
    from dataclasses import replace

    page_bbox = image_bbox_to_page_bbox(observation.bbox, geometry)
    page_baseline = ocr_baseline_to_page(observation.baseline, geometry)
    token_type = observation.token_type
    if token_type is None and token_type_classifier is not None:
        token_type = token_type_classifier(observation.text)
    return replace(
        observation,
        source=source or observation.source,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        page_bbox=page_bbox,
        page_baseline=page_baseline,
        token_type=token_type,
        provenance=observation_provenance(geometry),
    )


def annotate_ocr_observation_page_geometry(
    observation: OcrObservation,
    geometry: ImageSpace,
    *,
    source: str = "",
) -> OcrObservation:
    from dataclasses import replace

    return replace(
        observation,
        source=source or observation.source,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        provenance=observation_provenance(geometry),
    )


def observation_provenance(geometry: ImageSpace) -> Provenance:
    page_bbox = geometry.effective_page_bbox
    return provenance_tuple(
        source=geometry.source,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        image_resolution=geometry.image_resolution,
        page_bbox=page_bbox,
        clockwise_quarter_turns=geometry.clockwise_quarter_turns,
    )


def provenance_tuple(
    mapping: Mapping[str, object] | None = None,
    **values: object,
) -> Provenance:
    items: list[tuple[str, object]] = []
    if mapping is not None:
        items.extend((str(key), value) for key, value in mapping.items())
    items.extend((key, value) for key, value in values.items())
    return tuple((key, value) for key, value in items if value is not None)


def collect_page_observations(
    page: Any,
    *,
    ocr_candidates: Iterable[Any] = (),
    rendered_page: Any | None = None,
    raster_lines: Iterable[Any] = (),
    vector_lines: Iterable[Any] | None = None,
    include_native_glyphs: bool = True,
    include_native_lines: bool = True,
    include_native_words: bool = True,
    include_vector_lines: bool = True,
    include_images: bool = True,
) -> PageObservationSet:
    observations: list[PageObservation] = []
    observations.extend(
        page_observations_from_native_text(
            page,
            include_glyphs=include_native_glyphs,
            include_lines=include_native_lines,
            include_words=include_native_words,
        )
    )
    for candidate in ocr_candidates:
        observations.extend(page_observations_from_ocr_candidate(page, candidate))
    if include_vector_lines:
        lines = vector_lines
        if lines is None:
            lines = page_grid_lines(page)
        observations.extend(
            page_observations_from_lines(
                lines,
                source="vector_grid",
                kind="vector_line",
            )
        )
    observations.extend(
        page_observations_from_lines(
            raster_lines,
            source="raster_grid",
            kind="raster_line",
        )
    )
    if include_images:
        if rendered_page is not None:
            observations.extend(page_observations_from_rendered_page(rendered_page))
        else:
            observations.extend(page_observations_from_page_images(page))
    return PageObservationSet(
        PageSpace.from_page(page, source="page"),
        tuple(observations),
    )


def page_observations_from_native_text(
    page: Any,
    *,
    include_glyphs: bool = True,
    include_lines: bool = True,
    include_words: bool = True,
) -> tuple[PageObservation, ...]:
    observations: list[PageObservation] = []
    if include_glyphs:
        glyph_observations = native_glyph_observations(page)
        for glyph_index, glyph in enumerate(glyph_observations):
            observation = page_observation_from_glyph_observation(
                glyph,
                source="native_text",
                kind="native_glyph",
                glyph_index=glyph_index,
            )
            if observation is not None:
                observations.append(observation)
        for cluster_index, cluster in enumerate(native_glyph_clusters(page)):
            observation = page_observation_from_glyph_cluster(
                cluster,
                source="native_text",
                kind="native_glyph_cluster",
                cluster_index=cluster_index,
            )
            if observation is not None:
                observations.append(observation)
    if include_lines or include_words:
        for line_index, line in enumerate(page_text_lines(page)):
            if include_lines:
                observation = page_observation_from_text_line(
                    line,
                    source="native_text",
                    kind="native_line",
                    line_index=line_index,
                )
                if observation is not None:
                    observations.append(observation)
            if include_words:
                for word_index, word in enumerate(text_line_words(line)):
                    observation = page_observation_from_text_line(
                        word,
                        source="native_text",
                        kind="native_word",
                        line_index=line_index,
                        word_index=word_index,
                    )
                    if observation is not None:
                        observations.append(observation)
    return tuple(observations)


def native_glyph_observations(page: Any) -> tuple[Any, ...]:
    state = getattr(page, "state", None)
    glyphs = tuple(getattr(state, "glyphs", ()) or ())
    if glyphs:
        return glyphs
    get_state = getattr(page, "get_state", None)
    if not callable(get_state):
        return ()
    try:
        state = get_state()
    except Exception:
        return ()
    return tuple(getattr(state, "glyphs", ()) or ())


def native_glyph_clusters(page: Any) -> tuple[Any, ...]:
    state = getattr(page, "state", None)
    clusters = tuple(getattr(state, "glyph_clusters", ()) or ())
    if clusters:
        return clusters
    get_state = getattr(page, "get_state", None)
    if not callable(get_state):
        return ()
    try:
        state = get_state()
    except Exception:
        return ()
    return tuple(getattr(state, "glyph_clusters", ()) or ())


def page_text_lines(page: Any) -> tuple[Any, ...]:
    get_text_lines = getattr(page, "get_text_lines", None)
    if callable(get_text_lines):
        try:
            return tuple(get_text_lines())
        except Exception:
            return ()
    extract_lines = getattr(page, "extract_lines", None)
    if callable(extract_lines):
        try:
            return tuple(extract_lines(include_words=True))
        except Exception:
            return ()
    return ()


def text_line_words(line: Any) -> tuple[Any, ...]:
    if isinstance(line, Mapping):
        words = line.get("words", ())
        return tuple(words) if words is not None else ()
    words = getattr(line, "words", None)
    if callable(words):
        try:
            return tuple(words())
        except Exception:
            return ()
    if words is None:
        return ()
    try:
        return tuple(words)
    except TypeError:
        return ()


def page_observations_from_ocr_candidate(
    page: Any,
    candidate: Any,
) -> tuple[PageObservation, ...]:
    result = getattr(candidate, "result", None)
    if result is None:
        return ()
    geometry = image_space_from_ocr_candidate(page, candidate)
    result_observations = tuple(getattr(result, "observations", ()) or ())
    if result_observations:
        observations: list[PageObservation] = []
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


def image_space_from_ocr_candidate(
    page: Any | None,
    candidate: Any,
    *,
    fallback_resolution: int | None = None,
) -> ImageSpace | None:
    image_width = getattr(candidate, "image_width", None)
    image_height = getattr(candidate, "image_height", None)
    if image_width is None or image_height is None:
        return None
    image_resolution = getattr(candidate, "image_resolution", None) or fallback_resolution
    source = str(getattr(candidate, "name", "") or "")
    page_bbox = normalize_rect(getattr(candidate, "page_bbox", None))
    page_width = getattr(candidate, "page_width", None)
    page_height = getattr(candidate, "page_height", None)
    if page_bbox is not None or (page_width is not None and page_height is not None):
        return ImageSpace.from_dimensions(
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
            page_width=page_width,
            page_height=page_height,
            page_bbox=page_bbox,
            source=source,
        )
    if page is None:
        return ImageSpace.from_dimensions(
            image_width=image_width,
            image_height=image_height,
            image_resolution=image_resolution,
            source=source,
        )
    page_space = PageSpace.from_page(page, source="page")
    if page_space is None:
        return None
    return ImageSpace.from_dimensions(
        image_width=image_width,
        image_height=image_height,
        image_resolution=image_resolution,
        page_width=page_space.width,
        page_height=page_space.height,
        page_bbox=page_space.bbox,
        clockwise_quarter_turns=page_rotation_to_clockwise_quarter_turns(page_space.rotation),
        source=source,
    )


def page_observation_from_ocr_candidate_row(
    row: Mapping[str, Any],
    *,
    candidate: Any,
    geometry: ImageSpace | None,
    kind: str,
) -> PageObservation | None:
    source = str(getattr(candidate, "name", "") or "")
    text = str(row.get("text", "")).strip()
    if not text:
        return None
    page_bbox = normalize_rect(row.get("page_bbox"))
    if page_bbox is None and geometry is not None:
        pixel_bbox = ocr_row_pixel_bbox(row)
        if pixel_bbox is not None:
            candidate_bbox = normalize_rect(getattr(candidate, "bbox", None))
            if candidate_bbox is not None:
                offset_x, offset_y = int(candidate_bbox[0]), int(candidate_bbox[1])
                pixel_bbox = (
                    pixel_bbox[0] + offset_x,
                    pixel_bbox[1] + offset_y,
                    pixel_bbox[2] + offset_x,
                    pixel_bbox[3] + offset_y,
                )
            page_bbox = image_bbox_to_page_bbox(pixel_bbox, geometry)
    if page_bbox is None:
        return None
    page_baseline = normalize_segment(row.get("page_baseline"))
    if page_baseline is None and geometry is not None:
        page_baseline = ocr_baseline_to_page(row.get("baseline"), geometry)
    provenance = dict(observation_provenance(geometry)) if geometry is not None else {}
    provenance["row_kind"] = kind
    return PageObservation(
        kind=kind,
        source=source,
        bbox=page_bbox,
        confidence=numeric_confidence(row.get("conf")),
        text=text,
        baseline=page_baseline,
        provenance=provenance_tuple(provenance),
    )


def ocr_row_pixel_bbox(row: Mapping[str, Any]) -> PixelRect | None:
    try:
        left = int(row["left"])
        top = int(row["top"])
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return (left, top, left + width, top + height)


def page_grid_lines(page: Any) -> tuple[Any, ...]:
    get_grid_lines = getattr(page, "get_grid_lines", None)
    if not callable(get_grid_lines):
        return ()
    try:
        return tuple(get_grid_lines())
    except Exception:
        return ()


def page_observations_from_lines(
    lines: Iterable[Any],
    *,
    source: str,
    kind: str,
) -> tuple[PageObservation, ...]:
    observations: list[PageObservation] = []
    for line_index, line in enumerate(lines):
        segment = line_segment(line)
        if segment is None:
            continue
        observation = page_observation_from_line(
            segment,
            source=source,
            kind=kind,
            line_width=numeric_confidence(line_value(line, "line_width")) or 1.0,
            provenance={
                "line_index": line_index,
                "line_width": line_value(line, "line_width"),
            },
        )
        observations.append(observation)
    return tuple(observations)


def page_observations_from_page_images(page: Any) -> tuple[PageObservation, ...]:
    extract_images = getattr(page, "extract_images", None)
    if not callable(extract_images):
        return ()
    try:
        image_records = extract_images()
    except Exception:
        return ()
    return page_observations_from_image_records(
        image_records,
        source="page.extract_images",
    )


def page_observations_from_image_records(
    image_records: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> tuple[PageObservation, ...]:
    observations: list[PageObservation] = []
    for image_index, image in enumerate(image_records):
        bbox = normalize_rect(image.get("bbox"))
        if bbox is None:
            continue
        provenance = {
            "image_index": image_index,
            "seqno": image.get("seqno"),
            "width": image.get("width"),
            "height": image.get("height"),
            "pixels": image.get("pixels"),
        }
        observation = page_observation_from_bbox(
            bbox,
            source=source,
            kind=str(image.get("kind") or "image"),
            provenance=provenance,
        )
        if observation is not None:
            observations.append(observation)
    return tuple(observations)


def page_observations_from_rendered_page(
    rendered_page: Any,
    *,
    source: str = "rendered_page",
) -> tuple[PageObservation, ...]:
    display_list = getattr(rendered_page, "display_list", None)
    items = getattr(display_list, "items", ())
    observations: list[PageObservation] = []
    for item_index, item in enumerate(items):
        if getattr(item, "kind", None) not in {"image", "inline-image"}:
            continue
        data = getattr(item, "data", {})
        if not isinstance(data, Mapping):
            continue
        bbox = normalize_rect(data.get("bbox"))
        if bbox is None:
            continue
        metadata = data.get("image_metadata")
        provenance = {
            "item_index": item_index,
            "seqno": getattr(item, "seqno", None),
        }
        if isinstance(metadata, Mapping):
            for key in ("width", "height", "pixels", "bits_per_component"):
                provenance[key] = metadata.get(key)
        observation = page_observation_from_bbox(
            bbox,
            source=source,
            kind=str(getattr(item, "kind", "image")),
            provenance=provenance,
        )
        if observation is not None:
            observations.append(observation)
    return tuple(observations)


def page_observation_from_ocr_observation(
    observation: OcrObservation,
) -> PageObservation:
    return PageObservation(
        kind=ocr_observation_kind(observation),
        source=observation.source,
        bbox=observation.page_bbox,
        advance_bbox=observation.page_bbox,
        ink_bbox=observation.page_bbox,
        confidence=float(observation.confidence) if observation.confidence is not None else None,
        text=observation.text,
        baseline=observation.page_baseline,
        provenance=observation.provenance,
    )


def page_observation_from_text_line(
    line: Any,
    *,
    source: str,
    kind: str,
    line_index: int | None = None,
    word_index: int | None = None,
) -> PageObservation | None:
    embedded = line_value(line, "observation")
    if isinstance(embedded, PageObservation):
        return PageObservation(
            kind=kind,
            source=source,
            bbox=embedded.bbox,
            advance_bbox=embedded.advance_bbox,
            ink_bbox=embedded.ink_bbox,
            confidence=embedded.confidence,
            text=object_text(line) or embedded.text,
            baseline=embedded.baseline,
            provenance=(
                *embedded.provenance,
                *provenance_tuple(
                    object_type=type(line).__name__,
                    line_index=line_index,
                    word_index=word_index,
                ),
            ),
        )
    bbox = object_bbox(line)
    if not valid_rect(bbox):
        return None
    confidence = numeric_confidence(line_value(line, "confidence"))
    return PageObservation(
        kind=kind,
        source=source,
        bbox=bbox,
        advance_bbox=bbox,
        ink_bbox=bbox,
        confidence=confidence,
        text=object_text(line),
        provenance=provenance_tuple(
            object_type=type(line).__name__,
            line_index=line_index,
            word_index=word_index,
            min_order=line_value(line, "min_order"),
            max_order=line_value(line, "max_order"),
            max_depth=line_value(line, "max_depth"),
            rotation_angle=line_value(line, "rotation_angle"),
            is_vertical=line_value(line, "is_vertical"),
        ),
    )


def page_observation_from_glyph_observation(
    glyph: Any,
    *,
    source: str = "native",
    kind: str = "native_glyph",
    glyph_index: int | None = None,
) -> PageObservation | None:
    ink_bbox = normalize_rect(getattr(glyph, "ink_rect", None))
    advance_bbox = normalize_rect(getattr(glyph, "advance_rect", None))
    bbox = ink_bbox or advance_bbox
    if bbox is None:
        return None
    visible = getattr(glyph, "visible", None)
    return PageObservation(
        kind=kind,
        source=source,
        bbox=bbox,
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        confidence=getattr(glyph, "confidence", None),
        text=str(getattr(glyph, "text", "")),
        provenance=provenance_tuple(
            visible=visible,
            seqno=getattr(glyph, "seqno", None),
            code=getattr(glyph, "cid", None),
            gid=getattr(glyph, "gid", None),
            font_name=getattr(glyph, "font_name", None),
            unicode_source=getattr(glyph, "unicode_source", None),
            code_bytes=getattr(glyph, "code_bytes", None),
            cluster_id=getattr(glyph, "cluster_id", None),
            cluster_index=getattr(glyph, "cluster_index", None),
            cluster_size=getattr(glyph, "cluster_size", None),
            glyph_index=glyph_index,
        ),
    )


def page_observation_from_glyph_cluster(
    cluster: Any,
    *,
    source: str = "native",
    kind: str = "native_glyph_cluster",
    cluster_index: int | None = None,
) -> PageObservation | None:
    ink_bbox = normalize_rect(getattr(cluster, "ink_bbox", None))
    advance_bbox = normalize_rect(getattr(cluster, "advance_bbox", None))
    bbox = ink_bbox or advance_bbox
    if bbox is None:
        return None
    return PageObservation(
        kind=kind,
        source=source,
        bbox=bbox,
        advance_bbox=advance_bbox,
        ink_bbox=ink_bbox,
        confidence=getattr(cluster, "confidence", None),
        text=str(getattr(cluster, "text", "")),
        provenance=provenance_tuple(
            seqno=getattr(cluster, "seqno", None),
            font_name=getattr(cluster, "font_name", None),
            cluster_id=getattr(cluster, "cluster_id", None),
            cluster_kind=getattr(cluster, "kind", None),
            cluster_index=cluster_index,
            glyph_count=len(getattr(cluster, "glyphs", ()) or ()),
        ),
    )


def page_observation_from_bbox(
    bbox: Any,
    *,
    source: str,
    kind: str,
    text: str = "",
    confidence: float | None = None,
    baseline: Segment | None = None,
    provenance: Mapping[str, object] | None = None,
) -> PageObservation | None:
    normalized = normalize_rect(bbox)
    if normalized is None:
        return None
    return PageObservation(
        kind=kind,
        source=source,
        bbox=normalized,
        advance_bbox=normalized,
        ink_bbox=normalized,
        confidence=confidence,
        text=text,
        baseline=baseline,
        provenance=provenance_tuple(provenance),
    )


def page_observation_from_line(
    segment: Segment,
    *,
    source: str,
    kind: str,
    confidence: float | None = None,
    line_width: float = 0.0,
    provenance: Mapping[str, object] | None = None,
) -> PageObservation:
    segment = normalize_segment(segment) or segment
    return PageObservation(
        kind=kind,
        source=source,
        bbox=line_segment_bbox(segment, line_width=line_width),
        advance_bbox=line_segment_bbox(segment, line_width=line_width),
        ink_bbox=line_segment_bbox(segment, line_width=line_width),
        confidence=confidence,
        baseline=segment,
        provenance=provenance_tuple(provenance),
    )


def line_segment_bbox(segment: Segment, *, line_width: float = 0.0) -> Rect | None:
    x0, y0, x1, y1 = segment
    half_width = max(0.0, float(line_width) * 0.5)
    if x0 == x1 and y0 == y1:
        half_width = max(half_width, 0.5)
    box = normalize_rect((x0, y0, x1, y1))
    if box is None:
        return None
    if half_width <= 0.0:
        return box
    return (
        box[0] - half_width,
        box[1] - half_width,
        box[2] + half_width,
        box[3] + half_width,
    )


def object_bbox(value: Any) -> Rect | None:
    if isinstance(value, Mapping):
        for key in ("bbox", "rect"):
            box = normalize_rect(value.get(key))
            if valid_rect(box):
                return box
        return None
    for key in ("bbox", "rect"):
        box = normalize_rect(getattr(value, key, None))
        if valid_rect(box):
            return box
    box = normalize_rect(value)
    return box if valid_rect(box) else None


def object_text(value: Any) -> str:
    if isinstance(value, Mapping):
        text = value.get("text", "")
    else:
        text = getattr(value, "text", "")
    if callable(text):
        text = text()
    return "" if text is None else str(text)


def line_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def line_segment(value: Any) -> Segment | None:
    if isinstance(value, Mapping):
        segment = normalize_segment(value.get("segment"))
        if segment is not None:
            return segment
        try:
            return (
                float(value["x0"]),
                float(value["y0"]),
                float(value["x1"]),
                float(value["y1"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    segment = normalize_segment(value)
    if segment is not None:
        return segment
    try:
        return (
            float(getattr(value, "x0")),
            float(getattr(value, "y0")),
            float(getattr(value, "x1")),
            float(getattr(value, "y1")),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def observation_area(observation: PageObservation) -> float:
    bbox = observation.bbox
    if bbox is None:
        return 0.0
    return rect_area(bbox)


def observation_width(observation: PageObservation) -> float:
    if observation.bbox is None:
        return 0.0
    return max(0.0, observation.bbox[2] - observation.bbox[0])


def observation_height(observation: PageObservation) -> float:
    if observation.bbox is None:
        return 0.0
    return max(0.0, observation.bbox[3] - observation.bbox[1])


def observation_center(observation: PageObservation) -> Point | None:
    if observation.bbox is None:
        return None
    x0, y0, x1, y1 = observation.bbox
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def observation_mid_x(observation: PageObservation) -> float:
    center = observation_center(observation)
    return center[0] if center is not None else 0.0


def observation_mid_y(observation: PageObservation) -> float:
    center = observation_center(observation)
    return center[1] if center is not None else 0.0


def observation_reading_order_key(observation: PageObservation) -> tuple[float, float]:
    if observation.bbox is None:
        return (0.0, 0.0)
    return (-observation_mid_y(observation), observation.bbox[0])


def observation_union_bbox(
    observations: Iterable[PageObservation],
) -> Rect | None:
    boxes = [observation.bbox for observation in observations if observation.bbox is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def observation_intersection_area(
    left: PageObservation,
    right: PageObservation,
) -> float:
    if left.bbox is None or right.bbox is None:
        return 0.0
    return rect_intersection_area(left.bbox, right.bbox)


def observation_overlap_ratio(
    left: PageObservation,
    right: PageObservation,
    *,
    denominator: str = "left",
) -> float:
    overlap = observation_intersection_area(left, right)
    if overlap <= 0.0:
        return 0.0
    left_area = observation_area(left)
    right_area = observation_area(right)
    if denominator == "right":
        area = right_area
    elif denominator == "smaller":
        area = min(left_area, right_area)
    elif denominator == "larger":
        area = max(left_area, right_area)
    else:
        area = left_area
    if area <= 0.0:
        return 0.0
    return overlap / area


def observation_geometry_match_score(
    left: PageObservation,
    right: PageObservation,
) -> float:
    return observation_geometry_match_metrics(left, right)[0]


def observation_geometry_match_metrics(
    left: PageObservation,
    right: PageObservation,
) -> tuple[float, float]:
    if left.bbox is None or right.bbox is None:
        return (0.0, 0.0)
    left_bbox = left.bbox
    right_bbox = right.bbox
    if right_bbox < left_bbox:
        left_bbox, right_bbox = right_bbox, left_bbox
    return bbox_geometry_match_metrics(left_bbox, right_bbox)


def bbox_geometry_match_score(
    left_bbox: Rect,
    right_bbox: Rect,
) -> float:
    return bbox_geometry_match_metrics(left_bbox, right_bbox)[0]


def bbox_geometry_match_metrics(
    left_bbox: Rect,
    right_bbox: Rect,
) -> tuple[float, float]:
    left_x0, left_y0, left_x1, left_y1 = left_bbox
    right_x0, right_y0, right_x1, right_y1 = right_bbox
    left_width = max(0.0, left_x1 - left_x0)
    left_height = max(0.0, left_y1 - left_y0)
    right_width = max(0.0, right_x1 - right_x0)
    right_height = max(0.0, right_y1 - right_y0)
    if min(left_width, left_height, right_width, right_height) <= 0.0:
        return (0.0, 0.0)
    y_overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    x_overlap = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    intersection_area = x_overlap * y_overlap
    vertical_overlap = y_overlap / min(left_height, right_height)
    horizontal_overlap = x_overlap / min(left_width, right_width)
    left_center_y = (left_y0 + left_y1) * 0.5
    right_center_y = (right_y0 + right_y1) * 0.5
    vertical_center = max(
        0.0,
        1.0 - abs(left_center_y - right_center_y) / max(left_height, right_height),
    )
    row_alignment = max(vertical_overlap, vertical_center)
    if row_alignment < 0.45 or horizontal_overlap < 0.18:
        return (0.0, intersection_area)
    left_center_x = (left_x0 + left_x1) * 0.5
    right_center_x = (right_x0 + right_x1) * 0.5
    horizontal_center = max(
        0.0,
        1.0 - abs(left_center_x - right_center_x) / max(left_width, right_width),
    )
    return (
        min(
            1.0,
            row_alignment * 0.72 + horizontal_overlap * 0.18 + horizontal_center * 0.10,
        ),
        intersection_area,
    )


def observation_is_covered_by(
    observation: PageObservation,
    coverers: Iterable[PageObservation],
    *,
    single_overlap_ratio: float = 0.35,
    cumulative_overlap_ratio: float = 0.45,
) -> bool:
    area = observation_area(observation)
    if area <= 0.0:
        return True
    covered_area = 0.0
    for coverer in coverers:
        overlap = observation_intersection_area(observation, coverer)
        if overlap <= 0.0:
            continue
        if overlap / area >= single_overlap_ratio:
            return True
        covered_area += overlap
        if covered_area / area >= cumulative_overlap_ratio:
            return True
    return False


def normalize_segment(value: Any) -> Segment | None:
    value_type = type(value)
    if (value_type is not list and value_type is not tuple) or len(value) != 4:
        return None
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )
    except (TypeError, ValueError):
        return None


def numeric_confidence(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ocr_observation_kind(observation: OcrObservation) -> str:
    level = observation.level
    if level == 2:
        return "ocr_textline"
    if level == 3:
        return "ocr_word"
    if level == 4:
        return "ocr_symbol"
    return "ocr"
