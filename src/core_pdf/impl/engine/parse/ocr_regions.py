# SPDX-License-Identifier: AGPL-3.0-only
"""Choosing what to recognize, and batching it into tasks.

Recognizing a whole page at one resolution is rarely the best use of the budget:
embedded images, weakly-covered bands, and tiles each want their own pass. This
module picks those regions, decides when the primary pass was already good enough
to skip the rescue passes, and groups the surviving work into batched tasks.
"""

from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    replace,
)
from typing import (
    Any,
    cast,
)

import numpy

from core_pdf.impl.engine.layout.spatial import (
    SpatialIndex,
    bbox_intersection_area,
)
from core_pdf.impl.engine.model.geometry import (
    bbox_union,
    rect_tuple,
)
from core_pdf.impl.engine.parse.grid_geometry import (
    internal_axis_segments,
    internal_grid_components,
    internal_split_grid_component,
)
from core_pdf.impl.engine.parse.model import (
    MAX_OCR_PIXELS,
    OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY,
    OCR_RESCUE_LARGE_TEXT_HEIGHT,
    OCR_RESCUE_MIN_CONFIDENCE,
    OCR_RESCUE_MIN_WEAK_INK_RATIO,
    OCR_RESCUE_SATURATED_MEAN_INK,
    PRIMARY_OCR_PIXELS,
    VECTOR_PAINT_KINDS,
    CapturedPage,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    WorkPlan,
    internal_bbox_tuple,
    internal_Candidate,
    internal_OCR_RESCUE_DENSE_MIN_CHARACTERS,
    internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE,
    internal_text_utility_stats,
)
from core_pdf.impl.engine.parse.ocr_model import (
    internal_ocr_region_box,
    internal_ocr_region_coverage,
    internal_ocr_region_overlap,
    internal_OcrRegion,
    internal_OcrTask,
    internal_Raster,
    internal_raster_rectangle_page_box,
    internal_RasterRegion,
    internal_RecognitionTrace,
)
from core_pdf.impl.engine.parse.ocr_raster import (
    DirectImageOrientation,
    internal_adaptive_ocr_raster,
    internal_compact_ocr_image,
    internal_decoded_image_raster,
    internal_direct_image_orientation,
    internal_orient_direct_image_raster,
    internal_raster_ink_grid,
    internal_rendered_page_raster,
)
from core_pdf.impl.engine.render.display import RenderOptions
from core_pdf.impl.engine.render.page import compose_page
from core_pdf.impl.engine.spec.s_07_content.page_program import line_coordinate_columns
from core_pdf.impl.runtime.image_cache import ImageCacheKey

OCR_BATCH_MAX_TASKS = 16


OCR_BATCH_MAX_PIXELS = 8_000_000


def internal_ocr_task_groups(
    tasks: tuple[internal_OcrTask, ...],
) -> tuple[tuple[internal_OcrTask, ...], ...]:
    """Create ordered same-raster/mode batches without duplicating image setup."""
    groups: list[tuple[internal_OcrTask, ...]] = []
    current: list[internal_OcrTask] = []
    current_pixels = 0
    for task in tasks:
        pixels = task.rectangle[2] * task.rectangle[3]
        if current and (
            task.image is not current[0].image
            or task.mode != current[0].mode
            or len(current) >= OCR_BATCH_MAX_TASKS
            or current_pixels + pixels > OCR_BATCH_MAX_PIXELS
        ):
            groups.append(tuple(current))
            current = []
            current_pixels = 0
        current.append(task)
        current_pixels += pixels
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def internal_tile_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    if ocr_pass.preprocess == "binary-clean":
        raster = internal_adaptive_ocr_raster(raster)
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    requested_tiles = ocr_pass.tiles if ocr_pass.scope is OcrPassScope.TILES else 1
    tiles = max(1, min(requested_tiles, raster.height))
    if tiles == 1:
        return tuple(
            internal_OcrTask(
                mode=mode,
                image=image,
                rectangle=(0, 0, raster.width, raster.height),
                page_box=page_box,
                resolution=raster.resolution,
                minimum_confidence=ocr_pass.minimum_confidence,
                character_confidence_threshold=ocr_pass.character_confidence_threshold,
                recognize_words=ocr_pass.recognize_words,
                collect_symbols=ocr_pass.collect_symbols,
            )
            for mode in ocr_pass.modes
        )
    overlap = max(24, int(round(raster.resolution * 0.35)))
    base_height = math.ceil(raster.height / tiles)
    tasks = []
    for mode in ocr_pass.modes:
        for tile_index in range(tiles):
            y0 = max(0, tile_index * base_height - overlap)
            y1 = min(raster.height, (tile_index + 1) * base_height + overlap)
            tasks.append(
                internal_OcrTask(
                    mode=mode,
                    image=image,
                    rectangle=(0, y0, raster.width, y1 - y0),
                    page_box=page_box,
                    resolution=raster.resolution,
                    minimum_confidence=ocr_pass.minimum_confidence,
                    character_confidence_threshold=ocr_pass.character_confidence_threshold,
                    recognize_words=ocr_pass.recognize_words,
                    collect_symbols=ocr_pass.collect_symbols,
                )
            )
    return tuple(tasks)


def internal_estimated_text_height(raster: internal_Raster) -> float:
    """Estimate ordinary text-band height from a bounded raster preview.

    Horizontal projections are substantially cheaper than an exploratory OCR
    pass.  Sampling several vertical strips avoids letting table borders or a
    single illustration join otherwise independent text lines.
    """
    pixels = raster.image.array()
    sample_step = max(1, math.ceil(math.sqrt(raster.width * raster.height / 1_000_000)))
    sampled = pixels[::sample_step, ::sample_step]
    gray = sampled[:, :, 0] if raster.image.channels == 1 else numpy.min(sampled[:, :, :3], axis=2)
    background = float(numpy.percentile(gray, 90.0))
    threshold = max(80.0, min(225.0, background - 24.0))
    ink = gray < threshold
    if not numpy.any(ink):
        return 0.0
    strip_count = max(4, min(12, ink.shape[1] // 48))
    heights: list[int] = []
    for strip in numpy.array_split(ink, strip_count, axis=1):
        if strip.shape[1] < 4:
            continue
        required = max(2, int(math.ceil(strip.shape[1] * 0.01)))
        active = numpy.count_nonzero(strip, axis=1) >= required
        # Close a one-row break caused by ascenders, punctuation, or scan noise.
        if len(active) >= 3:
            active[1:-1] |= active[:-2] & active[2:]
        padded = numpy.pad(active.astype(numpy.int8), (1, 1))
        transitions = numpy.diff(padded)
        starts = numpy.flatnonzero(transitions == 1)
        ends = numpy.flatnonzero(transitions == -1)
        for height in ends - starts:
            if 2 <= height <= max(12, sampled.shape[0] // 12):
                heights.append(int(height))
    if len(heights) < 4:
        return 0.0
    values = numpy.asarray(heights, dtype=numpy.float32)
    lower = float(numpy.percentile(values, 25.0))
    upper = float(numpy.percentile(values, 85.0))
    typical = values[(values >= lower) & (values <= upper)]
    return float(numpy.median(typical if len(typical) else values)) * sample_step


def internal_observation_coverage_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    output = numpy.zeros((rows, columns), dtype=numpy.float32)
    for text, confidence, raw_box in zip(
        observations.text,
        observations.confidence,
        observations.bbox,
        strict=True,
    ):
        raw_x0, raw_y0, raw_x1, raw_y1 = internal_bbox_tuple(raw_box)
        box_x0 = max(x0, raw_x0)
        box_y0 = max(y0, raw_y0)
        box_x1 = min(x1, raw_x1)
        box_y1 = min(y1, raw_y1)
        box_width = box_x1 - box_x0
        box_height = box_y1 - box_y0
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        utility = internal_text_utility_stats(text, float(confidence)).utility
        if utility <= 0.0:
            continue
        column_start = max(0, min(columns - 1, int((box_x0 - x0) * columns / width)))
        column_end = max(
            column_start,
            min(columns - 1, math.ceil((box_x1 - x0) * columns / width) - 1),
        )
        row_start = max(0, min(rows - 1, int((y1 - box_y1) * rows / height)))
        row_end = max(
            row_start,
            min(rows - 1, math.ceil((y1 - box_y0) * rows / height) - 1),
        )
        box_area = box_width * box_height
        for row in range(row_start, row_end + 1):
            cell_y0 = y1 - (row + 1) * height / rows
            cell_y1 = y1 - row * height / rows
            overlap_y = max(0.0, min(box_y1, cell_y1) - max(box_y0, cell_y0))
            if overlap_y <= 0.0:
                continue
            for column in range(column_start, column_end + 1):
                cell_x0 = x0 + column * width / columns
                cell_x1 = x0 + (column + 1) * width / columns
                overlap_x = max(0.0, min(box_x1, cell_x1) - max(box_x0, cell_x0))
                if overlap_x > 0.0:
                    output[row, column] += utility * overlap_x * overlap_y / box_area
    return output.reshape(-1)


def internal_observation_utility_grid(
    observations: ObservationBatch,
    page_box: tuple[float, float, float, float],
    rows: int,
    columns: int,
) -> numpy.ndarray[Any, Any]:
    """Assign each observation to one cell for stable weak-region ranking."""
    if not len(observations):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    x0, y0, x1, y1 = page_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    centers_x = (observations.bbox[:, 0] + observations.bbox[:, 2]) * 0.5
    centers_y = (observations.bbox[:, 1] + observations.bbox[:, 3]) * 0.5
    inside = (centers_x >= x0) & (centers_x <= x1) & (centers_y >= y0) & (centers_y <= y1)
    if not numpy.any(inside):
        return numpy.zeros(rows * columns, dtype=numpy.float32)
    centers_x = centers_x[inside]
    centers_y = centers_y[inside]
    columns_by_observation = numpy.clip(
        ((centers_x - x0) * columns / width).astype(numpy.int64),
        0,
        columns - 1,
    )
    rows_by_observation = numpy.clip(
        ((y1 - centers_y) * rows / height).astype(numpy.int64),
        0,
        rows - 1,
    )
    utility = numpy.fromiter(
        (
            internal_text_utility_stats(text, float(confidence)).utility
            for text, confidence in zip(
                (
                    text
                    for text, selected in zip(observations.text, inside, strict=True)
                    if selected
                ),
                observations.confidence[inside],
                strict=True,
            )
        ),
        dtype=numpy.float32,
        count=int(numpy.count_nonzero(inside)),
    )
    return numpy.bincount(
        rows_by_observation * columns + columns_by_observation,
        weights=utility,
        minlength=rows * columns,
    ).astype(numpy.float32, copy=False)


def internal_weak_region_grid_shape(
    raster: internal_Raster,
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[int, int]:
    rows = max(1, min(ocr_pass.tiles, raster.height))
    columns = max(1, min(ocr_pass.region_columns, raster.width))
    if len(primary) >= 40:
        rows = min(rows, 6)
        columns = min(columns, 3)
    return rows, columns


def internal_weak_region_rectangles(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> tuple[tuple[int, int, int, int], ...]:
    """Find visually occupied cells where the primary OCR recovered little text."""
    rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
    ink = internal_raster_ink_grid(raster, rows, columns)
    utility = internal_observation_utility_grid(primary, page_box, rows, columns)
    expected_utility = float(numpy.sum(utility)) / max(1, rows * columns)
    utility_limit = max(4.0, expected_utility * 0.45)
    eligible = numpy.flatnonzero((ink >= 0.01) & (utility < utility_limit))
    if not len(eligible):
        return ()
    priority = ink[eligible] / (1.0 + utility[eligible] * 0.05)
    region_limit = ocr_pass.max_regions
    if len(primary) >= 40:
        region_limit = max(1, region_limit // 2)
        region_limit = min(region_limit, 8)
    ranked = eligible[numpy.argsort(priority)[::-1][:region_limit]]
    # Tesseract's sparse-text layout pass scans connected components that can cross
    # the requested rectangle.  A narrow horizontal margin can therefore make
    # Leptonica reject a component as being outside the active rectangle.  Keep a
    # generous, resolution-scaled margin so region boundaries do not bisect glyphs
    # or text lines.
    overlap_x = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.width // columns - 1) // 2),
    )
    overlap_y = min(
        max(48, int(round(raster.resolution * 0.20))),
        max(0, (raster.height // rows - 1) // 2),
    )
    rectangles: list[tuple[int, int, int, int]] = []
    for raw_cell in ranked:
        cell = int(raw_cell)
        row, column = divmod(cell, columns)
        cell_x0 = column * raster.width // columns
        cell_x1 = (column + 1) * raster.width // columns
        cell_y0 = row * raster.height // rows
        cell_y1 = (row + 1) * raster.height // rows
        rectangle_x0 = max(0, cell_x0 - overlap_x)
        rectangle_x1 = min(raster.width, cell_x1 + overlap_x)
        rectangle_y0 = max(0, cell_y0 - overlap_y)
        rectangle_y1 = min(raster.height, cell_y1 + overlap_y)
        rectangles.append(
            (
                rectangle_x0,
                rectangle_y0,
                rectangle_x1 - rectangle_x0,
                rectangle_y1 - rectangle_y0,
            )
        )
    return tuple(rectangles)


def internal_weak_region_tasks(
    raster: internal_Raster,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    compact_image: bool | str = False,
) -> tuple[internal_OcrTask, ...]:
    """Create OCR tasks for weak regions in an already materialized raster."""
    image = (
        internal_compact_ocr_image(raster.image, grayscale=compact_image == "grayscale")
        if compact_image
        else raster.image
    )
    return tuple(
        internal_OcrTask(
            mode=mode,
            image=image,
            rectangle=rectangle,
            page_box=page_box,
            resolution=raster.resolution,
            minimum_confidence=ocr_pass.minimum_confidence,
            character_confidence_threshold=ocr_pass.character_confidence_threshold,
            recognize_words=ocr_pass.recognize_words,
            collect_symbols=ocr_pass.collect_symbols,
        )
        for mode in ocr_pass.modes
        for rectangle in internal_weak_region_rectangles(raster, page_box, ocr_pass, primary)
    )


@dataclass(frozen=True, slots=True)
class internal_RescueCoverage:
    raster_count: int = 0
    cell_count: int = 0
    ink_cells: int = 0
    weak_cells: int = 0
    ink: float = 0.0
    weak_ink: float = 0.0

    @property
    def mean_ink(self) -> float:
        return self.ink / max(1, self.cell_count)

    @property
    def weak_ink_ratio(self) -> float:
        return self.weak_ink / max(1e-9, self.ink)

    def as_record(self) -> dict[str, int | float]:
        return {
            "raster_count": self.raster_count,
            "cell_count": self.cell_count,
            "ink_cells": self.ink_cells,
            "weak_cells": self.weak_cells,
            "mean_ink": self.mean_ink,
            "weak_ink_ratio": self.weak_ink_ratio,
        }


def internal_adaptive_rescue_coverage(
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
) -> internal_RescueCoverage:
    """Measure ink not spatially explained by the primary OCR observations."""
    raster_count = 0
    cell_count = 0
    ink_cells = 0
    weak_cells = 0
    total_ink = 0.0
    weak_ink = 0.0
    seen: set[tuple[int, tuple[float, float, float, float], int]] = set()
    for task in source_tasks:
        key = (id(task.image), task.page_box, task.resolution)
        if key in seen:
            continue
        seen.add(key)
        raster = internal_Raster(task.image, task.resolution)
        rows, columns = internal_weak_region_grid_shape(raster, ocr_pass, primary)
        ink = internal_raster_ink_grid(raster, rows, columns)
        coverage = internal_observation_coverage_grid(primary, task.page_box, rows, columns)
        utility_limit = max(4.0, float(numpy.sum(coverage)) / (rows * columns) * 0.45)
        occupied = ink >= 0.01
        weak = occupied & (coverage < utility_limit)
        raster_count += 1
        cell_count += rows * columns
        ink_cells += int(numpy.count_nonzero(occupied))
        weak_cells += int(numpy.count_nonzero(weak))
        total_ink += float(numpy.sum(ink, dtype=numpy.float64))
        weak_ink += float(numpy.sum(ink[weak], dtype=numpy.float64))
    return internal_RescueCoverage(
        raster_count=raster_count,
        cell_count=cell_count,
        ink_cells=ink_cells,
        weak_cells=weak_cells,
        ink=total_ink,
        weak_ink=weak_ink,
    )


def internal_primary_text_is_sufficient(candidate: internal_Candidate) -> bool:
    """Return whether a sparse primary result is already large and trustworthy.

    Resolution escalation cannot add detail to text that is already comfortably
    sampled. Keep this decision shared by the adaptive rescue and subsequent
    full-page fallbacks so the latter cannot repeat work the former rejected.
    """
    metrics = candidate.metrics
    return (
        metrics.characters < 32
        and metrics.median_text_height >= OCR_RESCUE_LARGE_TEXT_HEIGHT
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
    )


def internal_adaptive_rescue_decision(
    candidate: internal_Candidate,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
) -> tuple[bool, dict[str, object]]:
    """Decide whether another raster pass has enough unresolved visual evidence."""
    metrics = candidate.metrics
    coverage_pass = replace(
        ocr_pass,
        scope=OcrPassScope.WEAK_REGIONS,
        tiles=max(6, ocr_pass.tiles),
        region_columns=max(3, ocr_pass.region_columns),
        max_regions=max(8, ocr_pass.max_regions),
    )
    coverage = internal_adaptive_rescue_coverage(
        source_tasks,
        coverage_pass,
        candidate.observations,
    )
    reason = "unresolved-ink"
    run = True
    if internal_primary_text_is_sufficient(candidate):
        run = False
        reason = "primary-text-already-large"
    elif coverage.mean_ink >= OCR_RESCUE_SATURATED_MEAN_INK and (
        (metrics.characters >= 1_000 and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE)
        or (
            metrics.characters >= internal_OCR_RESCUE_DENSE_MIN_CHARACTERS
            and metrics.mean_confidence >= internal_OCR_RESCUE_DENSE_MIN_CONFIDENCE
        )
    ):
        # A nearly solid source gives the coarse ink grid no useful localization
        # signal.  Reprocessing arbitrary cells cannot target missing text.
        run = False
        reason = "ink-map-saturated"
    elif (
        metrics.characters >= 300
        and metrics.mean_confidence >= OCR_RESCUE_MIN_CONFIDENCE
        and coverage.raster_count
        and coverage.weak_ink_ratio < OCR_RESCUE_MIN_WEAK_INK_RATIO
        and (metrics.characters >= 600 or coverage.weak_ink_ratio == 0.0)
    ):
        run = False
        reason = "primary-covers-ink"
    return run, {
        "run": run,
        "reason": reason,
        "characters": metrics.characters,
        "mean_confidence": metrics.mean_confidence,
        "median_text_height": metrics.median_text_height,
        **coverage.as_record(),
    }


def internal_page_image_regions(
    capture: CapturedPage,
    *,
    minimum_area_ratio: float,
    max_pixels: int = MAX_OCR_PIXELS,
    maximum_axis_deviation: float = 1e-5,
    upscale: bool = True,
) -> tuple[internal_RasterRegion, ...]:
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    regions: list[internal_RasterRegion] = []
    for image in getattr(capture, "drawings", ()):
        if getattr(image, "kind", None) != "image":
            continue
        orientation = internal_direct_image_orientation(
            image,
            maximum_axis_deviation=maximum_axis_deviation,
        )
        if orientation is None:
            continue
        box = rect_tuple(getattr(image, "rect", None))
        if box is None:
            continue
        clipped = (
            max(0.0, box[0]),
            max(0.0, box[1]),
            min(page_width, box[2]),
            min(page_height, box[3]),
        )
        # A decoded source raster represents the full image. If the image is clipped by
        # the page, mapping that full raster onto the clipped rectangle would compress
        # its OCR coordinates. Let the page compositor produce the correct crop instead.
        clip_tolerance = max(2.0, max(page_width, page_height) * 0.005)
        if any(
            abs(float(original) - clipped_value) > clip_tolerance
            for original, clipped_value in zip(box, clipped, strict=True)
        ):
            continue
        display_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if display_area / page_area < minimum_area_ratio:
            continue
        raster = internal_decoded_image_raster(
            image,
            display_area,
            image_cache=getattr(getattr(capture.page, "document", None), "image_cache", None),
            max_pixels=max_pixels,
            upscale=upscale,
        )
        if raster is not None:
            oriented = raster
            if orientation is not DirectImageOrientation.IDENTITY:
                source = getattr(image, "image_source", None)
                source_key = getattr(source, "cache_key", None)
                if not isinstance(source_key, tuple):
                    source_key = ("image", id(image))
                oriented_key = ImageCacheKey(
                    "ocr-oriented-raster",
                    tuple(source_key),
                    (orientation.value, float(display_area), int(max_pixels), upscale),
                )
                cache = getattr(getattr(capture.page, "document", None), "image_cache", None)
                if cache is not None:
                    cached_oriented = cache.get_or_create(
                        oriented_key,
                        lambda image=image, raster=raster, orientation=orientation: (
                            internal_orient_direct_image_raster(
                                image,
                                raster,
                                orientation=orientation,
                            )
                        ),
                    )
                    if isinstance(cached_oriented, internal_Raster):
                        oriented = cached_oriented
                else:
                    oriented = internal_orient_direct_image_raster(
                        image,
                        raster,
                        orientation=orientation,
                    )
            regions.append(
                internal_RasterRegion(
                    oriented,
                    clipped,
                )
            )
    return tuple(regions)


def internal_dominant_image_region(
    capture: CapturedPage,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    upscale: bool = True,
) -> internal_RasterRegion | None:
    def box_area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.65,
        max_pixels=max_pixels,
        upscale=upscale,
    )
    substantial = tuple(
        region for region in regions if region.raster.width * region.raster.height >= 4_096
    )
    if not substantial:
        return None
    if len(substantial) > 1:
        largest = max(substantial, key=lambda region: box_area(region.page_box))
        largest_area = max(1.0, box_area(largest.page_box))
        overlapping = sum(
            max(
                0.0,
                min(region.page_box[2], largest.page_box[2])
                - max(region.page_box[0], largest.page_box[0]),
            )
            * max(
                0.0,
                min(region.page_box[3], largest.page_box[3])
                - max(region.page_box[1], largest.page_box[1]),
            )
            / largest_area
            >= 0.90
            for region in substantial
            if region is not largest
        )
        if overlapping:
            return None
    return max(substantial, key=lambda region: region.raster.width * region.raster.height)


OCR_REGION_INITIAL_COUNT = 8


OCR_REGION_MAX_COUNT = 16


OCR_REGION_INITIAL_AREA_RATIO = 0.25


OCR_REGION_MAX_AREA_RATIO = 0.60


OCR_DIRECT_REGION_MIN_COVERAGE = 0.65


def internal_merge_ocr_regions(regions: list[internal_OcrRegion]) -> tuple[internal_OcrRegion, ...]:
    merged: list[internal_OcrRegion] = []
    merged_areas: list[float] = []
    for region in sorted(regions, key=lambda item: (-item.score, item.page_box)):
        region_box = region.page_box
        region_area = max(0.0, region_box[2] - region_box[0]) * max(
            0.0, region_box[3] - region_box[1]
        )
        match = None
        for index, existing in enumerate(merged):
            existing_box = existing.page_box
            smaller = min(merged_areas[index], region_area)
            if not smaller:
                continue
            intersection_width = max(
                0.0, min(existing_box[2], region_box[2]) - max(existing_box[0], region_box[0])
            )
            intersection_height = max(
                0.0, min(existing_box[3], region_box[3]) - max(existing_box[1], region_box[1])
            )
            if intersection_width * intersection_height >= smaller * 0.35:
                match = index
                break
        if match is None:
            merged.append(region)
            merged_areas.append(region_area)
            continue
        existing = merged[match]
        existing_box = existing.page_box
        merged_box = (
            min(existing_box[0], region_box[0]),
            min(existing_box[1], region_box[1]),
            max(existing_box[2], region_box[2]),
            max(existing_box[3], region_box[3]),
        )
        merged[match] = internal_OcrRegion(
            merged_box,
            max(existing.score, region.score) + min(existing.score, region.score) * 0.15,
            tuple(dict.fromkeys((*existing.reasons, *region.reasons))),
        )
        merged_areas[match] = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
    return tuple(sorted(merged, key=lambda item: (-item.score, item.page_box)))


def internal_candidate_ocr_regions(capture: CapturedPage) -> tuple[internal_OcrRegion, ...]:
    """Select likely OCR areas using capture-time geometry only.

    This deliberately does not render a preview image.  Native text, image bounds,
    captured paths, and grid lines are already available from the canonical page IR.
    """
    cache = getattr(capture.page, "extraction_cache", None)
    cache_key = "ocr_candidate_regions_v1"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, tuple) and all(
            isinstance(item, internal_OcrRegion) for item in cached
        ):
            return cached

    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    page_area = max(1.0, page_width * page_height)
    padding = max(6.0, min(36.0, min(page_width, page_height) * 0.01))
    candidates: list[internal_OcrRegion] = []

    for box in getattr(capture.evidence, "image_boxes", ()):
        image_box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in box),
        )
        padded = internal_ocr_region_box(
            image_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            candidates.append(internal_OcrRegion(padded, 5.0, ("image",)))

    native = getattr(capture, "observations", ObservationBatch.empty())
    native_boxes = tuple(tuple(float(value) for value in box) for box in native.bbox)
    native_index = SpatialIndex.from_boxes(native_boxes) if native_boxes else None

    def native_overlap(box: tuple[float, float, float, float]) -> float:
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        if native_index is not None:
            return min(
                1.0,
                sum(
                    bbox_intersection_area(box, hit.bbox)
                    for hit in native_index.intersecting_hits(box)
                )
                / area,
            )
        return min(
            1.0,
            sum(
                max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
                * max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
                for other in native_boxes
            )
            / area,
        )

    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in {"fill", "fillstroke", "stroke"}:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        drawing_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if drawing_area <= 0.0 or drawing_area >= page_area * 0.80:
            continue
        uncovered = native_overlap(box) < 0.25
        if uncovered and getattr(drawing, "kind", None) in {"fill", "fillstroke"}:
            padded = internal_ocr_region_box(
                box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.5, ("uncovered-vector",)))

    if hasattr(capture, "grid_lines"):
        horizontal, vertical = internal_axis_segments(capture)
    else:
        horizontal = numpy.empty((0, 3), dtype=numpy.float32)
        vertical = numpy.empty((0, 3), dtype=numpy.float32)
    for component_horizontal, component_vertical in internal_grid_components(horizontal, vertical):
        x0 = min(float(component_horizontal[:, 0].min()), float(component_vertical[:, 0].min()))
        y0 = min(float(component_horizontal[:, 2].min()), float(component_vertical[:, 1].min()))
        x1 = max(float(component_horizontal[:, 1].max()), float(component_vertical[:, 0].max()))
        y1 = max(float(component_horizontal[:, 2].max()), float(component_vertical[:, 2].max()))
        for split_horizontal, split_vertical in internal_split_grid_component(
            component_horizontal,
            component_vertical,
        ):
            split_box = (
                min(float(split_horizontal[:, 0].min()), float(split_vertical[:, 0].min())),
                min(float(split_horizontal[:, 2].min()), float(split_vertical[:, 1].min())),
                max(float(split_horizontal[:, 1].max()), float(split_vertical[:, 0].max())),
                max(float(split_horizontal[:, 2].max()), float(split_vertical[:, 2].max())),
            )
            padded = internal_ocr_region_box(
                split_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None and (
                (padded[2] - padded[0]) * (padded[3] - padded[1]) < page_area * 0.45
            ):
                candidates.append(internal_OcrRegion(padded, 4.0, ("grid",)))
        if not component_horizontal.size or not component_vertical.size:
            continue
        component_box = (x0, y0, x1, y1)
        component_area = (x1 - x0) * (y1 - y0)
        if component_area < page_area * 0.45 and native_overlap(component_box) < 0.25:
            padded = internal_ocr_region_box(
                component_box,
                page_width=page_width,
                page_height=page_height,
                padding=padding,
            )
            if padded is not None:
                candidates.append(internal_OcrRegion(padded, 3.0, ("grid-labels",)))

    columns = 6
    rows = max(2, min(8, int(round(columns * page_height / max(1.0, page_width)))))
    vector_density = numpy.zeros(rows * columns, dtype=numpy.float32)
    for drawing in getattr(capture, "drawings", ()):
        if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
            continue
        box = rect_tuple(getattr(drawing, "rect", None))
        if box is None:
            continue
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        vector_density[row * columns + column] += 1.0
    grid_lines = getattr(capture, "grid_lines", ())
    if len(grid_lines):
        # Bin every grid line at once.  Iterating the capture would rebuild one
        # Python object per line just to read its four coordinates back out.
        line_x0, line_y0, line_x1, line_y1 = line_coordinate_columns(grid_lines)
        line_columns = numpy.clip(
            ((line_x0 + line_x1) * 0.5 * columns / max(1.0, page_width)).astype(numpy.int64),
            0,
            columns - 1,
        )
        line_rows = numpy.clip(
            ((line_y0 + line_y1) * 0.5 * rows / max(1.0, page_height)).astype(numpy.int64),
            0,
            rows - 1,
        )
        vector_density += (
            numpy.bincount(
                line_rows * columns + line_columns,
                minlength=rows * columns,
            ).astype(numpy.float32)
            * 0.5
        )

    native_counts = numpy.zeros(rows * columns, dtype=numpy.float32)
    for text, raw_box in zip(native.text, native.bbox, strict=True):
        box = internal_bbox_tuple(raw_box)
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        column = min(columns - 1, max(0, int(center_x * columns / max(1.0, page_width))))
        row = min(rows - 1, max(0, int(center_y * rows / max(1.0, page_height))))
        native_counts[row * columns + column] += sum(not char.isspace() for char in text)

    for cell, density in enumerate(vector_density):
        if density <= 0.0:
            continue
        row, column = divmod(cell, columns)
        cell_box = (
            column * page_width / columns,
            row * page_height / rows,
            (column + 1) * page_width / columns,
            (row + 1) * page_height / rows,
        )
        sparse = native_counts[cell] < 8.0
        header_band = row in {0, rows - 1} and native_counts[cell] < 24.0
        if not sparse and not header_band:
            continue
        padded = internal_ocr_region_box(
            cell_box,
            page_width=page_width,
            page_height=page_height,
            padding=padding,
        )
        if padded is not None:
            reasons = ["vector-density"]
            if sparse:
                reasons.append("sparse-label")
            if header_band:
                reasons.append("header-band")
            candidates.append(
                internal_OcrRegion(
                    padded,
                    1.5 + min(2.0, float(density) / 8.0),
                    tuple(reasons),
                )
            )

    if (
        capture.evidence.vector_complexity >= 180
        and capture.evidence.text_coverage < 0.05
        and (not native_boxes or len(native_boxes) >= 8)
    ):
        # Component labels are often isolated from the larger paths they
        # annotate. Use finer cells for these vector-only pages so the region
        # budget can select several label clusters instead of one broad artwork
        # box. The existing coarse density pass remains responsible for larger
        # diagram areas.
        label_columns = 12
        label_rows = max(
            4,
            min(12, int(round(label_columns * page_height / max(1.0, page_width)))),
        )
        label_density = numpy.zeros(label_rows * label_columns, dtype=numpy.float32)
        label_boxes: list[list[tuple[float, float, float, float]]] = [
            [] for _ in range(label_rows * label_columns)
        ]
        for drawing in getattr(capture, "drawings", ()):
            if getattr(drawing, "kind", None) not in VECTOR_PAINT_KINDS:
                continue
            box = rect_tuple(getattr(drawing, "rect", None))
            if box is None:
                continue
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5
            column = min(
                label_columns - 1,
                max(0, int(center_x * label_columns / max(1.0, page_width))),
            )
            row = min(
                label_rows - 1,
                max(0, int(center_y * label_rows / max(1.0, page_height))),
            )
            label_density[row * label_columns + column] += 1.0
            label_boxes[row * label_columns + column].append(box)

        for cell, density in enumerate(label_density):
            if density <= 0.0:
                continue
            row, column = divmod(cell, label_columns)
            cell_box = (
                column * page_width / label_columns,
                row * page_height / label_rows,
                (column + 1) * page_width / label_columns,
                (row + 1) * page_height / label_rows,
            )
            component_boxes = label_boxes[cell]
            optional_component_box = bbox_union(component_boxes)
            assert optional_component_box is not None
            component_box = optional_component_box
            component_area = max(0.0, component_box[2] - component_box[0]) * max(
                0.0, component_box[3] - component_box[1]
            )
            label_padding = max(
                padding,
                min(72.0, min(page_width, page_height) * 0.03),
            )
            candidate_box = component_box if component_area <= page_area * 0.08 else cell_box
            padded = internal_ocr_region_box(
                candidate_box,
                page_width=page_width,
                page_height=page_height,
                padding=label_padding if candidate_box == component_box else padding,
            )
            if padded is not None:
                candidates.append(
                    internal_OcrRegion(
                        padded,
                        1.0 + min(3.0, float(density) / 8.0),
                        ("vector-label-density", "vector-label-neighborhood")
                        if candidate_box == component_box
                        else ("vector-label-density",),
                    )
                )

    regions = internal_merge_ocr_regions(candidates)
    if not regions:
        regions = (
            internal_OcrRegion(
                (0.0, 0.0, page_width, page_height),
                0.0,
                ("page-fallback",),
            ),
        )
    if cache is not None:
        cache[cache_key] = regions
    return regions


def internal_has_distributed_outline_text(capture: CapturedPage) -> bool:
    """Detect pages whose text was converted into many small filled vector paths."""
    page_width = float(capture.page.width)
    page_height = float(capture.page.height)
    max_width = max(24.0, page_width * 0.04)
    max_height = max(24.0, page_height * 0.04)
    boxes = tuple(
        box
        for drawing in getattr(capture, "drawings", ())
        if getattr(drawing, "kind", None) in {"fill", "fillstroke"}
        and (box := rect_tuple(getattr(drawing, "rect", None))) is not None
        and 0.0 < box[2] - box[0] <= max_width
        and 0.0 < box[3] - box[1] <= max_height
    )
    if len(boxes) < 200:
        return False
    bounds = bbox_union(boxes)
    assert bounds is not None
    width_ratio = (bounds[2] - bounds[0]) / max(1.0, page_width)
    height_ratio = (bounds[3] - bounds[1]) / max(1.0, page_height)
    return width_ratio >= 0.60 and height_ratio >= 0.60


def internal_direct_scan_allowed(capture: CapturedPage, plan: WorkPlan) -> bool:
    """Decide whether a page-scope pass may OCR the decoded scan itself.

    Rendering a scanned page through the compositor resamples the scan a second
    time at whatever scale the pass chose, which is strictly worse than reading
    its own pixels.  The rendered page is still required whenever the page holds
    content the dominant image does not cover.
    """
    evidence = capture.evidence
    if not plan.allow_direct_image_ocr:
        return False
    if evidence.visible_native_characters >= 10 or not evidence.image_count:
        return True
    # No native text and one image covering the page: the image *is* the page, so
    # nothing is lost by reading it directly.  Any weaker signal keeps the render.
    return bool(evidence.full_page_image) and not evidence.visible_native_characters


def internal_ocr_region_batch(
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    expanded: bool,
    page_area: float,
) -> tuple[internal_OcrRegion, ...]:
    count_limit = max(
        ocr_pass.max_regions,
        OCR_REGION_MAX_COUNT if expanded else OCR_REGION_INITIAL_COUNT,
    )
    area_limit = OCR_REGION_MAX_AREA_RATIO if expanded else OCR_REGION_INITIAL_AREA_RATIO
    selected: list[internal_OcrRegion] = []
    area = 0.0
    page_area = max(1.0, page_area)
    if page_area <= 0.0:
        return ()
    for region in regions:
        if len(selected) >= count_limit:
            break
        if selected and area + region.area > page_area * area_limit:
            continue
        selected.append(region)
        area += region.area
    return tuple(selected)


def internal_candidate_region_tasks(
    capture: CapturedPage,
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    rendered: Any | None,
    compact_image: bool | str,
    trace: internal_RecognitionTrace | None = None,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    direct_regions = internal_page_image_regions(
        capture,
        minimum_area_ratio=0.02,
        max_pixels=ocr_pass.pixel_budget,
    )
    if not direct_regions:
        dominant = internal_dominant_image_region(
            capture,
            max_pixels=ocr_pass.pixel_budget,
        )
        if dominant is not None:
            direct_regions = (dominant,)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    rendered_boxes: list[tuple[float, float, float, float]] = []
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    direct_region_index = (
        SpatialIndex(((index, region.page_box) for index, region in enumerate(direct_regions)))
        if len(direct_regions) > 4
        else None
    )
    for region in regions:
        raster: internal_Raster | None
        direct_candidates = (
            (direct_regions[index] for index in direct_region_index.intersecting(region.page_box))
            if direct_region_index is not None
            else iter(direct_regions)
        )
        matching_direct = tuple(
            candidate
            for candidate in direct_candidates
            # Region proposals include padding, so a source image need not cover the
            # entire box. It must still cover most of the requested target: otherwise
            # a narrow banner can incorrectly replace a broad compositor render.
            if internal_ocr_region_coverage(region.page_box, candidate.page_box)
            >= OCR_DIRECT_REGION_MIN_COVERAGE
        )
        layered_scan = any(
            internal_ocr_region_overlap(left.page_box, right.page_box) >= 0.90
            for index, left in enumerate(matching_direct)
            for right in matching_direct[:index]
        )
        direct = (
            None
            if layered_scan
            else max(
                matching_direct,
                key=lambda candidate: candidate.raster.width * candidate.raster.height,
                default=None,
            )
        )
        if direct is not None:
            raster = direct.raster
            raster_box = direct.page_box
        else:
            if rendered is None:
                rendered = compose_page(
                    capture.page,
                    RenderOptions(include_text=ocr_pass.include_native_text),
                    page_program=capture.program,
                )
            raster = internal_rendered_page_raster(
                capture,
                ocr_pass.scale,
                crop=region.page_box,
                rendered=rendered,
                cache=True,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                trace=trace,
            )
            raster_box = region.page_box
        if raster is None:
            continue
        rendered_boxes.append(raster_box)
        raster_pixels += raster.width * raster.height
        full_page_region = (
            ocr_pass.scope is OcrPassScope.PAGE
            and len(regions) == 1
            and region.area
            >= getattr(
                getattr(capture, "evidence", None),
                "page_area",
                float(capture.page.width) * float(capture.page.height),
            )
            * 0.75
            and internal_ocr_region_coverage(
                region.page_box,
                (0.0, 0.0, float(capture.page.width), float(capture.page.height)),
            )
            >= 0.90
            and getattr(getattr(capture, "evidence", None), "vector_complexity", 0)
            >= OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY
            and 4_000_000 <= raster.width * raster.height <= PRIMARY_OCR_PIXELS
        )
        tile_count = ocr_pass.parallel_tiles if full_page_region else 1
        task_pass = (
            replace(
                region_pass,
                tiles=max(1, tile_count),
                recognize_words=True,
            )
            if layered_scan
            else replace(region_pass, tiles=max(1, tile_count))
        )
        tasks.extend(
            internal_tile_tasks(
                raster,
                raster_box,
                task_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(rendered_boxes)


def internal_high_resolution_weak_region_tasks(
    capture: CapturedPage,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    rendered: Any | None,
    compact_image: bool | str,
    trace: internal_RecognitionTrace | None = None,
) -> tuple[
    tuple[internal_OcrTask, ...], int, Any | None, tuple[tuple[float, float, float, float], ...]
]:
    """Rasterize only weak cells at rescue resolution instead of the whole page."""
    source_rasters: dict[tuple[int, tuple[float, float, float, float], int], internal_Raster] = {}
    for task in source_tasks:
        source_rasters.setdefault(
            (id(task.image), task.page_box, task.resolution),
            internal_Raster(task.image, task.resolution),
        )
    weak_regions: list[internal_OcrRegion] = []
    for (_, page_box, _), source_raster in source_rasters.items():
        for rectangle in internal_weak_region_rectangles(
            source_raster,
            page_box,
            ocr_pass,
            primary,
        ):
            weak_regions.append(
                internal_OcrRegion(
                    internal_raster_rectangle_page_box(source_raster, page_box, rectangle),
                    1.0,
                    ("adaptive-weak-region",),
                )
            )
    regions = internal_merge_ocr_regions(weak_regions)
    if not regions:
        return (), 0, rendered, ()
    if rendered is None:
        rendered = compose_page(
            capture.page,
            RenderOptions(include_text=ocr_pass.include_native_text),
            page_program=capture.program,
        )
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    tasks: list[internal_OcrTask] = []
    raster_pixels = 0
    boxes: list[tuple[float, float, float, float]] = []
    for region in regions:
        raster = internal_rendered_page_raster(
            capture,
            ocr_pass.scale,
            crop=region.page_box,
            rendered=rendered,
            cache=True,
            max_pixels=ocr_pass.pixel_budget,
            include_native_text=ocr_pass.include_native_text,
            trace=trace,
        )
        if raster is None:
            continue
        boxes.append(region.page_box)
        raster_pixels += raster.width * raster.height
        tasks.extend(
            internal_tile_tasks(
                raster,
                region.page_box,
                region_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks), raster_pixels, rendered, tuple(boxes)
