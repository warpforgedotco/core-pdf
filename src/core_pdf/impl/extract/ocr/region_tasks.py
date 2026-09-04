# SPDX-License-Identifier: AGPL-3.0-only
"""Materialize proposed OCR regions into bounded recognition tasks."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy

from core_pdf.impl.extract.contracts import (
    OCR_PARALLEL_TILE_MIN_VECTOR_COMPLEXITY,
    PRIMARY_OCR_PIXELS,
    CapturedPage,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    WorkPlan,
)
from core_pdf.impl.extract.ocr.raster import (
    internal_adaptive_ocr_raster,
    internal_compact_ocr_image,
    internal_raster_ink_grid,
    internal_rendered_page_raster,
)
from core_pdf.impl.extract.ocr.regions import (
    OCR_DIRECT_REGION_MIN_COVERAGE,
    internal_dominant_image_region,
    internal_merge_ocr_regions,
    internal_page_image_regions,
)
from core_pdf.impl.extract.ocr.types import (
    internal_OcrRegion,
    internal_OcrTask,
    internal_Raster,
    internal_raster_rectangle_page_box,
)
from core_pdf.impl.extract.quality import internal_text_utility_stats
from core_pdf.impl.model.geometry import (
    overlap_ratio_min_exact as internal_ocr_region_overlap,
)
from core_pdf.impl.model.geometry import (
    overlap_ratio_of as internal_ocr_region_coverage,
)
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from core_pdf.impl.runtime.array_views import finite_median

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
        rectangles: tuple[tuple[int, int, int, int], ...] = ((0, 0, raster.width, raster.height),)
    else:
        overlap = max(24, int(round(raster.resolution * 0.35)))
        base_height = math.ceil(raster.height / tiles)
        rectangles = tuple(
            (
                0,
                (y0 := max(0, tile_index * base_height - overlap)),
                raster.width,
                min(raster.height, (tile_index + 1) * base_height + overlap) - y0,
            )
            for tile_index in range(tiles)
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
        for rectangle in rectangles
    )


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
    return finite_median(typical if len(typical) else values) * sample_step


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
    # the requested rectangle. A narrow horizontal margin can therefore make
    # Leptonica reject a component as being outside the active rectangle.
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


def internal_candidate_region_tasks(
    capture: CapturedPage,
    regions: tuple[internal_OcrRegion, ...],
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str,
) -> tuple[internal_OcrTask, ...]:
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
    rendered = None
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    for region in regions:
        raster: internal_Raster | None
        matching_direct = tuple(
            candidate
            for candidate in direct_regions
            # Region proposals include padding, so a source image need not cover the
            # entire box. It must still cover most of the requested target.
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
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
            )
            raster_box = region.page_box
        if raster is None:
            continue
        full_page_region = (
            ocr_pass.scope is OcrPassScope.PAGE
            and len(regions) == 1
            and region.area
            >= getattr(
                getattr(capture, "evidence", None),
                "page_area",
                capture.width * capture.height,
            )
            * 0.75
            and internal_ocr_region_coverage(
                region.page_box,
                (0.0, 0.0, capture.width, capture.height),
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
    return tuple(tasks)


def internal_high_resolution_weak_region_tasks(
    capture: CapturedPage,
    source_tasks: tuple[internal_OcrTask, ...],
    ocr_pass: OcrPass,
    primary: ObservationBatch,
    *,
    compact_image: bool | str,
) -> tuple[internal_OcrTask, ...]:
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
        return ()
    rendered = compose_page(
        capture.page,
        RenderOptions(include_text=ocr_pass.include_native_text),
        page_program=capture.program,
    )
    region_pass = replace(ocr_pass, scope=OcrPassScope.TILES, tiles=1)
    tasks: list[internal_OcrTask] = []
    for region in regions:
        raster = internal_rendered_page_raster(
            capture,
            ocr_pass.scale,
            crop=region.page_box,
            rendered=rendered,
            max_pixels=ocr_pass.pixel_budget,
            include_native_text=ocr_pass.include_native_text,
        )
        if raster is None:
            continue
        tasks.extend(
            internal_tile_tasks(
                raster,
                region.page_box,
                region_pass,
                compact_image=compact_image,
            )
        )
    return tuple(tasks)
