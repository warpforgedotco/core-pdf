# SPDX-License-Identifier: AGPL-3.0-only
"""Tesseract integration: rasterization, recognition, and rescue passes."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy

from core_pdf.impl.extract.capture import internal_promoted_hidden_observations
from core_pdf.impl.extract.contracts import (
    HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
    HIDDEN_TEXT_VERIFY_PIXELS,
    MAX_OCR_PIXELS,
    MAX_OCR_RASTER_BYTES,
    OCR_PREFLIGHT_PIXELS,
    PRIMARY_OCR_PIXELS,
    PSM_SPARSE_TEXT,
    CapturedPage,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
    internal_Candidate,
)
from core_pdf.impl.extract.ocr.candidates import (
    internal_augment_candidate,
    internal_candidate_timing_record,
    internal_hidden_text_verification,
    internal_merge_candidate_batches,
    internal_record_candidates,
)
from core_pdf.impl.extract.ocr.grids import (
    internal_detect_ruling_grid,
    internal_grid_cell_tasks,
    internal_grid_is_regular_table,
    internal_GRID_MIN_CELLS,
    internal_grid_region_page_box,
    internal_grid_row_observations,
)
from core_pdf.impl.extract.ocr.raster import (
    internal_adaptive_ocr_raster,
    internal_raster_text_signal,
    internal_rendered_page_raster,
    internal_safe_image_crop,
)
from core_pdf.impl.extract.ocr.regions import (
    internal_adaptive_rescue_decision,
    internal_candidate_ocr_regions,
    internal_candidate_region_tasks,
    internal_direct_scan_allowed,
    internal_dominant_image_region,
    internal_estimated_text_height,
    internal_has_distributed_outline_text,
    internal_high_resolution_weak_region_tasks,
    internal_ocr_region_batch,
    internal_ocr_task_groups,
    internal_page_image_regions,
    internal_primary_text_is_sufficient,
    internal_tile_tasks,
    internal_weak_region_tasks,
)
from core_pdf.impl.extract.ocr.strokes import StrokedTextDecode
from core_pdf.impl.extract.ocr.tesseract import (
    internal_recognize_group,
    internal_recover_timed_out_tasks,
)
from core_pdf.impl.extract.ocr.types import (
    internal_OcrRegion,
    internal_OcrTask,
    internal_PackedStrokedTextRaster,
    internal_Raster,
    internal_RasterRegion,
)
from core_pdf.impl.extract.ocr.vector import (
    internal_decode_stroked_vector_text,
    internal_full_stroked_vector_text_raster,
    internal_packed_stroked_vector_decode_gate,
    internal_recover_stroked_vector_text,
    internal_remap_stroked_vector_candidate,
    internal_stroked_vector_text_raster,
)
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import compose_page
from core_pdf.impl.runtime.execution import TaskScope, WorkStage

# Small affine placement noise is cheaper to absorb in OCR coordinates than to
# recompose and rasterize the entire page around an otherwise usable source image.
OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION = 0.01


def recognize_page(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
) -> RecognitionResult:
    report = RecognitionReport()
    if not plan.ocr_passes:
        return RecognitionResult(ObservationBatch.empty(), report)
    with context.reserve_raster(MAX_OCR_RASTER_BYTES):
        context.raise_if_cancelled()
        observations, cached_stroked_decode = internal_recognize_page_with_reserved_raster(
            capture,
            plan,
            context,
            report=report,
        )
    observations = internal_recover_stroked_vector_text(
        capture,
        observations,
        report,
        cached_decode=cached_stroked_decode,
    )
    return RecognitionResult(observations, report)


def internal_raster_tasks(
    raster: internal_Raster | None,
    page_box: tuple[float, float, float, float],
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str,
) -> tuple[tuple[internal_OcrTask, ...], int]:
    """Tile one optional raster into OCR tasks, paired with its pixel count for the report."""
    if raster is None:
        return (), 0
    return (
        internal_tile_tasks(raster, page_box, ocr_pass, compact_image=compact_image),
        raster.width * raster.height,
    )


def internal_region_tasks(
    region: internal_RasterRegion | None,
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str,
) -> tuple[tuple[internal_OcrTask, ...], int]:
    if region is None:
        return (), 0
    return internal_raster_tasks(
        region.raster, region.page_box, ocr_pass, compact_image=compact_image
    )


def internal_recognize_page_with_reserved_raster(
    capture: CapturedPage,
    plan: WorkPlan,
    context: TaskScope,
    *,
    report: RecognitionReport | None = None,
) -> tuple[ObservationBatch, tuple[int, StrokedTextDecode, float] | None]:
    report = report or RecognitionReport()
    page = capture.page
    page_box = (0.0, 0.0, float(page.width), float(page.height))
    compact_image: bool | str = True
    if capture.evidence.full_page_image:
        image_filters = capture.evidence.image_filters
        if any("JPX" in str(filter_name).upper() for filter_name in image_filters):
            compact_image = "grayscale"
    dominant_regions: dict[int, internal_RasterRegion | None] = {}
    rendered_rasters: dict[tuple[float, int, bool], internal_Raster | None] = {}
    rendered_page: Any | None = None
    candidate_regions: tuple[internal_OcrRegion, ...] | None = None
    candidates: list[tuple[str, internal_Candidate]] = []
    pending_stroked_decode: tuple[int, StrokedTextDecode, float] | None = None
    selected_name = ""
    selected: internal_Candidate | None = None
    selected_tasks: tuple[internal_OcrTask, ...] = ()
    previous_region_additions = 0
    seeded_region_selected = False
    adaptive_rescue_used = False

    def recognize_batch(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        groups = internal_ocr_task_groups(tasks)
        results = context.map_ordered(internal_recognize_group, groups, stage=WorkStage.OCR)
        return tuple(candidate for group in results for candidate in group)

    def recognize_tasks(tasks: tuple[internal_OcrTask, ...]) -> tuple[internal_Candidate, ...]:
        candidates = recognize_batch(tasks)
        if any(candidate.recognition_status == "timeout" for candidate in candidates):
            context.raise_if_cancelled()
            candidates = internal_recover_timed_out_tasks(tasks, candidates, recognize_batch)
        return candidates

    def dominant_image_region_cached(pixel_budget: int) -> internal_RasterRegion | None:
        if pixel_budget not in dominant_regions:
            dominant_regions[pixel_budget] = internal_dominant_image_region(
                capture,
                max_pixels=pixel_budget,
            )
        return dominant_regions[pixel_budget]

    def rendered_raster_cached(ocr_pass: OcrPass) -> internal_Raster | None:
        raster_key = (ocr_pass.scale, ocr_pass.pixel_budget, ocr_pass.include_native_text)
        if raster_key not in rendered_rasters:
            rendered_rasters[raster_key] = internal_rendered_page_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                report=report,
            )
        return rendered_rasters[raster_key]

    if plan.verify_hidden_text:
        context.raise_if_cancelled()
        started = time.perf_counter()
        verification_pass = OcrPass(
            "hidden-text-verification",
            OcrPassScope.PAGE,
            1.0,
            (PSM_SPARSE_TEXT,),
            minimum_confidence=HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
            pixel_budget=HIDDEN_TEXT_VERIFY_PIXELS,
            recognize_words=True,
            region_first=False,
        )
        verification_region = internal_dominant_image_region(
            capture,
            max_pixels=HIDDEN_TEXT_VERIFY_PIXELS,
        )
        verification_tasks, raster_pixels = internal_region_tasks(
            verification_region, verification_pass, compact_image=compact_image
        )
        verification_candidates = recognize_tasks(verification_tasks)
        verification_candidate = internal_merge_candidate_batches(verification_candidates)
        verification = internal_hidden_text_verification(
            capture.observations,
            verification_candidate.observations,
        )
        verification_record: dict[str, object] = {
            "name": verification_pass.name,
            "scope": verification_pass.scope.value,
            "scale": verification_pass.scale,
            "modes": verification_pass.modes,
            "recognize_words": verification_pass.recognize_words,
            "character_confidence_threshold": None,
            "task_count": len(verification_tasks),
            "raster_pixels": raster_pixels,
            "region_stage": "dominant-image-preview",
            "region_boxes": (
                (verification_region.page_box,) if verification_region is not None else ()
            ),
            "full_page_fallback": False,
            "elapsed_seconds": time.perf_counter() - started,
            "render_timings": report.render_timings or {},
            **internal_candidate_timing_record(verification_candidates),
            "accepted_additions": 0,
            "adaptive_retry_scale": None,
            "adaptive_preflight": None,
            "adaptive_rescue_decision": None,
            "adaptive_rescue": None,
            "pixel_budget": verification_pass.pixel_budget,
            "rectangles": tuple(task.rectangle for task in verification_tasks),
            "selected": verification.accepted,
            **verification_candidate.metrics.as_record(),
            **verification.as_record(),
        }
        report.passes += (verification_record,)
        report.hidden_text_verification = {
            "raster_pixels": raster_pixels,
            **verification.as_record(),
        }
        if verification.accepted:
            return internal_promoted_hidden_observations(capture), pending_stroked_decode

    for ocr_pass in plan.ocr_passes:
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and internal_primary_text_is_sufficient(selected)
        ):
            continue
        if (
            selected is not None
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= ocr_pass.run_if_characters_below
        ):
            continue
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 28
            and selected.metrics.mean_confidence >= 97.0
        ):
            continue
        if (
            ocr_pass.run_if_additions_below is not None
            and previous_region_additions >= ocr_pass.run_if_additions_below
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is None
            and capture.evidence.visible_native_characters >= 3_000
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.WEAK_REGIONS
            and ocr_pass.run_if_additions_below is not None
            and previous_region_additions == 0
            and selected is not None
            and selected.metrics.characters >= 32
            and selected.metrics.mean_confidence >= 90.0
        ):
            continue
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and seeded_region_selected
            and ocr_pass.run_if_additions_below is not None
        ):
            selected = None
            selected_name = ""
            selected_tasks = ()
            seeded_region_selected = False
        context.raise_if_cancelled()
        started = time.perf_counter()
        adaptive_preflight: dict[str, object] | None = None
        vector_preview = bool(
            capture.evidence.image_count == 0
            and capture.evidence.vector_complexity >= 100_000
            and capture.evidence.text_coverage < 0.05
        )
        if (
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget == PRIMARY_OCR_PIXELS
            and (capture.evidence.full_page_image or vector_preview)
        ):
            preview_raster: internal_Raster | None = None
            if capture.evidence.full_page_image:
                if OCR_PREFLIGHT_PIXELS not in dominant_regions:
                    # The preview only measures text height, so enlarging it would
                    # cost time and shift the projection this decision depends on.
                    dominant_regions[OCR_PREFLIGHT_PIXELS] = internal_dominant_image_region(
                        capture,
                        max_pixels=OCR_PREFLIGHT_PIXELS,
                        upscale=False,
                    )
                preview_region = dominant_regions[OCR_PREFLIGHT_PIXELS]
                preview_raster = preview_region.raster if preview_region is not None else None
            else:
                if rendered_page is None:
                    rendered_page = compose_page(
                        capture.page,
                        RenderOptions(include_text=ocr_pass.include_native_text),
                        page_program=capture.program,
                    )
                preview_raster = internal_rendered_page_raster(
                    capture,
                    ocr_pass.scale,
                    rendered=rendered_page,
                    cache=True,
                    max_pixels=OCR_PREFLIGHT_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    report=report,
                )
            if preview_raster is not None:
                preview_height = internal_estimated_text_height(preview_raster)
                projected_height = preview_height * math.sqrt(
                    ocr_pass.pixel_budget / max(1, preview_raster.width * preview_raster.height)
                )
                projected_limit = 22.0 if vector_preview else 20.0
                if 12.0 <= projected_height < projected_limit:
                    original_scale = ocr_pass.scale
                    ocr_pass = replace(
                        ocr_pass,
                        scale=min(
                            8.0,
                            max(
                                original_scale + 0.5,
                                original_scale * 32.0 / projected_height,
                            ),
                        ),
                        pixel_budget=MAX_OCR_PIXELS,
                    )
                    adaptive_preflight = {
                        "preview_pixels": preview_raster.width * preview_raster.height,
                        "preview_text_height": preview_height,
                        "projected_primary_text_height": projected_height,
                        "selected_scale": ocr_pass.scale,
                        "source": "vector-render" if vector_preview else "dominant-image",
                    }
        tasks: tuple[internal_OcrTask, ...]
        packed_stroked: internal_PackedStrokedTextRaster | None = None
        raster_pixels = 0
        skipped_raster_pixels = 0
        image_text_preflight: tuple[dict[str, object], ...] = ()
        skipped_region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        region_stage = "page"
        region_boxes: tuple[tuple[float, float, float, float], ...] = ()
        if (
            ocr_pass.region_first
            and ocr_pass.scope in {OcrPassScope.PAGE, OcrPassScope.WEAK_REGIONS}
            and (
                ocr_pass.scope is not OcrPassScope.WEAK_REGIONS
                or selected is not None
                or ocr_pass.seed_with_native
            )
        ):
            if candidate_regions is None:
                candidate_regions = internal_candidate_ocr_regions(capture)
            distributed_outline_text = bool(
                ocr_pass.scope is OcrPassScope.PAGE
                and internal_has_distributed_outline_text(capture)
            )
            region_batch = (
                (
                    internal_OcrRegion(
                        page_box,
                        float("inf"),
                        ("distributed-outline-text",),
                    ),
                )
                if distributed_outline_text
                else internal_ocr_region_batch(
                    candidate_regions,
                    ocr_pass,
                    page_area=max(1.0, float(page.width) * float(page.height)),
                )
            )
            tasks, raster_pixels, rendered_page, region_boxes = internal_candidate_region_tasks(
                capture,
                region_batch,
                ocr_pass,
                rendered=rendered_page,
                compact_image=compact_image,
                report=report,
            )
            region_stage = (
                "distributed-outline-page" if distributed_outline_text else "initial-regions"
            )
            if len(region_batch) == 1 and "page-fallback" in region_batch[0].reasons:
                region_stage = "page"
        elif ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            if selected is None and not ocr_pass.seed_with_native:
                continue
            if selected is not None and selected_tasks:
                tasks, raster_pixels, rendered_page, region_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        selected_tasks,
                        ocr_pass,
                        selected.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        report=report,
                    )
                )
                region_stage = "weak-region-crops"
            else:
                direct_region = dominant_image_region_cached(ocr_pass.pixel_budget)
                raster = direct_region.raster if direct_region is not None else None
                raster_page_box = direct_region.page_box if direct_region is not None else page_box
                if raster is None:
                    raster = rendered_raster_cached(ocr_pass)
                    raster_page_box = page_box
                tasks = (
                    internal_weak_region_tasks(
                        raster,
                        raster_page_box,
                        ocr_pass,
                        selected.observations if selected is not None else capture.observations,
                        compact_image=compact_image,
                    )
                    if raster is not None
                    else ()
                )
                raster_pixels = (
                    sum(task.rectangle[2] * task.rectangle[3] for task in tasks)
                    if raster is not None
                    else 0
                )
        elif ocr_pass.scope is OcrPassScope.STROKED_VECTOR_TEXT:
            packed_stroked = internal_stroked_vector_text_raster(
                capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                report=report,
            )
            if packed_stroked is not None:
                region_stage = "packed-stroked-vector-text"
                region_boxes = (
                    (capture.evidence.stroked_vector_text.bbox,)
                    if capture.evidence.stroked_vector_text.bbox is not None
                    else ()
                )
                tasks, raster_pixels = internal_raster_tasks(
                    packed_stroked.raster,
                    packed_stroked.packed_box,
                    replace(ocr_pass, recognize_words=True, collect_symbols=True),
                    compact_image=compact_image,
                )
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    report=report,
                )
                region_stage = "stroked-vector-text-fallback"
                region_boxes = (fallback_region.page_box,) if fallback_region is not None else ()
                tasks, raster_pixels = internal_region_tasks(
                    fallback_region, ocr_pass, compact_image=compact_image
                )
                report.stroked_vector_packed = {
                    "accepted": False,
                    "cells": 0,
                    "raster_pixels": 0,
                    "unmapped_observations": 0,
                    "fallback_used": bool(tasks),
                }
        elif ocr_pass.scope is OcrPassScope.IMAGE_REGIONS:
            regions = internal_page_image_regions(
                capture,
                minimum_area_ratio=0.02,
                max_pixels=ocr_pass.pixel_budget,
                maximum_axis_deviation=OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION,
            )
            if regions:
                region_signals = tuple(
                    (region, internal_raster_text_signal(region.raster.image)) for region in regions
                )
                image_text_preflight = tuple(
                    {
                        "page_box": region.page_box,
                        "raster_pixels": region.raster.width * region.raster.height,
                        **signal.as_record(),
                    }
                    for region, signal in region_signals
                )
                eligible_regions = tuple(
                    region
                    for region, signal in region_signals
                    if signal.likely_text
                    or (
                        signal.horizontal_edge_ratio >= 0.035
                        and sum(len(t.strip()) for t in capture.observations.text) < 15
                    )
                )
                skipped_regions = tuple(
                    region for region, signal in region_signals if not signal.likely_text
                )
                skipped_raster_pixels = sum(
                    region.raster.width * region.raster.height for region in skipped_regions
                )
                skipped_region_boxes = tuple(region.page_box for region in skipped_regions)
                region_boxes = tuple(region.page_box for region in eligible_regions)
                region_stage = "direct-image-regions"
                tasks = tuple(
                    task
                    for region in eligible_regions
                    for task in internal_tile_tasks(
                        region.raster,
                        region.page_box,
                        ocr_pass,
                        compact_image=compact_image,
                    )
                )
                raster_pixels = sum(
                    region.raster.width * region.raster.height for region in eligible_regions
                )
            else:
                fallback_scale = max(2.0, ocr_pass.scale)
                image_crop = internal_safe_image_crop(capture)
                raster = internal_rendered_page_raster(
                    capture,
                    fallback_scale,
                    crop=image_crop,
                    max_pixels=ocr_pass.pixel_budget,
                    include_native_text=ocr_pass.include_native_text,
                    report=report,
                )
                raster_page_box = image_crop or page_box
                tasks, raster_pixels = internal_raster_tasks(
                    raster, raster_page_box, ocr_pass, compact_image=compact_image
                )
        else:
            direct_region = (
                dominant_image_region_cached(ocr_pass.pixel_budget)
                if internal_direct_scan_allowed(capture, plan)
                else None
            )
            raster = direct_region.raster if direct_region is not None else None
            raster_page_box = direct_region.page_box if direct_region is not None else page_box
            if raster is None:
                raster = rendered_raster_cached(ocr_pass)
                raster_page_box = page_box
            task_raster = (
                internal_adaptive_ocr_raster(raster)
                if raster is not None and ocr_pass.name == "adaptive-page"
                else raster
            )
            tasks = internal_raster_tasks(
                task_raster, raster_page_box, ocr_pass, compact_image=compact_image
            )[0]
            raster_pixels = raster.width * raster.height if raster is not None else 0
        if not tasks:
            if not image_text_preflight:
                continue
            region_stage = "image-text-preflight"

        candidate_source_tasks = tasks
        task_candidates = recognize_tasks(tasks)
        if packed_stroked is not None:
            remapped_with_counts = tuple(
                internal_remap_stroked_vector_candidate(candidate, packed_stroked)
                for candidate in task_candidates
            )
            task_candidates = tuple(item[0] for item in remapped_with_counts)
            unmapped_observations = sum(item[1] for item in remapped_with_counts)
            packed_candidate = internal_merge_candidate_batches(task_candidates)
            decode_started = time.perf_counter()
            packed_decode = internal_decode_stroked_vector_text(
                capture,
                packed_candidate.observations,
                packed_candidate.symbols,
            )
            decode_seconds = time.perf_counter() - decode_started
            packed_accepted, packed_gate = internal_packed_stroked_vector_decode_gate(
                packed_decode,
                len(packed_stroked.cells),
            )
            packed_pixels = raster_pixels
            fallback_used = False
            if packed_accepted:
                # Seed packing only rasterizes multi-glyph runs, so isolated
                # glyphs (pin numbers, lone digits) are never shown to OCR when
                # the packed decode gate passes. Recognize them from their own
                # high-scale montage as a supplement.
                isolated_packed = internal_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    variant="isolated",
                    report=report,
                )
                isolated_tasks, isolated_pixels = (
                    internal_raster_tasks(
                        isolated_packed.raster,
                        isolated_packed.packed_box,
                        replace(
                            ocr_pass,
                            recognize_words=True,
                            collect_symbols=True,
                            minimum_confidence=50.0,
                        ),
                        compact_image=compact_image,
                    )
                    if isolated_packed is not None
                    else ((), 0)
                )
                if isolated_tasks and isolated_packed is not None:
                    isolated_remapped = tuple(
                        internal_remap_stroked_vector_candidate(
                            candidate,
                            isolated_packed,
                            digit_bearing_only=True,
                        )
                        for candidate in recognize_tasks(isolated_tasks)
                    )
                    isolated_candidates = tuple(item[0] for item in isolated_remapped)
                    packed_gate["isolated_cells"] = len(isolated_packed.cells)
                    packed_gate["isolated_observations"] = sum(
                        len(item[0].observations) for item in isolated_remapped
                    )
                    task_candidates = (*task_candidates, *isolated_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *isolated_tasks)
                    tasks = (*tasks, *isolated_tasks)
                    packed_candidate = internal_merge_candidate_batches(task_candidates)
                    raster_pixels += isolated_pixels
                pending_stroked_decode = (
                    id(packed_candidate.observations),
                    packed_decode,
                    decode_seconds,
                )
            else:
                fallback_region = internal_full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                    report=report,
                )
                fallback_tasks, fallback_pixels = internal_region_tasks(
                    fallback_region,
                    replace(ocr_pass, recognize_words=False),
                    compact_image=compact_image,
                )
                if fallback_tasks:
                    fallback_used = True
                    fallback_candidates = recognize_tasks(fallback_tasks)
                    task_candidates = (*task_candidates, *fallback_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *fallback_tasks)
                    tasks = (*tasks, *fallback_tasks)
                    packed_candidate = internal_merge_candidate_batches(fallback_candidates)
                    raster_pixels += fallback_pixels
                    region_stage = "stroked-vector-text-fallback"
                    region_boxes = (
                        (fallback_region.page_box,) if fallback_region is not None else region_boxes
                    )
            report.stroked_vector_packed = {
                **packed_gate,
                "raster_pixels": packed_pixels,
                "unmapped_observations": unmapped_observations,
                "symbol_observations": len(packed_candidate.symbols),
                "fallback_used": fallback_used,
            }
            candidate = packed_candidate
        else:
            candidate = internal_merge_candidate_batches(task_candidates)
        if (
            selected is not None
            and plan.augment_page_candidates
            and ocr_pass.scope is OcrPassScope.PAGE
            and not capture.evidence.vector_complexity >= 180
        ):
            candidate, _ = internal_augment_candidate(
                selected,
                candidate,
                minimum_confidence=70.0,
            )
        adaptive_retry_scale: float | None = None
        adaptive_rescue: dict[str, object] | None = None
        adaptive_rescue_decision: dict[str, object] | None = None
        median_height = candidate.metrics.median_text_height
        rescue_eligible = bool(
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget < MAX_OCR_PIXELS
            and not adaptive_rescue_used
            and candidate.metrics.characters >= ocr_pass.minimum_characters_for_rescue
            and (candidate.metrics.characters < 32 or 0.0 < median_height < 24.0)
        )
        run_rescue = False
        if rescue_eligible:
            adaptive_rescue_used = True
            run_rescue, adaptive_rescue_decision = internal_adaptive_rescue_decision(
                candidate,
                candidate_source_tasks,
                ocr_pass,
            )
        if run_rescue:
            factor = 1.5 if median_height <= 0.0 else min(2.5, max(1.25, 32.0 / median_height))
            adaptive_retry_scale = min(8.0, max(ocr_pass.scale + 0.5, ocr_pass.scale * factor))
            retry_pass = replace(
                ocr_pass,
                name="adaptive-rescue",
                scale=adaptive_retry_scale,
                pixel_budget=MAX_OCR_PIXELS,
                region_first=False,
            )
            retry_scope = (
                "page"
                if candidate.metrics.characters < 32 or median_height < 18.0
                else "weak-regions"
            )
            retry_boxes: tuple[tuple[float, float, float, float], ...] = ()
            if retry_scope == "page":
                retry_raster = internal_rendered_page_raster(
                    capture,
                    adaptive_retry_scale,
                    max_pixels=MAX_OCR_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                    report=report,
                )
                retry_tasks, rescue_pixels = internal_raster_tasks(
                    retry_raster, page_box, retry_pass, compact_image=compact_image
                )
            else:
                retry_pass = replace(
                    retry_pass,
                    scope=OcrPassScope.WEAK_REGIONS,
                    tiles=max(6, retry_pass.tiles),
                    region_columns=max(3, retry_pass.region_columns),
                    max_regions=max(8, retry_pass.max_regions),
                )
                retry_tasks, rescue_pixels, rendered_page, retry_boxes = (
                    internal_high_resolution_weak_region_tasks(
                        capture,
                        tasks,
                        retry_pass,
                        candidate.observations,
                        rendered=rendered_page,
                        compact_image=compact_image,
                        report=report,
                    )
                )
            if retry_tasks:
                candidate_source_tasks = (*candidate_source_tasks, *retry_tasks)
                retry_candidates = recognize_tasks(retry_tasks)
                retry_candidate = internal_merge_candidate_batches(retry_candidates)
                augmented_candidate, rescue_additions = internal_augment_candidate(
                    candidate,
                    retry_candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
                if retry_candidate.metrics.utility > augmented_candidate.metrics.utility * 1.05:
                    candidate = retry_candidate
                elif augmented_candidate.metrics.utility > candidate.metrics.utility:
                    candidate = augmented_candidate
                task_candidates = (*task_candidates, *retry_candidates)
                raster_pixels += rescue_pixels
                adaptive_rescue = {
                    "scope": retry_scope,
                    "scale": adaptive_retry_scale,
                    "raster_pixels": rescue_pixels,
                    "task_count": len(retry_tasks),
                    "accepted_additions": rescue_additions,
                    "region_boxes": retry_boxes,
                }
        additions = 0
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            used_native_seed = selected is None
            if selected is not None:
                candidate, additions = internal_augment_candidate(
                    selected,
                    candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
            else:
                additions = len(candidate.observations)
        candidates.append((ocr_pass.name, candidate))
        elapsed = time.perf_counter() - started
        report.passes += (
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "recognize_words": any(task.recognize_words for task in tasks),
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "task_count": len(tasks),
                "raster_pixels": raster_pixels,
                "skipped_raster_pixels": skipped_raster_pixels,
                "image_text_preflight": image_text_preflight,
                "region_stage": region_stage,
                "region_boxes": region_boxes,
                "skipped_region_boxes": skipped_region_boxes,
                "full_page_fallback": (
                    region_stage == "page" and ocr_pass.scope is OcrPassScope.PAGE
                ),
                "elapsed_seconds": elapsed,
                "render_timings": report.render_timings or {},
                **internal_candidate_timing_record(task_candidates),
                "accepted_additions": additions,
                "adaptive_retry_scale": adaptive_retry_scale,
                "adaptive_preflight": adaptive_preflight,
                "adaptive_rescue_decision": adaptive_rescue_decision,
                "adaptive_rescue": adaptive_rescue,
                "pixel_budget": ocr_pass.pixel_budget,
                "rectangles": tuple(task.rectangle for task in tasks),
                "selected": False,
                **candidate.metrics.as_record(),
            },
        )
        if not tasks:
            continue
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            previous_region_additions = additions
            if additions:
                selected_name = ocr_pass.name
                selected = candidate
                selected_tasks = (*selected_tasks, *candidate_source_tasks)
                seeded_region_selected = used_native_seed and ocr_pass.seed_with_native
            continue
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            selected_name = ocr_pass.name
            selected = candidate
            selected_tasks = candidate_source_tasks

    if selected is None:
        internal_record_candidates(tuple(candidates), selected_name, report)
        return ObservationBatch.empty(), pending_stroked_decode
    for diagnostic in report.passes:
        diagnostic["selected"] = diagnostic["name"] == selected_name
    internal_record_candidates(tuple(candidates), selected_name, report)
    if selected_tasks:
        # Ruled scanned tables defeat Tesseract's page segmentation; when the
        # page raster shows a full ruling grid, re-recognize cell by cell and
        # let the grid text replace the page-segmented text inside the grid.
        source_task = max(
            selected_tasks,
            key=lambda task: task.rectangle[2] * task.rectangle[3],
        )
        grid = internal_detect_ruling_grid(source_task.image)
        if grid is not None and internal_grid_is_regular_table(
            grid, selected.observations, source_task
        ):
            x_lines, y_lines, source_samples, slope = grid
            cell_tasks = internal_grid_cell_tasks(
                source_task, x_lines, y_lines, source_samples, slope
            )
            if len(cell_tasks) >= internal_GRID_MIN_CELLS:
                cell_candidate = internal_merge_candidate_batches(recognize_tasks(cell_tasks))
                cell_observations = internal_grid_row_observations(cell_candidate.observations)
                if len(cell_observations):
                    grid_box = internal_grid_region_page_box(source_task, x_lines, y_lines)
                    prior = selected.observations
                    centers_x = (prior.bbox[:, 0] + prior.bbox[:, 2]) * 0.5
                    centers_y = (prior.bbox[:, 1] + prior.bbox[:, 3]) * 0.5
                    outside = ~(
                        (centers_x >= grid_box[0])
                        & (centers_x <= grid_box[2])
                        & (centers_y >= grid_box[1])
                        & (centers_y <= grid_box[3])
                    )
                    replaced_alnum = sum(
                        sum(character.isalnum() for character in prior.text[index])
                        for index in numpy.flatnonzero(~outside)
                    )
                    cell_alnum = sum(
                        sum(character.isalnum() for character in text)
                        for text in cell_observations.text
                    )
                    if cell_alnum < replaced_alnum * 0.8:
                        # The page-segmented reads carried more content than
                        # the cell reads; this grid's cells recognize worse
                        # than whole-page OCR, so keep the original.
                        return selected.observations, pending_stroked_decode
                    retained = prior.take(numpy.flatnonzero(outside))
                    report.grid_cell_ocr = {
                        "cells": len(cell_tasks),
                        "cell_observations": len(cell_observations),
                        "replaced_observations": int(numpy.count_nonzero(~outside)),
                        "grid_box": grid_box,
                        "columns": len(x_lines) - 1,
                        "rows": len(y_lines) - 1,
                    }
                    return (
                        ObservationBatch.concatenate(retained, cell_observations),
                        pending_stroked_decode,
                    )
    return selected.observations, pending_stroked_decode
