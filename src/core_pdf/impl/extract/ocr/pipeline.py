# SPDX-License-Identifier: AGPL-3.0-only
"""Tesseract integration: rasterization, recognition, and rescue passes."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy

from core_pdf.impl.extract.capture import internal_promoted_hidden_observations
from core_pdf.impl.extract.contracts import (
    HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
    HIDDEN_TEXT_VERIFY_PIXELS,
    MAX_OCR_PIXELS,
    MAX_OCR_RASTER_BYTES,
    PSM_SPARSE_TEXT,
    CapturedPage,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    RecognitionReport,
    RecognitionResult,
    WorkPlan,
)
from core_pdf.impl.extract.grids import (
    internal_detect_ruling_grid,
    internal_grid_cell_tasks,
    internal_grid_is_regular_table,
    internal_GRID_MIN_CELLS,
    internal_grid_region_page_box,
    internal_grid_row_observations,
)
from core_pdf.impl.extract.ocr.candidates import (
    internal_augment_candidate,
    internal_candidate_timing_record,
    internal_hidden_text_verification,
    internal_merge_candidate_batches,
)
from core_pdf.impl.extract.ocr.execution import internal_OcrPassExecution, internal_OcrPassState
from core_pdf.impl.extract.ocr.pass_tasks import (
    internal_OcrPassTaskResources,
    internal_raster_tasks,
    internal_region_tasks,
)
from core_pdf.impl.extract.ocr.raster import internal_rendered_page_raster
from core_pdf.impl.extract.ocr.region_tasks import internal_ocr_task_groups
from core_pdf.impl.extract.ocr.regions import internal_dominant_image_region
from core_pdf.impl.extract.ocr.rescue import internal_adaptive_rescue_decision
from core_pdf.impl.extract.ocr.strokes import StrokedTextDecode
from core_pdf.impl.extract.ocr.tesseract import (
    internal_recognize_group,
    internal_recover_timed_out_tasks,
)
from core_pdf.impl.extract.ocr.types import internal_OcrTask
from core_pdf.impl.extract.ocr.vector import (
    internal_decode_stroked_vector_text,
    internal_full_stroked_vector_text_raster,
    internal_packed_stroked_vector_decode_gate,
    internal_recover_stroked_vector_text,
    internal_remap_stroked_vector_candidate,
    internal_stroked_vector_text_raster,
)
from core_pdf.impl.extract.quality import internal_Candidate
from core_pdf.impl.runtime.execution import TaskScope, WorkStage


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
    task_resources = internal_OcrPassTaskResources(
        capture,
        plan,
        report,
        compact_image,
    )
    pending_stroked_decode: tuple[int, StrokedTextDecode, float] | None = None
    pass_state = internal_OcrPassState()
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
        prepared_state = pass_state.prepare(
            ocr_pass,
            visible_native_characters=capture.evidence.visible_native_characters,
        )
        if prepared_state is None:
            continue
        pass_state = prepared_state
        selected = pass_state.selected
        selected_tasks = pass_state.selected_tasks
        context.raise_if_cancelled()
        started = time.perf_counter()
        pass_tasks = task_resources.materialize(
            ocr_pass,
            selected=selected,
            selected_tasks=selected_tasks,
        )
        if pass_tasks is None:
            continue
        ocr_pass = pass_tasks.ocr_pass
        tasks = pass_tasks.tasks
        packed_stroked = pass_tasks.packed_stroked
        raster_pixels = pass_tasks.raster_pixels
        skipped_raster_pixels = pass_tasks.skipped_raster_pixels
        image_text_preflight = pass_tasks.image_text_preflight
        skipped_region_boxes = pass_tasks.skipped_region_boxes
        region_stage = pass_tasks.region_stage
        region_boxes = pass_tasks.region_boxes
        adaptive_preflight = pass_tasks.adaptive_preflight
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
                retry_regions = task_resources.internal_high_resolution_weak_region_tasks(
                    tasks,
                    retry_pass,
                    candidate.observations,
                )
                retry_tasks = retry_regions.tasks
                rescue_pixels = retry_regions.raster_pixels
                retry_boxes = retry_regions.region_boxes
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
        pass_state = internal_OcrPassExecution(
            ocr_pass=ocr_pass,
            candidate=candidate,
            candidate_source_tasks=candidate_source_tasks,
            task_candidates=task_candidates,
            tasks=tasks,
            started=started,
            raster_pixels=raster_pixels,
            skipped_raster_pixels=skipped_raster_pixels,
            image_text_preflight=image_text_preflight,
            region_stage=region_stage,
            region_boxes=region_boxes,
            skipped_region_boxes=skipped_region_boxes,
            adaptive_retry_scale=adaptive_retry_scale,
            adaptive_preflight=adaptive_preflight,
            adaptive_rescue_decision=adaptive_rescue_decision,
            adaptive_rescue=adaptive_rescue,
        ).complete(pass_state, report)

    pass_state.record_selection(report)
    selected = pass_state.selected
    if selected is None:
        return ObservationBatch.empty(), pending_stroked_decode
    selected_tasks = pass_state.selected_tasks
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
