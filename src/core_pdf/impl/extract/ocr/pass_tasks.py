# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive OCR pass preparation and scope-specific task materialization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from core_pdf.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    OCR_PREFLIGHT_PIXELS,
    PRIMARY_OCR_PIXELS,
    CapturedPage,
    ObservationBatch,
    OcrPass,
    OcrPassScope,
    RecognitionReport,
    WorkPlan,
)
from core_pdf.impl.extract.ocr.raster import (
    internal_adaptive_ocr_raster,
    internal_raster_text_signal,
    internal_rendered_page_raster,
    internal_safe_image_crop,
)
from core_pdf.impl.extract.ocr.region_tasks import (
    internal_candidate_region_tasks,
    internal_direct_scan_allowed,
    internal_estimated_text_height,
    internal_high_resolution_weak_region_tasks,
    internal_tile_tasks,
    internal_weak_region_tasks,
)
from core_pdf.impl.extract.ocr.regions import (
    internal_candidate_ocr_regions,
    internal_dominant_image_region,
    internal_has_distributed_outline_text,
    internal_ocr_region_batch,
    internal_page_image_regions,
)
from core_pdf.impl.extract.ocr.types import (
    internal_OcrRegion,
    internal_OcrTask,
    internal_PackedStrokedTextRaster,
    internal_Raster,
    internal_RasterRegion,
)
from core_pdf.impl.extract.ocr.vector import (
    internal_full_stroked_vector_text_raster,
    internal_stroked_vector_text_raster,
)
from core_pdf.impl.extract.quality import internal_Candidate
from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.render.page import RenderedPage, compose_page

internal_PageBox = tuple[float, float, float, float]

# Small affine placement noise is cheaper to absorb in OCR coordinates than to
# recompose and rasterize the entire page around an otherwise usable source image.
internal_OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION = 0.01


def internal_raster_tasks(
    raster: internal_Raster | None,
    page_box: internal_PageBox,
    ocr_pass: OcrPass,
    *,
    compact_image: bool | str,
) -> tuple[tuple[internal_OcrTask, ...], int]:
    """Tile one optional raster into OCR tasks, paired with its reportable pixel count."""
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
        region.raster,
        region.page_box,
        ocr_pass,
        compact_image=compact_image,
    )


@dataclass(frozen=True, slots=True)
class internal_OcrPassTasks:
    """Materialized work and diagnostics for one OCR pass."""

    ocr_pass: OcrPass
    tasks: tuple[internal_OcrTask, ...] = ()
    packed_stroked: internal_PackedStrokedTextRaster | None = None
    raster_pixels: int = 0
    skipped_raster_pixels: int = 0
    image_text_preflight: tuple[dict[str, object], ...] = ()
    skipped_region_boxes: tuple[internal_PageBox, ...] = ()
    region_stage: str = "page"
    region_boxes: tuple[internal_PageBox, ...] = ()
    adaptive_preflight: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class internal_OcrRegionTasks:
    """Region tasks whose rendering may advance the shared composed-page cache."""

    tasks: tuple[internal_OcrTask, ...]
    raster_pixels: int
    region_boxes: tuple[internal_PageBox, ...]


@dataclass(slots=True)
class internal_OcrPassTaskResources:
    """Page-local raster and region caches shared by scheduled OCR passes."""

    capture: CapturedPage
    plan: WorkPlan
    report: RecognitionReport
    compact_image: bool | str
    rendered_page: RenderedPage | None = None
    internal_dominant_regions: dict[int, internal_RasterRegion | None] = field(default_factory=dict)
    internal_rendered_rasters: dict[tuple[float, int, bool], internal_Raster | None] = field(
        default_factory=dict
    )
    internal_candidate_regions: tuple[internal_OcrRegion, ...] | None = None

    @property
    def page_box(self) -> internal_PageBox:
        page = self.capture.page
        return 0.0, 0.0, float(page.width), float(page.height)

    def internal_dominant_image_region_cached(
        self,
        pixel_budget: int,
    ) -> internal_RasterRegion | None:
        if pixel_budget not in self.internal_dominant_regions:
            self.internal_dominant_regions[pixel_budget] = internal_dominant_image_region(
                self.capture,
                max_pixels=pixel_budget,
            )
        return self.internal_dominant_regions[pixel_budget]

    def internal_rendered_raster_cached(self, ocr_pass: OcrPass) -> internal_Raster | None:
        raster_key = (ocr_pass.scale, ocr_pass.pixel_budget, ocr_pass.include_native_text)
        if raster_key not in self.internal_rendered_rasters:
            self.internal_rendered_rasters[raster_key] = internal_rendered_page_raster(
                self.capture,
                ocr_pass.scale,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                report=self.report,
            )
        return self.internal_rendered_rasters[raster_key]

    def internal_adapt_pass(
        self,
        ocr_pass: OcrPass,
    ) -> tuple[OcrPass, dict[str, object] | None]:
        vector_preview = bool(
            self.capture.evidence.image_count == 0
            and self.capture.evidence.vector_complexity >= 100_000
            and self.capture.evidence.text_coverage < 0.05
        )
        if not (
            ocr_pass.adaptive_scale
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.pixel_budget == PRIMARY_OCR_PIXELS
            and (self.capture.evidence.full_page_image or vector_preview)
        ):
            return ocr_pass, None

        preview_raster: internal_Raster | None = None
        if self.capture.evidence.full_page_image:
            if OCR_PREFLIGHT_PIXELS not in self.internal_dominant_regions:
                # The preview only measures text height, so enlarging it would
                # cost time and shift the projection this decision depends on.
                self.internal_dominant_regions[OCR_PREFLIGHT_PIXELS] = (
                    internal_dominant_image_region(
                        self.capture,
                        max_pixels=OCR_PREFLIGHT_PIXELS,
                        upscale=False,
                    )
                )
            preview_region = self.internal_dominant_regions[OCR_PREFLIGHT_PIXELS]
            preview_raster = preview_region.raster if preview_region is not None else None
        else:
            if self.rendered_page is None:
                self.rendered_page = compose_page(
                    self.capture.page,
                    RenderOptions(include_text=ocr_pass.include_native_text),
                    page_program=self.capture.program,
                )
            preview_raster = internal_rendered_page_raster(
                self.capture,
                ocr_pass.scale,
                rendered=self.rendered_page,
                cache=True,
                max_pixels=OCR_PREFLIGHT_PIXELS,
                include_native_text=ocr_pass.include_native_text,
                report=self.report,
            )
        if preview_raster is None:
            return ocr_pass, None

        preview_height = internal_estimated_text_height(preview_raster)
        projected_height = preview_height * math.sqrt(
            ocr_pass.pixel_budget / max(1, preview_raster.width * preview_raster.height)
        )
        projected_limit = 22.0 if vector_preview else 20.0
        if not 12.0 <= projected_height < projected_limit:
            return ocr_pass, None

        original_scale = ocr_pass.scale
        adapted_pass = replace(
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
        return adapted_pass, {
            "preview_pixels": preview_raster.width * preview_raster.height,
            "preview_text_height": preview_height,
            "projected_primary_text_height": projected_height,
            "selected_scale": adapted_pass.scale,
            "source": "vector-render" if vector_preview else "dominant-image",
        }

    def materialize(
        self,
        ocr_pass: OcrPass,
        *,
        selected: internal_Candidate | None,
        selected_tasks: tuple[internal_OcrTask, ...],
    ) -> internal_OcrPassTasks | None:
        """Adapt ``ocr_pass`` and construct the tasks belonging to its scope."""
        ocr_pass, adaptive_preflight = self.internal_adapt_pass(ocr_pass)
        if (
            ocr_pass.region_first
            and ocr_pass.scope in {OcrPassScope.PAGE, OcrPassScope.WEAK_REGIONS}
            and (
                ocr_pass.scope is not OcrPassScope.WEAK_REGIONS
                or selected is not None
                or ocr_pass.seed_with_native
            )
        ):
            result = self.internal_initial_region_tasks(ocr_pass)
        elif ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            if selected is None and not ocr_pass.seed_with_native:
                return None
            result = self.internal_weak_region_tasks(ocr_pass, selected, selected_tasks)
        elif ocr_pass.scope is OcrPassScope.STROKED_VECTOR_TEXT:
            result = self.internal_stroked_vector_tasks(ocr_pass)
        elif ocr_pass.scope is OcrPassScope.IMAGE_REGIONS:
            result = self.internal_image_region_tasks(ocr_pass)
        else:
            result = self.internal_page_tasks(ocr_pass)
        return replace(result, adaptive_preflight=adaptive_preflight)

    def internal_initial_region_tasks(self, ocr_pass: OcrPass) -> internal_OcrPassTasks:
        if self.internal_candidate_regions is None:
            self.internal_candidate_regions = internal_candidate_ocr_regions(self.capture)
        distributed_outline_text = bool(
            ocr_pass.scope is OcrPassScope.PAGE
            and internal_has_distributed_outline_text(self.capture)
        )
        region_batch = (
            (
                internal_OcrRegion(
                    self.page_box,
                    float("inf"),
                    ("distributed-outline-text",),
                ),
            )
            if distributed_outline_text
            else internal_ocr_region_batch(
                self.internal_candidate_regions,
                ocr_pass,
                page_area=max(
                    1.0,
                    float(self.capture.page.width) * float(self.capture.page.height),
                ),
            )
        )
        tasks, raster_pixels, self.rendered_page, region_boxes = internal_candidate_region_tasks(
            self.capture,
            region_batch,
            ocr_pass,
            rendered=self.rendered_page,
            compact_image=self.compact_image,
            report=self.report,
        )
        region_stage = "distributed-outline-page" if distributed_outline_text else "initial-regions"
        if len(region_batch) == 1 and "page-fallback" in region_batch[0].reasons:
            region_stage = "page"
        return internal_OcrPassTasks(
            ocr_pass,
            tasks=tasks,
            raster_pixels=raster_pixels,
            region_stage=region_stage,
            region_boxes=region_boxes,
        )

    def internal_weak_region_tasks(
        self,
        ocr_pass: OcrPass,
        selected: internal_Candidate | None,
        selected_tasks: tuple[internal_OcrTask, ...],
    ) -> internal_OcrPassTasks:
        if selected is not None and selected_tasks:
            result = self.internal_high_resolution_weak_region_tasks(
                selected_tasks,
                ocr_pass,
                selected.observations,
            )
            return internal_OcrPassTasks(
                ocr_pass,
                tasks=result.tasks,
                raster_pixels=result.raster_pixels,
                region_stage="weak-region-crops",
                region_boxes=result.region_boxes,
            )

        direct_region = self.internal_dominant_image_region_cached(ocr_pass.pixel_budget)
        raster = direct_region.raster if direct_region is not None else None
        raster_page_box = direct_region.page_box if direct_region is not None else self.page_box
        if raster is None:
            raster = self.internal_rendered_raster_cached(ocr_pass)
            raster_page_box = self.page_box
        tasks = (
            internal_weak_region_tasks(
                raster,
                raster_page_box,
                ocr_pass,
                selected.observations if selected is not None else self.capture.observations,
                compact_image=self.compact_image,
            )
            if raster is not None
            else ()
        )
        raster_pixels = (
            sum(task.rectangle[2] * task.rectangle[3] for task in tasks)
            if raster is not None
            else 0
        )
        return internal_OcrPassTasks(ocr_pass, tasks=tasks, raster_pixels=raster_pixels)

    def internal_high_resolution_weak_region_tasks(
        self,
        source_tasks: tuple[internal_OcrTask, ...],
        ocr_pass: OcrPass,
        primary: ObservationBatch,
    ) -> internal_OcrRegionTasks:
        tasks, raster_pixels, self.rendered_page, region_boxes = (
            internal_high_resolution_weak_region_tasks(
                self.capture,
                source_tasks,
                ocr_pass,
                primary,
                rendered=self.rendered_page,
                compact_image=self.compact_image,
                report=self.report,
            )
        )
        return internal_OcrRegionTasks(tasks, raster_pixels, region_boxes)

    def internal_stroked_vector_tasks(self, ocr_pass: OcrPass) -> internal_OcrPassTasks:
        packed_stroked = internal_stroked_vector_text_raster(
            self.capture,
            ocr_pass.scale,
            max_pixels=ocr_pass.pixel_budget,
            report=self.report,
        )
        if packed_stroked is not None:
            region_boxes = (
                (self.capture.evidence.stroked_vector_text.bbox,)
                if self.capture.evidence.stroked_vector_text.bbox is not None
                else ()
            )
            tasks, raster_pixels = internal_raster_tasks(
                packed_stroked.raster,
                packed_stroked.packed_box,
                replace(ocr_pass, recognize_words=True, collect_symbols=True),
                compact_image=self.compact_image,
            )
            return internal_OcrPassTasks(
                ocr_pass,
                tasks=tasks,
                packed_stroked=packed_stroked,
                raster_pixels=raster_pixels,
                region_stage="packed-stroked-vector-text",
                region_boxes=region_boxes,
            )

        fallback_region = internal_full_stroked_vector_text_raster(
            self.capture,
            ocr_pass.scale,
            max_pixels=ocr_pass.pixel_budget,
            report=self.report,
        )
        tasks, raster_pixels = internal_region_tasks(
            fallback_region,
            ocr_pass,
            compact_image=self.compact_image,
        )
        self.report.stroked_vector_packed = {
            "accepted": False,
            "cells": 0,
            "raster_pixels": 0,
            "unmapped_observations": 0,
            "fallback_used": bool(tasks),
        }
        return internal_OcrPassTasks(
            ocr_pass,
            tasks=tasks,
            raster_pixels=raster_pixels,
            region_stage="stroked-vector-text-fallback",
            region_boxes=(fallback_region.page_box,) if fallback_region is not None else (),
        )

    def internal_image_region_tasks(self, ocr_pass: OcrPass) -> internal_OcrPassTasks:
        regions = internal_page_image_regions(
            self.capture,
            minimum_area_ratio=0.02,
            max_pixels=ocr_pass.pixel_budget,
            maximum_axis_deviation=internal_OCR_IMAGE_REGIONS_MAX_AXIS_DEVIATION,
        )
        if not regions:
            fallback_scale = max(2.0, ocr_pass.scale)
            image_crop = internal_safe_image_crop(self.capture)
            raster = internal_rendered_page_raster(
                self.capture,
                fallback_scale,
                crop=image_crop,
                max_pixels=ocr_pass.pixel_budget,
                include_native_text=ocr_pass.include_native_text,
                report=self.report,
            )
            tasks, raster_pixels = internal_raster_tasks(
                raster,
                image_crop or self.page_box,
                ocr_pass,
                compact_image=self.compact_image,
            )
            return internal_OcrPassTasks(
                ocr_pass,
                tasks=tasks,
                raster_pixels=raster_pixels,
            )

        region_signals = tuple(
            (region, internal_raster_text_signal(region.raster.image)) for region in regions
        )
        image_text_preflight: tuple[dict[str, object], ...] = tuple(
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
                and sum(len(text.strip()) for text in self.capture.observations.text) < 15
            )
        )
        skipped_regions = tuple(
            region for region, signal in region_signals if not signal.likely_text
        )
        tasks = tuple(
            task
            for region in eligible_regions
            for task in internal_tile_tasks(
                region.raster,
                region.page_box,
                ocr_pass,
                compact_image=self.compact_image,
            )
        )
        return internal_OcrPassTasks(
            ocr_pass,
            tasks=tasks,
            raster_pixels=sum(
                region.raster.width * region.raster.height for region in eligible_regions
            ),
            skipped_raster_pixels=sum(
                region.raster.width * region.raster.height for region in skipped_regions
            ),
            image_text_preflight=image_text_preflight,
            skipped_region_boxes=tuple(region.page_box for region in skipped_regions),
            region_stage="direct-image-regions",
            region_boxes=tuple(region.page_box for region in eligible_regions),
        )

    def internal_page_tasks(self, ocr_pass: OcrPass) -> internal_OcrPassTasks:
        direct_region = (
            self.internal_dominant_image_region_cached(ocr_pass.pixel_budget)
            if internal_direct_scan_allowed(self.capture, self.plan)
            else None
        )
        raster = direct_region.raster if direct_region is not None else None
        raster_page_box = direct_region.page_box if direct_region is not None else self.page_box
        if raster is None:
            raster = self.internal_rendered_raster_cached(ocr_pass)
            raster_page_box = self.page_box
        task_raster = (
            internal_adaptive_ocr_raster(raster)
            if raster is not None and ocr_pass.name == "adaptive-page"
            else raster
        )
        tasks = internal_raster_tasks(
            task_raster,
            raster_page_box,
            ocr_pass,
            compact_image=self.compact_image,
        )[0]
        return internal_OcrPassTasks(
            ocr_pass,
            tasks=tasks,
            raster_pixels=raster.width * raster.height if raster is not None else 0,
        )
