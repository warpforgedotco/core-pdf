# SPDX-License-Identifier: AGPL-3.0-only
"""OCR pass admission and selection state."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Self

from core_pdf.impl.extract.contracts import (
    OcrPass,
    OcrPassScope,
    RecognitionReport,
)
from core_pdf.impl.extract.ocr.candidates import (
    internal_augment_candidate,
    internal_candidate_timing_record,
    internal_record_candidates,
)
from core_pdf.impl.extract.ocr.rescue import internal_primary_text_is_sufficient
from core_pdf.impl.extract.ocr.types import internal_OcrTask
from core_pdf.impl.extract.quality import internal_Candidate

internal_PageBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class internal_OcrPassState:
    """Candidate selection and task provenance carried between OCR passes."""

    selected_name: str = ""
    selected: internal_Candidate | None = None
    selected_tasks: tuple[internal_OcrTask, ...] = ()
    previous_region_additions: int = 0
    seeded_region_selected: bool = False
    candidates: tuple[tuple[str, internal_Candidate], ...] = ()

    def prepare(
        self,
        ocr_pass: OcrPass,
        *,
        visible_native_characters: int,
    ) -> Self | None:
        """Return state prepared for ``ocr_pass``, or ``None`` when it should be skipped."""
        selected = self.selected
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and internal_primary_text_is_sufficient(selected)
        ):
            return None
        if (
            selected is not None
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= ocr_pass.run_if_characters_below
        ):
            return None
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.IMAGE_REGIONS
            and ocr_pass.run_if_characters_below is not None
            and selected.metrics.characters >= 28
            and selected.metrics.mean_confidence >= 97.0
        ):
            return None
        if (
            ocr_pass.run_if_additions_below is not None
            and self.previous_region_additions >= ocr_pass.run_if_additions_below
        ):
            return None
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_additions_below is not None
            and self.previous_region_additions == 0
            and selected is None
            and visible_native_characters >= 3_000
        ):
            return None
        if (
            ocr_pass.scope is OcrPassScope.WEAK_REGIONS
            and ocr_pass.run_if_additions_below is not None
            and self.previous_region_additions == 0
            and selected is not None
            and selected.metrics.characters >= 32
            and selected.metrics.mean_confidence >= 90.0
        ):
            return None
        if (
            ocr_pass.scope is OcrPassScope.PAGE
            and self.seeded_region_selected
            and ocr_pass.run_if_additions_below is not None
        ):
            return replace(
                self,
                selected_name="",
                selected=None,
                selected_tasks=(),
                seeded_region_selected=False,
            )
        return self

    def record_selection(self, report: RecognitionReport) -> None:
        """Mark the winning pass and expose the complete candidate history."""
        if self.selected is not None:
            for diagnostic in report.passes:
                diagnostic["selected"] = diagnostic["name"] == self.selected_name
        internal_record_candidates(self.candidates, self.selected_name, report)


@dataclass(frozen=True, slots=True)
class internal_OcrPassExecution:
    """Completed pass data needed for reporting and the selection transition."""

    ocr_pass: OcrPass
    candidate: internal_Candidate
    candidate_source_tasks: tuple[internal_OcrTask, ...]
    task_candidates: tuple[internal_Candidate, ...]
    tasks: tuple[internal_OcrTask, ...]
    started: float
    raster_pixels: int
    skipped_raster_pixels: int
    image_text_preflight: tuple[dict[str, object], ...]
    region_stage: str
    region_boxes: tuple[internal_PageBox, ...]
    skipped_region_boxes: tuple[internal_PageBox, ...]
    adaptive_retry_scale: float | None
    adaptive_preflight: dict[str, object] | None
    adaptive_rescue_decision: dict[str, object] | None
    adaptive_rescue: dict[str, object] | None

    def complete(
        self,
        state: internal_OcrPassState,
        report: RecognitionReport,
    ) -> internal_OcrPassState:
        """Record this pass and return the candidate-selection state for the next one."""
        ocr_pass = self.ocr_pass
        candidate = self.candidate
        selected = state.selected
        additions = 0
        used_native_seed = False
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

        next_state = replace(
            state,
            candidates=(*state.candidates, (ocr_pass.name, candidate)),
        )
        report.passes += (
            {
                "name": ocr_pass.name,
                "scope": ocr_pass.scope.value,
                "scale": ocr_pass.scale,
                "modes": ocr_pass.modes,
                "recognize_words": any(task.recognize_words for task in self.tasks),
                "character_confidence_threshold": ocr_pass.character_confidence_threshold,
                "task_count": len(self.tasks),
                "raster_pixels": self.raster_pixels,
                "skipped_raster_pixels": self.skipped_raster_pixels,
                "image_text_preflight": self.image_text_preflight,
                "region_stage": self.region_stage,
                "region_boxes": self.region_boxes,
                "skipped_region_boxes": self.skipped_region_boxes,
                "full_page_fallback": (
                    self.region_stage == "page" and ocr_pass.scope is OcrPassScope.PAGE
                ),
                "elapsed_seconds": time.perf_counter() - self.started,
                "render_timings": report.render_timings or {},
                **internal_candidate_timing_record(self.task_candidates),
                "accepted_additions": additions,
                "adaptive_retry_scale": self.adaptive_retry_scale,
                "adaptive_preflight": self.adaptive_preflight,
                "adaptive_rescue_decision": self.adaptive_rescue_decision,
                "adaptive_rescue": self.adaptive_rescue,
                "pixel_budget": ocr_pass.pixel_budget,
                "rectangles": tuple(task.rectangle for task in self.tasks),
                "selected": False,
                **candidate.metrics.as_record(),
            },
        )
        if not self.tasks:
            return next_state
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            next_state = replace(next_state, previous_region_additions=additions)
            if not additions:
                return next_state
            return replace(
                next_state,
                selected_name=ocr_pass.name,
                selected=candidate,
                selected_tasks=(*state.selected_tasks, *self.candidate_source_tasks),
                seeded_region_selected=used_native_seed and ocr_pass.seed_with_native,
            )
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            return replace(
                next_state,
                selected_name=ocr_pass.name,
                selected=candidate,
                selected_tasks=self.candidate_source_tasks,
            )
        return next_state
