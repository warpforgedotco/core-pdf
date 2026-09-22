# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import replace
from typing import Any, ClassVar, NoReturn, Self

import numpy

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.runtime.execution import ExtractionScope
from core_pdf_ocr.impl.extract.capture import promoted_hidden_observations
from core_pdf_ocr.impl.extract.contracts import (
    HIDDEN_TEXT_VERIFY_MIN_CONFIDENCE,
    HIDDEN_TEXT_VERIFY_PIXELS,
    MAX_OCR_PIXELS,
    PSM_SPARSE_TEXT,
    OcrPass,
    OcrPassScope,
    PageAnalysis,
    RecognitionResult,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.grids import (
    GRID_MIN_CELLS,
    detect_ruling_grid,
    grid_cell_tasks,
    grid_is_regular_table,
    grid_region_page_box,
    grid_row_observations,
)
from core_pdf_ocr.impl.extract.ocr.candidates import (
    augment_candidate,
    hidden_text_verification,
    merge_candidate_batches,
)
from core_pdf_ocr.impl.extract.ocr.regions import dominant_image_region
from core_pdf_ocr.impl.extract.ocr.rescue import (
    adaptive_rescue_decision,
    primary_text_is_sufficient,
)
from core_pdf_ocr.impl.extract.ocr.session import (
    OcrSession,
    raster_tasks,
    region_tasks,
)
from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextProfile
from core_pdf_ocr.impl.extract.ocr.types import OcrTask
from core_pdf_ocr.impl.extract.ocr.vector import (
    decode_stroked_vector_text,
    full_stroked_vector_text_raster,
    packed_stroked_vector_decode_gate,
    recover_stroked_vector_text,
    remap_stroked_vector_candidate,
    stroked_vector_text_raster,
)
from core_pdf_ocr.impl.extract.quality import Candidate

frozen_setattr = object.__setattr__


class OcrPassState:
    __slots__ = (
        "selected",
        "selected_tasks",
        "previous_region_additions",
        "seeded_region_selected",
    )

    selected: Candidate | None
    selected_tasks: tuple[OcrTask, ...]
    previous_region_additions: int
    seeded_region_selected: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "selected",
        "selected_tasks",
        "previous_region_additions",
        "seeded_region_selected",
    )
    __match_args__ = (
        "selected",
        "selected_tasks",
        "previous_region_additions",
        "seeded_region_selected",
    )

    def __init__(
        self,
        selected: Candidate | None = None,
        selected_tasks: tuple[OcrTask, ...] = (),
        previous_region_additions: int = 0,
        seeded_region_selected: bool = False,
    ) -> None:
        frozen_setattr(self, "selected", selected)
        frozen_setattr(self, "selected_tasks", selected_tasks)
        frozen_setattr(self, "previous_region_additions", previous_region_additions)
        frozen_setattr(self, "seeded_region_selected", seeded_region_selected)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"selected={self.selected!r}, "
            f"selected_tasks={self.selected_tasks!r}, "
            f"previous_region_additions={self.previous_region_additions!r}, "
            f"seeded_region_selected={self.seeded_region_selected!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.selected == other.selected
            and self.selected_tasks == other.selected_tasks
            and self.previous_region_additions == other.previous_region_additions
            and self.seeded_region_selected == other.seeded_region_selected
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.selected,
                self.selected_tasks,
                self.previous_region_additions,
                self.seeded_region_selected,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        selected = changes.pop("selected", self.selected)
        selected_tasks = changes.pop("selected_tasks", self.selected_tasks)
        previous_region_additions = changes.pop(
            "previous_region_additions", self.previous_region_additions
        )
        seeded_region_selected = changes.pop("seeded_region_selected", self.seeded_region_selected)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            selected,
            selected_tasks,
            previous_region_additions,
            seeded_region_selected,
        )

    def prepare(self, ocr_pass: OcrPass, *, visible_native_characters: int) -> OcrPassState | None:
        selected = self.selected
        if (
            selected is not None
            and ocr_pass.scope is OcrPassScope.PAGE
            and ocr_pass.run_if_characters_below is not None
            and primary_text_is_sufficient(selected)
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
                selected=None,
                selected_tasks=(),
                seeded_region_selected=False,
            )
        return self

    def complete(
        self,
        ocr_pass: OcrPass,
        candidate: Candidate,
        candidate_source_tasks: tuple[OcrTask, ...],
    ) -> OcrPassState:
        selected = self.selected
        if ocr_pass.scope is OcrPassScope.WEAK_REGIONS:
            used_native_seed = selected is None
            if selected is not None:
                candidate, additions = augment_candidate(
                    selected,
                    candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
            else:
                additions = len(candidate.observations)
            if not additions:
                return replace(self, previous_region_additions=0)
            return replace(
                self,
                previous_region_additions=additions,
                selected=candidate,
                selected_tasks=(*self.selected_tasks, *candidate_source_tasks),
                seeded_region_selected=used_native_seed and ocr_pass.seed_with_native,
            )
        if selected is None or candidate.metrics.utility > (
            selected.metrics.utility * ocr_pass.minimum_utility_gain
        ):
            return replace(
                self,
                selected=candidate,
                selected_tasks=candidate_source_tasks,
            )
        return self


def recognize_page(
    capture: PageAnalysis,
    plan: WorkPlan,
    context: ExtractionScope,
    *,
    stroked_profile: StrokedTextProfile | None,
) -> RecognitionResult:
    if not plan.ocr_passes:
        return RecognitionResult(ObservationBatch.empty())
    context.raise_if_cancelled()
    observations = recognize_page_with_reserved_raster(
        capture, plan, context, stroked_profile=stroked_profile
    )
    observations, alphabet = recover_stroked_vector_text(stroked_profile, observations)
    return RecognitionResult(observations, stroked_vector_alphabet=alphabet)


def recognize_page_with_reserved_raster(
    capture: PageAnalysis,
    plan: WorkPlan,
    context: ExtractionScope,
    *,
    stroked_profile: StrokedTextProfile | None,
) -> ObservationBatch:
    compact_image: bool | str = True
    if capture.evidence.full_page_image:
        image_filters = capture.evidence.image_filters
        if any("JPX" in str(filter_name).upper() for filter_name in image_filters):
            compact_image = "grayscale"
    session = OcrSession(
        capture,
        plan,
        compact_image,
        context,
        stroked_profile,
    )
    page_box = session.page_box
    pass_state = OcrPassState()
    adaptive_rescue_used = False

    if plan.verify_hidden_text:
        context.raise_if_cancelled()
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
        verification_region = dominant_image_region(
            capture,
            max_pixels=HIDDEN_TEXT_VERIFY_PIXELS,
        )
        verification_tasks = region_tasks(
            verification_region, verification_pass, compact_image=compact_image
        )
        verification_candidates = session.recognize_tasks(verification_tasks)
        verification_candidate = merge_candidate_batches(verification_candidates)
        if hidden_text_verification(
            capture.observations,
            verification_candidate.observations,
        ):
            return promoted_hidden_observations(capture)

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
        pass_tasks = session.materialize(
            ocr_pass,
            selected=selected,
            selected_tasks=selected_tasks,
        )
        if pass_tasks is None:
            continue
        ocr_pass = pass_tasks.ocr_pass
        tasks = pass_tasks.tasks
        packed_stroked = pass_tasks.packed_stroked
        if not tasks:
            continue

        candidate_source_tasks = tasks
        task_candidates = session.recognize_tasks(tasks)
        if packed_stroked is not None:
            remapped_with_counts = tuple(
                remap_stroked_vector_candidate(candidate, packed_stroked)
                for candidate in task_candidates
            )
            task_candidates = tuple(item[0] for item in remapped_with_counts)
            packed_candidate = merge_candidate_batches(task_candidates)
            packed_decode = decode_stroked_vector_text(
                stroked_profile,
                packed_candidate.observations,
                packed_candidate.symbols,
            )
            packed_accepted = packed_stroked_vector_decode_gate(
                packed_decode,
                len(packed_stroked.cells),
            )
            if packed_accepted:
                isolated_packed = stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    profile=stroked_profile,
                    max_pixels=ocr_pass.pixel_budget,
                    variant="isolated",
                )
                isolated_tasks = (
                    raster_tasks(
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
                    else ()
                )
                if isolated_tasks and isolated_packed is not None:
                    isolated_remapped = tuple(
                        remap_stroked_vector_candidate(
                            candidate,
                            isolated_packed,
                            digit_bearing_only=True,
                        )
                        for candidate in session.recognize_tasks(isolated_tasks)
                    )
                    isolated_candidates = tuple(item[0] for item in isolated_remapped)
                    task_candidates = (*task_candidates, *isolated_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *isolated_tasks)
                    tasks = (*tasks, *isolated_tasks)
                    packed_candidate = merge_candidate_batches(task_candidates)
            else:
                fallback_region = full_stroked_vector_text_raster(
                    capture,
                    ocr_pass.scale,
                    max_pixels=ocr_pass.pixel_budget,
                )
                fallback_tasks = region_tasks(
                    fallback_region,
                    replace(ocr_pass, recognize_words=False),
                    compact_image=compact_image,
                )
                if fallback_tasks:
                    fallback_candidates = session.recognize_tasks(fallback_tasks)
                    task_candidates = (*task_candidates, *fallback_candidates)
                    candidate_source_tasks = (*candidate_source_tasks, *fallback_tasks)
                    tasks = (*tasks, *fallback_tasks)
                    packed_candidate = merge_candidate_batches(fallback_candidates)
            candidate = packed_candidate
        else:
            candidate = merge_candidate_batches(task_candidates)
        if (
            selected is not None
            and plan.augment_page_candidates
            and ocr_pass.scope is OcrPassScope.PAGE
            and not capture.evidence.vector_complexity >= 180
        ):
            candidate, _ = augment_candidate(
                selected,
                candidate,
                minimum_confidence=70.0,
            )
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
            run_rescue = adaptive_rescue_decision(
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
            if retry_scope == "page":
                retry_raster = session.render_raster(
                    adaptive_retry_scale,
                    max_pixels=MAX_OCR_PIXELS,
                    include_native_text=ocr_pass.include_native_text,
                )
                retry_tasks = raster_tasks(
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
                retry_regions = session.high_resolution_weak_region_tasks(
                    tasks,
                    retry_pass,
                    candidate.observations,
                )
                retry_tasks = retry_regions
            if retry_tasks:
                candidate_source_tasks = (*candidate_source_tasks, *retry_tasks)
                retry_candidates = session.recognize_tasks(retry_tasks)
                retry_candidate = merge_candidate_batches(retry_candidates)
                augmented_candidate, _rescue_additions = augment_candidate(
                    candidate,
                    retry_candidate,
                    minimum_confidence=ocr_pass.minimum_confidence,
                )
                if retry_candidate.metrics.utility > augmented_candidate.metrics.utility * 1.05:
                    candidate = retry_candidate
                elif augmented_candidate.metrics.utility > candidate.metrics.utility:
                    candidate = augmented_candidate
        pass_state = pass_state.complete(ocr_pass, candidate, candidate_source_tasks)

    selected = pass_state.selected
    if selected is None:
        return ObservationBatch.empty()
    selected_tasks = pass_state.selected_tasks
    source_task = max(
        selected_tasks,
        key=lambda task: task.rectangle[2] * task.rectangle[3],
    )
    grid = detect_ruling_grid(source_task.image)
    if grid is not None and grid_is_regular_table(grid, selected.observations, source_task):
        x_lines, y_lines, source_samples, slope = grid
        cell_tasks = grid_cell_tasks(source_task, x_lines, y_lines, source_samples, slope)
        if len(cell_tasks) >= GRID_MIN_CELLS:
            cell_candidate = merge_candidate_batches(session.recognize_tasks(cell_tasks))
            cell_observations = grid_row_observations(cell_candidate.observations)
            if len(cell_observations):
                grid_box = grid_region_page_box(source_task, x_lines, y_lines)
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
                    return selected.observations
                retained = prior.take(numpy.flatnonzero(outside))
                return ObservationBatch.concatenate(retained, cell_observations)
    return selected.observations
