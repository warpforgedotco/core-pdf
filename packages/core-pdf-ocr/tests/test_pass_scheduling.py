from copy import replace

import pytest

from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import OcrPass, OcrPassScope, PageAnalysis
from core_pdf_ocr.impl.extract.ocr.pipeline import OcrPassState
from core_pdf_ocr.impl.extract.ocr.types import OcrTask
from core_pdf_ocr.impl.extract.quality import Candidate, internal_candidate


def candidate(
    text: str = "recognized",
    *,
    utility: float = 100,
    confidence: float = 80,
    height: float = 10,
    x: float = 0,
) -> Candidate:
    result = internal_candidate(
        3,
        ObservationBatch.from_columns(
            (text,), ((x, 0, x + 20, 10),), source=1, confidence=(confidence,)
        ),
    )
    return replace(
        result, metrics=replace(result.metrics, utility=utility, median_text_height=height)
    )


def task(mode: int = 3) -> OcrTask:
    return OcrTask(mode, RasterImage(bytes(100), 10, 10, 1), (0, 0, 10, 10), (0, 0, 100, 100), 300)


@pytest.mark.parametrize(("utility", "replaced"), [(109, False), (110, False), (111, True)])
def test_replacement_requires_strict_improvement_and_tracks_only_winning_tasks(
    utility: float,
    replaced: bool,
) -> None:
    first, second = candidate(), candidate("replacement", utility=utility)
    original_tasks, new_tasks = (task(),), (task(6),)
    state = OcrPassState(first, original_tasks)
    result = state.complete(OcrPass("fallback", OcrPassScope.PAGE, 1, (6,)), second, new_tasks)
    assert result.selected is (second if replaced else first)
    assert result.selected_tasks == (new_tasks if replaced else original_tasks)


@pytest.mark.parametrize(
    ("scope", "text", "confidence", "height", "limit", "skipped"),
    [
        (OcrPassScope.PAGE, "abc", 99, 100, 100, True),
        (OcrPassScope.PAGE, "abc", 80, 10, 3, True),
        (OcrPassScope.PAGE, "abc", 80, 10, 4, False),
        (OcrPassScope.IMAGE_REGIONS, "a" * 28, 97, 10, 100, True),
        (OcrPassScope.IMAGE_REGIONS, "a" * 27, 97, 10, 100, False),
        (OcrPassScope.IMAGE_REGIONS, "a" * 28, 96, 10, 100, False),
    ],
)
def test_character_fallback_boundaries(
    scope: OcrPassScope, text: str, confidence: float, height: float, limit: int, skipped: bool
) -> None:
    state = OcrPassState(candidate(text, confidence=confidence, height=height))
    ocr_pass = OcrPass("fallback", scope, 1, (3,), run_if_characters_below=limit)
    assert (state.prepare(ocr_pass, visible_native_characters=0) is None) is skipped


@pytest.mark.parametrize(
    ("additions", "limit", "skipped"),
    [(0, 0, True), (1, 0, True), (0, 1, False), (3, 4, False), (4, 4, True)],
)
def test_region_addition_threshold(additions: int, limit: int, skipped: bool) -> None:
    state = OcrPassState(previous_region_additions=additions)
    ocr_pass = OcrPass("fallback", OcrPassScope.PAGE, 1, (3,), run_if_additions_below=limit)
    assert (state.prepare(ocr_pass, visible_native_characters=0) is None) is skipped


@pytest.mark.parametrize(("characters", "skipped"), [(2999, False), (3000, True)])
def test_native_text_can_suppress_empty_region_fallback(characters: int, skipped: bool) -> None:
    ocr_pass = OcrPass("fallback", OcrPassScope.PAGE, 1, (3,), run_if_additions_below=4)
    assert (
        OcrPassState().prepare(ocr_pass, visible_native_characters=characters) is None
    ) is skipped


def test_native_seed_is_discarded_before_full_page_fallback() -> None:
    seed = OcrPass("seed", OcrPassScope.WEAK_REGIONS, 1, (3,), seed_with_native=True)
    state = OcrPassState().complete(seed, candidate(), (task(),))
    assert state.seeded_region_selected
    assert state.previous_region_additions == 1
    fallback = OcrPass("fallback", OcrPassScope.PAGE, 1, (3,), run_if_additions_below=4)
    prepared = state.prepare(fallback, visible_native_characters=0)
    assert prepared is not None
    assert prepared.selected is None
    assert prepared.selected_tasks == ()
    assert not prepared.seeded_region_selected
    assert state.selected is not None


def test_region_augmentation_preserves_task_provenance_and_rejects_duplicates() -> None:
    ocr_pass = OcrPass("regions", OcrPassScope.WEAK_REGIONS, 1, (3,))
    first = candidate("Original text", confidence=99)
    source = (task(),)
    state = OcrPassState(first, source)
    duplicate = state.complete(ocr_pass, first, (task(6),))
    assert duplicate.selected is first
    assert duplicate.selected_tasks == source
    assert duplicate.previous_region_additions == 0
    extra_tasks = (task(6),)
    augmented = state.complete(
        ocr_pass, candidate("Additional text", confidence=99, x=100), extra_tasks
    )
    assert augmented.selected is not None
    assert tuple(augmented.selected.observations.text) == ("Original text", "Additional text")
    assert augmented.selected_tasks == (*source, *extra_tasks)
    assert augmented.previous_region_additions == 1


@pytest.mark.parametrize(
    ("characters", "confidence", "skipped"), [(31, 90, False), (32, 90, True), (32, 89, False)]
)
def test_weak_region_fallback_requires_unresolved_text(
    characters: int,
    confidence: float,
    skipped: bool,
) -> None:
    state = OcrPassState(candidate("a" * characters, confidence=confidence))
    ocr_pass = OcrPass("weak", OcrPassScope.WEAK_REGIONS, 1, (3,), run_if_additions_below=4)
    assert (state.prepare(ocr_pass, visible_native_characters=0) is None) is skipped


def test_vector_plan_contains_only_reachable_passes(ocr_capture: PageAnalysis) -> None:
    from core_pdf_ocr.impl.extract.observations import plan_page

    capture = replace(
        ocr_capture, evidence=replace(ocr_capture.evidence, uncovered_vector_area=30_000)
    )
    plan = plan_page(capture)
    assert [ocr_pass.name for ocr_pass in plan.ocr_passes] == ["schematic-regions", "primary-page"]
    assert all(
        ocr_pass.run_if_additions_below is None or ocr_pass.run_if_additions_below > 0
        for ocr_pass in plan.ocr_passes
    )


def test_pipeline_schedules_fallbacks_with_winning_task_provenance(
    ocr_capture: PageAnalysis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core_pdf.impl.runtime.execution import ExtractionScope
    from core_pdf_ocr.impl.extract.contracts import PageRoute, WorkPlan
    from core_pdf_ocr.impl.extract.ocr import pipeline
    from core_pdf_ocr.impl.extract.ocr.session import OcrPassTasks

    primary, fallback = candidate("Initial", utility=100), candidate("Improved", utility=120)
    primary_tasks, fallback_tasks = (task(3),), (task(6),)
    seen: list[tuple[str, Candidate | None, tuple[OcrTask, ...]]] = []

    class Session:
        page_box = (0, 0, 100, 100)

        def __init__(self, *args: object) -> None:
            pass

        def materialize(
            self,
            ocr_pass: OcrPass,
            *,
            selected: Candidate | None,
            selected_tasks: tuple[OcrTask, ...],
        ) -> OcrPassTasks:
            seen.append((ocr_pass.name, selected, selected_tasks))
            tasks = primary_tasks if ocr_pass.name == "primary" else fallback_tasks
            if ocr_pass.name == "empty-regions":
                tasks = ()
            return OcrPassTasks(ocr_pass, tasks)

        def recognize_tasks(self, tasks: tuple[OcrTask, ...]) -> tuple[Candidate, ...]:
            return (primary if tasks == primary_tasks else fallback,)

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    plan = WorkPlan(
        PageRoute.OCR,
        ocr_passes=(
            OcrPass("primary", OcrPassScope.PAGE, 1, (3,)),
            OcrPass("skipped", OcrPassScope.PAGE, 1, (6,), run_if_characters_below=7),
            OcrPass("fallback", OcrPassScope.PAGE, 1, (6,), run_if_characters_below=8),
            OcrPass("empty-regions", OcrPassScope.WEAK_REGIONS, 1, (6,)),
        ),
    )
    result = pipeline.recognize_page(ocr_capture, plan, ExtractionScope(), stroked_profile=None)
    assert result.observations is fallback.observations
    assert seen == [
        ("primary", None, ()),
        ("fallback", primary, primary_tasks),
        ("empty-regions", fallback, fallback_tasks),
    ]


@pytest.mark.parametrize("seed_with_native", [False, True])
def test_weak_region_materialization_uses_native_observations_only_when_enabled(
    ocr_capture: PageAnalysis,
    monkeypatch: pytest.MonkeyPatch,
    seed_with_native: bool,
) -> None:
    from core_pdf.impl.runtime.execution import ExtractionScope
    from core_pdf_ocr.impl.extract.contracts import PageRoute, WorkPlan
    from core_pdf_ocr.impl.extract.ocr import session
    from core_pdf_ocr.impl.extract.ocr.types import Raster, RasterRegion

    native = ObservationBatch.from_columns(("Native words",), ((0, 0, 20, 10),), source=0)
    capture = replace(ocr_capture, observations=native)
    source_task = task()
    region = RasterRegion(Raster(source_task.image, 300), source_task.page_box)
    ocr_pass = OcrPass(
        "weak",
        OcrPassScope.WEAK_REGIONS,
        1,
        (3,),
        region_first=False,
        seed_with_native=seed_with_native,
    )
    monkeypatch.setattr(session.OcrSession, "compose", lambda *args: object())
    monkeypatch.setattr(session, "dominant_image_region", lambda *args, **kwargs: region)
    seen = []

    def weak_tasks(
        raster: Raster,
        page_box: object,
        ocr_pass: OcrPass,
        observations: ObservationBatch,
        **kwargs: object,
    ) -> tuple[OcrTask, ...]:
        seen.append(observations)
        assert raster is region.raster
        assert page_box == region.page_box
        return (source_task,)

    monkeypatch.setattr(session, "weak_region_tasks", weak_tasks)
    owner = session.OcrSession(
        capture, WorkPlan(PageRoute.OCR, ocr_passes=(ocr_pass,)), True, ExtractionScope(), None
    )
    result = owner.materialize(ocr_pass, selected=None, selected_tasks=())
    if seed_with_native:
        assert result is not None
        assert result.tasks == (source_task,)
        assert seen == [native]
    else:
        assert result is None
        assert not seen
