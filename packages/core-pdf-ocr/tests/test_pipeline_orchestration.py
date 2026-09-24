from copy import replace

import pytest

from core_pdf.impl.execution import ExtractionScope
from core_pdf.impl.extract.contracts import ObservationBatch
from core_pdf.impl.render.model import RasterImage
from core_pdf_ocr.impl.extract.contracts import (
    MAX_OCR_PIXELS,
    OcrPass,
    OcrPassScope,
    PageRoute,
    WorkPlan,
)
from core_pdf_ocr.impl.extract.ocr import pipeline
from core_pdf_ocr.impl.extract.ocr.session import OcrPassTasks
from core_pdf_ocr.impl.extract.ocr.types import (
    OcrTask,
    Raster,
    RasterRegion,
)
from core_pdf_ocr.impl.extract.quality import make_candidate


def make_result(text: str, *, height: float = 10, box=(0, 0, 20, 10), confidence=90):
    return make_candidate(
        6,
        ObservationBatch.from_columns((text,), (box,), source=1, confidence=(confidence,)),
        median_text_height=height,
    )


def make_task():
    return OcrTask(
        6, RasterImage(bytes([255]) * 10000, 100, 100, 1), (0, 0, 100, 100), (0, 0, 100, 100), 72
    )


@pytest.mark.parametrize(
    ("height", "text", "retry_text", "retry_box", "available", "expected"),
    [
        (0, "abc", "better result", (0, 0, 20, 10), True, ("better result",)),
        (10, "abc", "extra", (40, 0, 60, 10), True, ("abc", "extra")),
        (10, "abc", "a", (0, 0, 20, 10), True, ("abc",)),
        (10, "abc", "extra", (40, 0, 60, 10), False, ("abc",)),
        (
            20,
            "a useful sentence containing several words",
            "extra",
            (40, 0, 60, 10),
            True,
            ("a useful sentence containing several words", "extra"),
        ),
        (
            20,
            "a useful sentence containing several words",
            "extra",
            (40, 0, 60, 10),
            False,
            ("a useful sentence containing several words",),
        ),
    ],
)
def test_adaptive_retry_selects_scope_and_preserves_best_content(
    monkeypatch: pytest.MonkeyPatch,
    ocr_capture,
    height,
    text,
    retry_text,
    retry_box,
    available,
    expected,
) -> None:
    primary = make_result(text, height=height)
    retry = make_result(retry_text, box=retry_box)
    source = make_task()
    calls = []
    operation = OcrPass(
        "primary", OcrPassScope.PAGE, 2, (6,), adaptive_scale=True, pixel_budget=1000000
    )

    class Session:
        page_box = source.page_box

        def __init__(self, *args):
            self.recognitions = 0

        def materialize(self, ocr_pass, **kwargs):
            return OcrPassTasks(ocr_pass, (source,))

        def recognize_tasks(self, tasks):
            self.recognitions += 1
            return (primary if self.recognitions == 1 else retry,)

        def render_raster(self, scale, *, max_pixels, include_native_text):
            calls.append("page")
            assert scale == (3 if height == 0 else 5)
            assert max_pixels == MAX_OCR_PIXELS
            assert not include_native_text
            return Raster(source.image, 72) if available else None

        def high_resolution_weak_region_tasks(self, tasks, retry_pass, observations):
            calls.append("weak")
            assert tasks == (source,)
            assert observations is primary.observations
            assert retry_pass.scope is OcrPassScope.WEAK_REGIONS
            assert retry_pass.scale == 3.2
            assert retry_pass.pixel_budget == MAX_OCR_PIXELS
            assert (retry_pass.tiles, retry_pass.region_columns, retry_pass.max_regions) == (
                6,
                3,
                8,
            )
            return (source,) if available else ()

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    result = pipeline.recognize_page(
        ocr_capture,
        WorkPlan(PageRoute.OCR, ocr_passes=(operation,)),
        ExtractionScope(),
        stroked_profile=None,
    )
    assert result.observations.text == expected
    assert calls == (["weak"] if height == 20 else ["page"])


@pytest.mark.parametrize("verified", [False, True])
def test_hidden_layer_verification_short_circuits_only_after_matching_preview(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, verified: bool
) -> None:
    from core_pdf.impl.capture.program import CapturedProgram, PageProgram
    from core_pdf.impl.runs import TextRun

    texts = tuple(f"word{i}" for i in range(24))
    hidden = ObservationBatch.from_columns(
        texts,
        tuple((i * 20, 0, i * 20 + 15, 10) for i in range(24)),
        source=1,
        confidence=(100,) * 24,
    )
    preview = make_candidate(11, hidden if verified else ObservationBatch.empty())
    source = make_task()
    fallback = make_result("fallback")
    seen = []
    runs = tuple(
        TextRun(text, i * 20, 0, i * 20 + 15, 10, i * 20, 10, 10, 2, 0, 0, 0, visible=False)
        for i, text in enumerate(texts)
    )
    capture = replace(
        ocr_capture,
        observations=hidden,
        program=PageProgram(CapturedProgram(runs=runs)),
        evidence=replace(ocr_capture.evidence, full_page_image=True, image_filters=("JPXDecode",)),
    )

    class Session:
        page_box = source.page_box

        def __init__(self, capture, plan, compact, context, profile):
            assert compact == "grayscale"

        def materialize(self, operation, **kwargs):
            seen.append("fallback")
            return OcrPassTasks(operation, (source,))

        def recognize_tasks(self, tasks):
            if tasks[0].mode == 11:
                assert tasks[0].minimum_confidence == 80
                assert tasks[0].recognize_words
                seen.append("verify")
                return (preview,)
            return (fallback,)

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    monkeypatch.setattr(
        pipeline,
        "dominant_image_region",
        lambda *a, **k: RasterRegion(Raster(source.image, 72), source.page_box),
    )
    plan = WorkPlan(
        PageRoute.OCR,
        verify_hidden_text=True,
        ocr_passes=(OcrPass("regular", OcrPassScope.PAGE, 1, (6,)),),
    )
    result = pipeline.recognize_page(capture, plan, ExtractionScope(), stroked_profile=None)
    assert result.observations.text == (texts if verified else ("fallback",))
    assert seen == (["verify"] if verified else ["verify", "fallback"])


@pytest.mark.parametrize(
    ("populated", "cell_text", "replace_grid"),
    [(48, "replacement", True), (48, "x", False), (48, "", False), (6, "replacement", False)],
)
def test_ruled_table_retry_preserves_outside_text_and_rejects_worse_cell_reads(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, populated: int, cell_text: str, replace_grid: bool
) -> None:
    import numpy

    samples = numpy.full((400, 300, 1), 255, dtype=numpy.uint8)
    for x in range(30, 271, 40):
        samples[30:351, x] = 0
    for y in range(30, 351, 40):
        samples[y, 30:271] = 0
    for index in range(populated):
        row, column = divmod(index, 6)
        samples[40 + row * 40 : 44 + row * 40, 40 + column * 40 : 44 + column * 40] = 0
    source = OcrTask(
        6, RasterImage(samples.tobytes(), 300, 400, 1), (0, 0, 300, 400), (0, 0, 300, 400), 72
    )
    prior = make_candidate(
        6,
        ObservationBatch.from_columns(
            ("outside heading", "abcdefghij"),
            ((0, 380, 100, 390), (40, 330, 65, 340)),
            source=1,
            confidence=(95, 95),
        ),
    )
    cells = make_candidate(
        7,
        ObservationBatch.from_columns(
            (cell_text,) if cell_text else (),
            ((40, 330, 65, 340),) if cell_text else (),
            source=1,
            confidence=(95,) if cell_text else (),
        ),
    )
    calls = []

    class Session:
        page_box = source.page_box

        def __init__(self, *args):
            pass

        def materialize(self, operation, **kwargs):
            return OcrPassTasks(operation, (source,))

        def recognize_tasks(self, tasks):
            calls.append(tasks)
            if tasks[0].mode == 7:
                assert len(tasks) == populated
                assert all(task.image is source.image for task in tasks)
                return (cells,)
            return (prior,)

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    plan = WorkPlan(PageRoute.OCR, ocr_passes=(OcrPass("page", OcrPassScope.PAGE, 1, (6,)),))
    result = pipeline.recognize_page(ocr_capture, plan, ExtractionScope(), stroked_profile=None)
    assert len(calls) == (2 if populated >= 12 else 1)
    assert result.observations.text == (
        ("outside heading", "replacement") if replace_grid else prior.observations.text
    )
    if not replace_grid:
        assert result.observations is prior.observations


@pytest.mark.parametrize(
    ("accepted", "supplement_available"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_packed_vector_pipeline_remaps_seeds_and_chooses_isolated_or_full_fallback(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, accepted: bool, supplement_available: bool
) -> None:
    from core_pdf_ocr.impl.extract.ocr.strokes import StrokedTextDecode, StrokedTextObservation
    from core_pdf_ocr.impl.extract.ocr.types import (
        PackedStrokedTextRaster,
        StrokedTextCell,
    )

    source = make_task()
    raster = Raster(source.image, 72)
    packed = PackedStrokedTextRaster(
        raster,
        source.page_box,
        (StrokedTextCell((100, 200, 120, 210), (0, 0, 20, 10), (7,)),),
    )
    isolated = PackedStrokedTextRaster(
        raster,
        source.page_box,
        (StrokedTextCell((300, 200, 320, 210), (0, 0, 20, 10), (9,)),),
    )
    seed = make_result("AB")
    supplement = make_result(
        "7" if accepted else "full fallback",
        box=(0, 0, 20, 10) if accepted else (100, 200, 120, 210),
    )
    calls = []

    class Session:
        page_box = source.page_box

        def __init__(self, *args):
            self.count = 0

        def materialize(self, operation, **kwargs):
            return OcrPassTasks(operation, (source,), packed)

        def recognize_tasks(self, tasks):
            self.count += 1
            if self.count == 1:
                return (seed,)
            assert tasks[0].recognize_words is accepted
            if accepted:
                assert tasks[0].collect_symbols
                assert tasks[0].minimum_confidence == 50
            calls.append("supplement")
            return (supplement,)

    def decode(profile, observations, symbols):
        assert observations.text == ("AB",)
        assert observations.bbox.tolist() == [[100, 200, 120, 210]]
        return StrokedTextDecode(
            aligned_seeds=4 if accepted else 0,
            learned_signatures=8,
            observations=tuple(
                StrokedTextObservation("AB", (100, 200, 120, 210), i, i) for i in range(8)
            ),
        )

    def isolated_raster(*args, **kwargs):
        calls.append("isolated")
        assert kwargs["variant"] == "isolated"
        return isolated if supplement_available else None

    def full_raster(*args, **kwargs):
        calls.append("full")
        return RasterRegion(raster, source.page_box) if supplement_available else None

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    monkeypatch.setattr(pipeline, "decode_stroked_vector_text", decode)
    monkeypatch.setattr(pipeline, "stroked_vector_text_raster", isolated_raster)
    monkeypatch.setattr(pipeline, "full_stroked_vector_text_raster", full_raster)
    operation = OcrPass("vector", OcrPassScope.STROKED_VECTOR_TEXT, 1, (6,), recognize_words=True)
    result = pipeline.recognize_page(
        ocr_capture,
        WorkPlan(PageRoute.OCR, ocr_passes=(operation,)),
        ExtractionScope(),
        stroked_profile=None,
    )
    assert calls == (
        ["isolated" if accepted else "full"] + (["supplement"] if supplement_available else [])
    )
    if accepted and supplement_available:
        assert result.observations.text == ("AB", "7")
        assert result.observations.bbox.tolist() == [[100, 200, 120, 210], [300, 200, 320, 210]]
    elif supplement_available:
        assert result.observations.text == ("full fallback",)
    else:
        assert result.observations.text == ("AB",)


@pytest.mark.parametrize("unavailable", [False, True])
def test_pipeline_returns_empty_when_no_pass_materializes_tasks(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, unavailable: bool
) -> None:
    class Session:
        page_box = (0, 0, 100, 100)

        def __init__(self, *args):
            pass

        def materialize(self, operation, **kwargs):
            return None if unavailable else OcrPassTasks(operation)

        def recognize_tasks(self, tasks):
            pytest.fail("Unavailable or empty tasks must not invoke OCR")

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    plan = WorkPlan(PageRoute.OCR, ocr_passes=(OcrPass("empty", OcrPassScope.PAGE, 1, (6,)),))
    result = pipeline.recognize_page(ocr_capture, plan, ExtractionScope(), stroked_profile=None)
    assert not len(result.observations)


@pytest.mark.parametrize(
    ("complexity", "expected"), [(0, ("original", "additional text")), (180, ("additional text",))]
)
def test_page_augmentation_retains_original_text_except_for_complex_vector_pages(
    monkeypatch: pytest.MonkeyPatch, ocr_capture, complexity: int, expected: tuple[str, ...]
) -> None:
    source = make_task()
    original = make_result("original")
    extra = make_result("additional text", box=(40, 0, 80, 10))

    class Session:
        page_box = source.page_box

        def __init__(self, *args):
            self.count = 0

        def materialize(self, operation, **kwargs):
            return OcrPassTasks(operation, (source,))

        def recognize_tasks(self, tasks):
            self.count += 1
            return (original if self.count == 1 else extra,)

    monkeypatch.setattr(pipeline, "OcrSession", Session)
    capture = replace(
        ocr_capture, evidence=replace(ocr_capture.evidence, vector_complexity=complexity)
    )
    plan = WorkPlan(
        PageRoute.OCR,
        augment_page_candidates=True,
        ocr_passes=tuple(
            OcrPass(name, OcrPassScope.PAGE, 1, (6,)) for name in ("primary", "extra")
        ),
    )
    result = pipeline.recognize_page(capture, plan, ExtractionScope(), stroked_profile=None)
    assert result.observations.text == expected
